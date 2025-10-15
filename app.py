import streamlit as st
from google import genai
from google.genai import Client
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    Tool,
)

# from dotenv import load_dotenv
import os
from pydantic import BaseModel

# === Your existing setup (AgentState, prompts, nodes, should_continue, etc.) ===

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay.\
write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the section."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""


class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revision: int


class Queries(BaseModel):
    queries: List[str]


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.3,
    google_api_key=GEMINI_API_KEY,
)

genai_client = Client(api_key=GEMINI_API_KEY)


def plan_node(state: AgentState):
    messages = [SystemMessage(content=PLAN_PROMPT), HumanMessage(content=state["task"])]
    response = model.invoke(messages)
    print(response)
    return {"plan": response.content}


def research_plan_node(state: AgentState):
    rresponse = genai_client.models.generate_content(
        model="gemini-2.5-pro",
        contents=state["task"],
        config=GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Queries,
            system_instruction=PLAN_PROMPT,
        ),
    )
    # # Use the response as a JSON string.
    # print(response.text)

    # Use instantiated objects.
    my_recipes: Queries = rresponse.parsed

    content = state.get("content", [])
    queries = my_recipes.queries
    for query in queries:

        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{query}",
            config=GenerateContentConfig(
                tools=[
                    # Use Google Search Tool
                    Tool(google_search=GoogleSearch())
                ],
            ),
        )
        content.append(response.text)
    return {"content": content}


def generation_node(state: AgentState):
    content = "\n\n".join(state["content"] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
    )
    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        user_message,
    ]
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }


def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state["draft"]),
    ]
    response = model.invoke(messages)
    return {"critique": response.content}


def research_critique_node(state: AgentState):

    rresponse = genai_client.models.generate_content(
        model="gemini-2.5-pro",
        contents=state["critique"],
        config=GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Queries,
            system_instruction=RESEARCH_CRITIQUE_PROMPT,
        ),
    )

    # Use instantiated objects.
    my_recipes: Queries = rresponse.parsed
    query_list = my_recipes.queries

    content = state.get("content", [])

    for query in query_list:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{query}",
            config=GenerateContentConfig(
                tools=[
                    # Use Google Search Tool
                    Tool(google_search=GoogleSearch())
                ],
            ),
        )
        content.append(response.text)
    return {"content": content}


def should_continue(state):
    if state["revision_number"] > state["max_revision"]:
        return END
    return "reflect"


# === Build the graph ===
builder = StateGraph(AgentState)

builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)
builder.set_entry_point("planner")

builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_conditional_edges(
    "generate", should_continue, {END: END, "reflect": "reflect"}
)
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

graph = builder.compile()


# === Streamlit App ===
def main():
    st.title("📚 Essay Writing Agent")

    task = st.text_area(
        "Enter your essay topic:", placeholder="e.g. The impact of AI on healthcare"
    )

    max_revisions = st.slider("Max revisions", min_value=1, max_value=5, value=3)

    if st.button("Generate Essay"):
        if not task.strip():
            st.warning("Please enter a topic first.")
        else:
            # Initialize state
            state: AgentState = {
                "task": task,
                "plan": "",
                "draft": "",
                "critique": "",
                "content": [],
                "revision_number": 1,
                "max_revision": max_revisions,
            }

            report = {}
            # steps = ["planner", "research_plan", "generate", "reflect", "research_critique"]

            # Run the agent graph
            with st.spinner("Researching and drafting..."):
                # completed_steps = set()

                # progress_area = st.empty()

                for step in graph.stream(state):
                    step_name = list(step.keys())[0]
                    # completed_steps.add(step_name)
                    # # Build progress display
                    # progress_markdown = "### 🧩 Progress\n"
                    # for s in steps:
                    #     if s in completed_steps:
                    #         progress_markdown += f"- ✅ **{s}**\n"
                    #     else:
                    #         progress_markdown += f"- ⏳ {s}\n"
                    # progress_area.markdown(progress_markdown)
                    st.write(f"### 🔹 Step: {step_name} ✅")
                    # st.write(step)
                    report.update(step)

            # Final draft
            st.subheader("📄 Final Draft")
            st.write(report["generate"]["draft"])


if __name__ == "__main__":
    main()
