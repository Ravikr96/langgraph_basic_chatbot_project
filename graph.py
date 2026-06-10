
from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from rag import build_retriever
from tools import calculator
from tenacity import retry, stop_after_attempt, wait_fixed

llm = ChatOpenAI(model="gpt-4o-mini")

retriever = build_retriever()

class MCPState(TypedDict):

    user_query: str
    route: Optional[str]

    rag_response: Optional[str]
    tool_response: Optional[str]
    planner_response: Optional[str]

    final_response: Optional[str]

def controller_node(state: MCPState):

    query = state["user_query"]

    prompt = f"""
    Decide which agent should handle this query.

    Options:
    - rag
    - tool
    - planner

    Query:
    {query}
    """

    response = llm.invoke(prompt).content.lower()

    return {
        "route": response
    }

def router(state: MCPState) -> Literal[
    "rag_node",
    "tool_node",
    "planner_node"
]:

    route = state["route"]

    if "tool" in route:
        return "tool_node"

    elif "planner" in route:
        return "planner_node"

    else:
        return "rag_node"

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def rag_node(state: MCPState):

    query = state["user_query"]

    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer using this context:

    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt).content

    return {
        "rag_response": response
    }

def tool_node(state: MCPState):

    query = state["user_query"]

    expression = query.replace("calculate", "").strip()

    result = calculator.invoke(expression)

    return {
        "tool_response": result
    }

def planner_node(state: MCPState):

    query = state["user_query"]

    prompt = f"""
    Create step-by-step plan for:

    {query}
    """

    response = llm.invoke(prompt).content

    return {
        "planner_response": response
    }

def summarizer_node(state: MCPState):

    response = f"""
    RAG RESPONSE:
    {state.get('rag_response', '')}

    TOOL RESPONSE:
    {state.get('tool_response', '')}

    PLANNER RESPONSE:
    {state.get('planner_response', '')}
    """

    final = llm.invoke(
        f"Create final helpful response:\n{response}"
    ).content

    return {
        "final_response": final
    }

graph = StateGraph(MCPState)

graph.add_node("controller", controller_node)

graph.add_node("rag_node", rag_node)

graph.add_node("tool_node", tool_node)

graph.add_node("planner_node", planner_node)

graph.add_node("summarizer", summarizer_node)

graph.add_edge(START, "controller")

graph.add_conditional_edges(
    "controller",
    router
)

graph.add_edge("rag_node", "summarizer")

graph.add_edge("tool_node", "summarizer")

graph.add_edge("planner_node", "summarizer")

graph.add_edge("summarizer", END)

workflow = graph.compile()
