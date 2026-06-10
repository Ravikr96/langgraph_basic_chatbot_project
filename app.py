
import streamlit as st
from graph import workflow

st.set_page_config(
    page_title="LangGraph MCP Chatbot",
    page_icon="🤖"
)

st.title("🤖 LangGraph MCP Multi-Agent Chatbot")

query = st.text_input("Ask your question")

if st.button("Submit"):

    if query:

        state = {
            "user_query": query
        }

        result = workflow.invoke(state)

        response = result["final_response"]

        st.success(response)

        st.subheader("Human-in-the-Loop Review")

        edited = st.text_area(
            "Edit response if needed",
            value=response,
            height=200
        )

        approved = st.checkbox("Approve Response")

        if approved:
            st.success("Response approved.")
        else:
            st.warning("Response not approved.")
