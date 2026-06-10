
# 🚀 LangGraph Multi-Agent Chatbot

Advanced Agentic AI chatbot using:
- LangGraph
- LangChain
- Streamlit
- RAG
- Tool Calling
- Multi-Agent Architecture
- LangSmith
- HITL
- Retry/Fault Tolerance

## Run Steps

### Create virtual environment
python -m venv venv

### Activate
Windows:
venv\Scripts\activate

Linux/Mac:
source venv/bin/activate

### Install packages
pip install -r requirements.txt

### Create .env file
OPENAI_API_KEY=your_openai_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=LangGraph-BASIC-Chatbot

### Run app
streamlit run app.py
