need setup 
pip install ollama 
pip install openai 
pip install chromadb 
pip install sentence-transformers 
pip install langgraph 
pip install langchain 
pip install pydantic
pip install python-dotenv

General ARCHITECTURE:

User question
    ↓
LangGraph Router Decides tool:

- knowledge_base
- calculator
- date
- general_llm
    ↓
Tool runs
    ↓
LLM gives final answer
