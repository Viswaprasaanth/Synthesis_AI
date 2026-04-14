from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from app.config import get_settings

_llm = None


def get_llm():
    """Return a reusable ChatGroq instance."""
    global _llm
    if _llm is None:
        s = get_settings()
        _llm = ChatGroq(
            model=s.llm_model,
            api_key=s.groq_api_key,
            temperature=0.1,
        )
    return _llm


def build_chain(system_msg: str, human_msg: str):
    """Build a LangChain prompt → LLM → string output chain."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", human_msg),
    ])
    return prompt | get_llm() | StrOutputParser()