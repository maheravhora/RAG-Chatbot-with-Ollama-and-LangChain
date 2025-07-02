# chatbot api using langchain with Ollama LLM
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv

# ---------------------- Load .env variables ----------------------
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# ---------------------- FastAPI App ----------------------
app = FastAPI(
    title="LangChain Mistral Chatbot with FastAPI",
    description="A FastAPI service using LangChain, Ollama (Mistral), and LangSmith tracing.",
    version="1.0",
)

# ---------------------- LangChain Chain ----------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}"),
    ]
)

llm = OllamaLLM(model="mistral")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# ---------------------- Pydantic Model ----------------------
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# ---------------------- API Endpoint ----------------------
@app.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    answer = chain.invoke({"question": query.question})
    return QueryResponse(answer=answer)
