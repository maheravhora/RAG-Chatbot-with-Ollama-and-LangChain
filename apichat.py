# chatbot api using langchain with Ollama LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama  import OllamaLLM
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = "include your api key"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

os.environ["LANGCHAIN_TRACING_V2"]="true"

## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)
## streamlit framework

st.title('Langchain Chatbot With mistral API Using Streamlit')
input_text=st.text_input("Search the topic u want")

# ollama mistral LLm 
llm = OllamaLLM(model="mistral")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
