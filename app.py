
import os
import time
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains import LLMChain
from langchain import hub
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import chromadb
import ollama

import streamlit as st

from utils.utils import *


ollama.pull("nomic-embed-text")
load_dotenv()

# # Process data
# print("loading pdf.....")
# pdf_loc = "data/TISAX Participant Handbook.pdf"
# elements = process_pdf(pdf_path=pdf_loc)
# docs = partition_metadata(raw_elements=elements, file_id= "123",topic="TISAX", category_id= "12345")
# print("pdf loaded successfully......")
# # Create Vectorstore
# vectorstore = Chroma.from_documents(documents=docs, embedding=OllamaEmbeddings(model="nomic-embed-text"))

vectorstore = Chroma(persist_directory="./vectordb/", collection_name="TISAX", embedding_function=OllamaEmbeddings(model="nomic-embed-text"))

# Define LLM
llm = ChatGroq(temperature=0, model_name=os.getenv("MODEL_NAME"), groq_api_key=os.getenv("GROQ_API_KEY"))

#Define RAG Chain
retriever = vectorstore.as_retriever()
# Load the Contextualize prompt from a file
with open(r'system_prompt.txt', 'r') as file:
    contextualize_q_system_prompt = file.read().strip()

contextualize_q_prompt = ChatPromptTemplate.from_messages([
SystemMessagePromptTemplate.from_template(contextualize_q_system_prompt),
MessagesPlaceholder(variable_name="chat_history"),
HumanMessagePromptTemplate.from_template("{input}")
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


### Answer question ###
qa_system_prompt = """You are a helpful assistant for Daimler truck cyber security department. You can answer the user query from the given context below:

{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
    ])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Statefully manage chat history ###
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(rag_chain, get_session_history, input_messages_key="input", history_messages_key="chat_history",output_messages_key="answer")



def process(text):
    start_time = time.time()

    # Go get the response from the LLM
    response = conversational_rag_chain.invoke({"input": text}, config={"configurable": {"session_id": "abc123"}})
    # response = self.conversation.invoke({"text": text})
    end_time = time.time()

    elapsed_time = int((end_time - start_time) * 1000)
    #print(f"LLM ({elapsed_time}ms): {response['answer']}")
    return response["answer"]

def main():
    while True:
        #print(f"history: {llm.store}")
        qs = input("Human: ")
        llm_response = process(qs)
        print(f"Assistant: {llm_response}")

if __name__ == "__main__":
    main()