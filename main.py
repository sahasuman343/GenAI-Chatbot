import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
import ollama
import json
from flask import Flask

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
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()
ollama.pull("nomic-embed-text")

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name=os.getenv("MODEL_NAME"), groq_api_key=os.getenv("GROQ_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", openai_api_key=os.getenv("OPENAI_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125", openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        #construct the retriever
        self.loader = TextLoader("data/data.txt")
        self.documents = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.splits = self.text_splitter.split_documents(self.documents)

        self.vectorstore = Chroma.from_documents(documents=self.splits, embedding=OllamaEmbeddings(model="nomic-embed-text"))
        self.retriever = self.vectorstore.as_retriever()

        # Load the Contextualize prompt from a file
        with open(r'system_prompt.txt', 'r') as file:
            contextualize_q_system_prompt = file.read().strip()
        
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
        ])

        self.history_aware_retriever = create_history_aware_retriever(self.llm, self.retriever, self.contextualize_q_prompt)


        ### Answer question ###
        qa_system_prompt = """You are a helpful assistant for Daimler truck cyber security department. You can answer the user query from the given context below:

        {context}"""
        
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
            ])
        
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

        ### Statefully manage chat history ###
        store = {}
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        self.conversational_rag_chain = RunnableWithMessageHistory(self.rag_chain, get_session_history, input_messages_key="input", history_messages_key="chat_history",output_messages_key="answer")

        # self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # self.conversation = LLMChain(
        #     llm=self.llm,
        #     prompt=self.prompt,
        #     memory=self.memory
        # )
        
    def process(self, text):
            start_time = time.time()

            # Go get the response from the LLM
            response = self.conversational_rag_chain.invoke({"input": text}, config={"configurable": {"session_id": "abc123"}})
            # response = self.conversation.invoke({"text": text})
            end_time = time.time()

            elapsed_time = int((end_time - start_time) * 1000)
            #print(f"LLM ({elapsed_time}ms): {response['answer']}")
            return response["answer"]
    

if __name__ == "__main__":
    llm = LanguageModelProcessor()
    while True:
        #print(f"history: {llm.store}")
        qs = input("Human: ")
        llm_response = llm.process(qs)
        print(f"Assistant: {llm_response}")
        
            

