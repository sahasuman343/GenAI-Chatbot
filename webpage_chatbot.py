from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
import ollama
ollama.pull("llama3.2")
ollama.pull("nomic-embed-text")

# Load the webpage
loader = WebBaseLoader("https://www.enx.com/handbook/tisax-participant-handbook.html")
documents = loader.load()

# Split the text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create Ollama embeddings
embeddings = OllamaEmbeddings(model = "nomic-embed-text")

# Create Chroma vectorstore
vectordb = Chroma(
    collection_name="TISAX_bot_data", 
    embedding_function=embeddings, 
    persist_directory="." 
)
vectordb.add_documents(texts)

# Initialize Ollama LLM
llm = OllamaLLM(model_name="llama3.2") 

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=vectordb.as_retriever(n_results=3) 
)