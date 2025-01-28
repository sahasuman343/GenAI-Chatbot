#Personal WOrk

from langchain_community.vectorstores import Chroma
import chromadb
from chromadb import Settings
import uuid
from utils.all_utils import read_config
from langchain_community.embeddings import AzureOpenAIEmbeddings
import os

class CustomOpenAIEmbeddings(AzureOpenAIEmbeddings):

    def __init__(self, openai_api_key, *args, **kwargs):
        super().__init__(openai_api_key=openai_api_key, *args, **kwargs)
        
    def _embed_documents(self, texts):
        return super().embed_documents(texts)  # <--- use OpenAIEmbedding's embedding function

    def __call__(self, input):
        return self._embed_documents(input)    # <--- get the embeddings

class VectorDB:
    def __init__(self, 
                 config_path = "./config/config.yaml"):
        self.config_path = config_path
        self.config = read_config(config_path)
        self.partition_config = self.config["partitioning"]
        
        
    def chormaDB_local(self, chunks, save_to_disk=False):
        """
        Creates a local Chroma vector store from a collection of documents.

        This function uses the Chroma.from_documents method to create a vector store from a collection of documents. 
        The collection name, embedding model, and persist directory are specified in the chorma_db_local configuration. 
        If save_to_disk is True, the vector store is persisted to disk.

        Args:
            chunks (list): The collection of documents to create the vector store from.
            save_to_disk (bool, optional): Whether to persist the vector store to disk. Default is False.

        """
        
        self.chormaDB_local_config = self.config["chorma_db_local"]
        vectorstore = Chroma.from_documents(collection_name=self.chormaDB_local_config['collection_name'],
                                   documents=chunks,
                                   embedding=AzureOpenAIEmbeddings(azure_deployment=self.chormaDB_local_config['embedding_model'],
                                                                           openai_api_version="2023-05-15"),
                                   persist_directory=self.chormaDB_local_config['persist_directory'],
                                   )
        if save_to_disk:
            vectorstore.persist()
            
    def chormaDB_server(self, chunks, save_to_disk=False):
        """
        Creates or updates a Chroma vector store on a server from a collection of documents.

        This function uses the ChromaDB HttpClient to interact with a Chroma server. It retrieves the specified collection from the server, and adds the provided documents to it. Each document is assigned a unique ID and its metadata is stored. The embedding function used is CustomOpenAIEmbeddings.

        Args:
            chunks (list): The collection of documents to add to the vector store.
            save_to_disk (bool, optional): This argument is not used in the current implementation. Default is False.

        Environment Variables:
            OPENAI_API_KEY (str): The API key for OpenAI.
            CHROMA_HOST (str): The host address of the Chroma server.
            CHROMA_PORT (str): The port number of the Chroma server.
            CHROMA_COLLECTION_NAME (str): The name of the collection to retrieve from the Chroma server.
        """
        
        emb_fn = CustomOpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])  
        client = chromadb.HttpClient(host=os.environ['CHROMA_HOST'], 
                                     port=os.environ['CHROMA_PORT'], 
                                     settings=Settings(allow_reset=True, anonymized_telemetry=False))
        
        collections = client.list_collections()
        collections = [collection.name for collection in collections]
        collection_name = os.environ["CHROMA_COLLECTION_NAME"]
        collection = client.get_collection(name=os.environ['CHROMA_COLLECTION_NAME'], embedding_function=emb_fn)
        for doc in chunks:
            collection.add(
                ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
            )
        
        