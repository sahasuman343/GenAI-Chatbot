#Personal Work



import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from utils.all_utils import read_config
import chromadb
from chromadb import Settings
from langchain.storage import InMemoryStore
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(r'.env')

logger = logging.getLogger(__name__)

class DataRetriever:
    def __init__(self, config_path="./config/config.yaml"):
        self.config_path = config_path
        self.config = read_config(config_path)
        self.db = None  # Initialize db instance variable

    def chroma_vector_store(self):
        if os.environ.get("CHROMA_HOST"):
            try:
                logger.info("Retrieving from server chroma db")
                emb_fn = AzureOpenAIEmbeddings(azure_deployment=os.environ["EMBEDDING_MODEL"],
                                               openai_api_version="2023-05-15")
                collection_name = os.environ["CHROMA_COLLECTION_NAME"]
                client = chromadb.HttpClient(host=os.environ["CHROMA_HOST"],
                                             port=os.environ["CHROMA_PORT"],
                                             settings=Settings(allow_reset=True, anonymized_telemetry=False))
                logger.info("client.heartbeat:%s", client.heartbeat())
            except Exception as e:
                logger.error(f'\n Error in connecting to chroma db server {os.environ["CHROMA_HOST"]} : {e}')
        else:
            try:
                logger.info("Retrieving from local chroma db")
                self.chormaDB_local_config = self.config["chorma_db_local"]
                emb_fn = AzureOpenAIEmbeddings(azure_deployment=self.chormaDB_local_config['embedding_model'],
                                               openai_api_version="2023-05-15")
                collection_name = self.chormaDB_local_config["collection_name"]

                client = chromadb.PersistentClient(path=self.chormaDB_local_config["persist_directory"])
                collections = client.list_collections()
                collections = [collection.name for collection in collections]
                print(f'\n collections : {collections}')
            except Exception as e:
                logger.error(f'\n Error in connecting to local chroma db : {e}')

        client.get_or_create_collection(name=collection_name)

        self.db = Chroma(client=client, collection_name=collection_name, embedding_function=emb_fn)
        
        # Return db instance
        return self.db

    def multivectorretrieval(self):
        store = InMemoryStore()
        vectorstore = self.chroma_vector_store()
        retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store)
        return retriever

# Create a singleton instance of DataRetriever
data_retriever_instance = DataRetriever()
