
# Personal Work



from langchain_community.embeddings import AzureOpenAIEmbeddings
from utils.all_utils import read_config
from data_ingestion.partition import data_partition
import os
import json
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class data_chunking:
    def __init__(self, 
                 config_path = "./config/config.yaml"):
        self.config_path = config_path
        self.config = read_config(config_path)
        self.partition_config = self.config["partitioning"]
        
    def semantic_chunking(self, elements):
        '''
        Split text,table and image elements into semantic chunks
        Issue with semantic Chunking: We'll loose metadata information, if we combine all text elements'''
        semantic_chunk_config = self.config["semantic_chunking"]
        elements = None
        return elements
    
    def partition_default_chunks(self, elements):
        logger.info("\ndefault_chunks :%s", len(elements))
        return elements
    
if __name__ == '__main__':
    obj = data_chunking()
    obj.partition_default_chunks(text_elements=None)
    

