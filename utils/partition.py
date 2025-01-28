#Personal Work


from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.cleaners.core import clean
from langchain.schema.document import Document
from tqdm import tqdm
import sys
from utils.all_utils import read_config
import argparse
import os
import logging

logger = logging.getLogger(__name__)

class data_partition:
    """
    A class used to partition data from .docx and .pdf files.

    This class reads a configuration file and uses it to partition data from .docx and .pdf files. The partitioning strategy and parameters are determined by the configuration file. The class supports processing of individual files or all files in a directory.

    Attributes
    ----------
    config : dict
        The configuration dictionary read from the config file.
    """
    def __init__(self, 
                 config_path = "./config/config.yaml"):
        self.config = read_config(config_path)
        self.partition_config = self.config["partitioning"]
        

    def docx_process(self, file=None):     
        """
        Processes a .docx file, extracting elements based on the partition configuration.

        This function uses the partition_docx function to extract elements from the .docx file. The extraction strategy
        and parameters are determined by the partition_config attribute.

        Args:
            file_id (str, optional): The path to the .docx file to process. If None, the function processes the file
                provided in the file argument. Default is None.
            file (file-like object, optional): A file-like object to process. If None, the function processes the file
                at the location specified by file_id. Default is None.

        Returns:
            list: A list of elements extracted from the .docx file.
        """      

        raw_docx_elements = partition_docx(
            file = file,
            strategy= self.partition_config['strategy'],  
            extract_images_in_pdf= self.partition_config['extract_images'],
            infer_table_structure= self.partition_config['infer_table'],  
            chunking_strategy= self.partition_config['chunking_strategy'],  
            max_characters= self.partition_config['max_characters'],  
            new_after_n_chars= self.partition_config['new_after_n_chars'],  
            combine_text_under_n_chars= self.partition_config['combine_text_under_n_chars'],  
            image_output_dir_path= self.partition_config['image_output_dir_path'],
        )
        return raw_docx_elements
    
    def pdf_process(self, file): 
        raw_pdf_elements = partition_pdf(
            file = file,
            strategy= self.partition_config['strategy'],  
            extract_images_in_pdf= self.partition_config['extract_images'],
            # extract_image_block_types=["Image", "Table"],  
            infer_table_structure= self.partition_config['infer_table'],  
            chunking_strategy= self.partition_config['chunking_strategy'],  
            max_characters= self.partition_config['max_characters'],  
            new_after_n_chars= self.partition_config['new_after_n_chars'],  
            combine_text_under_n_chars= self.partition_config['combine_text_under_n_chars'],  
            image_output_dir_path= self.partition_config['image_output_dir_path'],
        )
        return raw_pdf_elements 
    
    def process(self, file_id=None, file=None,topic=None, category_id=None):
        """
        Processes the given file or files in the given directory, extracting text and table elements.

        This function supports .docx and .pdf files. For each file, it extracts "CompositeElement" and "Table" elements,
        cleans them, and stores them as Document objects in the text_elements and table_elements lists respectively.

        Args:
            data_path (str, optional): The path to the directory containing the files to process. If None, the function
                uses the 'data' field from the partition_config attribute. Default is None.
            file (file-like object, optional): A file-like object to process. If None, the function processes all files
                in the directory specified by data_path. Default is None.

        Returns:
            list: A list of Document objects representing the text and table elements extracted from the files.

        Raises:
            ValueError: If a file type is not supported.
        """
        
        try:
            logger.info("\nprocessing file from Stored directory:")

            if file_id.endswith('.docx'):
                raw_docx_elements = self.docx_process(file=file)
                elements = self.partition_metadata(raw_elements = raw_docx_elements, 
                                                   file_id = file_id,
                                                   category_id = category_id)
                
                return elements

            elif file_id.endswith('.pdf'):
                raw_pdf_elements = self.pdf_process(file=file)
                elements = self.partition_metadata(raw_elements = raw_pdf_elements, 
                                                   file_id = file_id,
                                                   category_id = category_id)
                
                return elements

                
            else:
                raise ValueError("File type not supported")
                
        except Exception as e:
            logger.exception("Error in data_partition : %s", e)
            pass

    
    def partition_metadata(self, raw_elements, file_id=None,topic=None, category_id=None):
        text_elements=[]
        table_elements=[]
        for element in raw_elements:
            if "unstructured.documents.elements.CompositeElement" in str(type(element)):
                metadata = {key:str(value) for key, value in element.metadata.to_dict().items()}
                metadata.update({'type':'text','file_id':file_id,'category_id':category_id})
                text_elements.append(Document(page_content=str(element),metadata=metadata))
            elif "unstructured.documents.elements.Table" in str(type(element)):
                metadata = {key:str(value) for key, value in element.metadata.to_dict().items()}
                metadata.update({'type':'table','file_id':file_id, 'category_id':category_id})
                ele = clean(element.metadata.text_as_html, extra_whitespace=True)
                table_elements.append(Document(page_content=ele,metadata=metadata))
                
        logger.info("\ntext_elements: %s,\ntable_elements: %s", len(text_elements), len(table_elements))
        return text_elements + table_elements
        
    
if __name__ == '__main__':
    obj = data_partition()
    obj.process(data_path=None)