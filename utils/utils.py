import yaml
import os
import time
from dotenv import load_dotenv

from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.cleaners.core import clean
from langchain.schema.document import Document

def read_config(config_file):
    """
    Reads a YAML configuration file.

    Args:
        config_file: Path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration data.
    """

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def process_pdf(pdf_path:str):
    '''
    '''
    elements = partition_pdf(filename=pdf_path,
                         extract_images_in_pdf=False,
                         extract_image_block_output_dir="data/images",
                         extract_image_block_to_payload = False,
                         infer_table_structure= True,
                         chunking_strategy = "by_title"
                         )
    comp_elements = [el for el in elements if el.category =="CompositeElement"]
    table_elements = [el for el in elements if el.category =="Table"]

    for el in comp_elements:
        el.metadata.orig_elements = [e for e in el.metadata.orig_elements if e.category != "Image"]
    
    return comp_elements + table_elements

def partition_metadata(raw_elements, file_id=None,topic=None, category_id=None):
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
                
        return text_elements + table_elements


if __name__ == "__main__":
    config_path = "path/to/your/config.yml"  # Replace with the actual path
    config_data = read_config(config_path)
    print(config_data)