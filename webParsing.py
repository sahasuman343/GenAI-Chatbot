from langchain_unstructured import UnstructuredLoader
import tracemalloc

tracemalloc.start() 
page_url = 'https://www.enx.com/handbook/tisax-participant-handbook.html'
loader = UnstructuredLoader(web_url=page_url)

def load_web(loader = loader):
    docs = []
    for doc in loader.lazy_load():
        print(doc)
        docs.append(doc)
    return docs

if __name__=="__main__":
    docs = load_web(loader)
    