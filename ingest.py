from langchain.document_loaders.pdf import PDFMinerLoader
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
import chromadb

client = chromadb.PersistentClient('db')
def main():
    pdf_loader = DirectoryLoader(path = 'docs',
                                 glob='*.pdf',
                                 loader_cls=PyPDFLoader)
    documents = pdf_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=50000)
    split_documents = splitter.split_documents(documents)
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs={'device': 'cpu'})
    db = Chroma.from_documents(documents = split_documents, embedding = embeddings,
                          persist_directory='db')
    db.persist()
    db = None

if __name__ == '__main__':
    main()