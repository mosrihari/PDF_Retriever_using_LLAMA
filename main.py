from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import CTransformers
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def create_custom_prompt():
    custom_prompt = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    return custom_prompt

def set_custom_prompt():
    custom_prompt = create_custom_prompt()
    prompt_set = PromptTemplate(template=custom_prompt,
                                input_variables=['question','context'])
    return prompt_set

def get_llm():

    qa_llm = CTransformers(model='D:\Raghu Studies\LLAMA2_pdf\llama-2-7b-chat.ggmlv3.q4_0.bin',
                           model_type="llama",
                           max_new_tokens = 512,
                           temperature = 0.5)
    return qa_llm

def qa_llm():
    my_llm = get_llm()
    prompt = set_custom_prompt()
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    db =  Chroma(embedding_function=embeddings, persist_directory='db')
    retriever = db.as_retriever(search_kwargs={'k': 2})
    qa_retriever = RetrievalQA.from_chain_type(return_source_documents=True, 
                                               retriever= retriever,
                                               chain_type='stuff', 
                                               llm=my_llm,
                                               chain_type_kwargs={'prompt': prompt}
                                               )
    return qa_retriever

def process_instruction(query):
    qa = qa_llm()
    results = qa({'query': query})
    print(results["result"])
    return 1

if __name__ == '__main__':

    process_instruction('What is Amino Acids disorder screening?')
