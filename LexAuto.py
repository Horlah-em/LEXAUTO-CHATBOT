# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:20:42 2023

@author: Emmanuel
"""

from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os



#create qdrant_client
os.environ['QDRANT_HOST'] = 'https://cf2f02e8-1e5e-40d8-bc0c-10551a95f7aa.us-east-1-0.aws.cloud.qdrant.io:6333/collections'
os.environ['QDRANT_API_KEY'] = 'z-Ju8cCamyR8TxFtemuNYvpk_cVcwaP_DWtR4ljGoAC_1JLt9UvdVw'

client = qdrant_client.QdrantClient(
    os.getenv('QDRANT_HOST'),
    api_key=os.getenv('QDRANT_API_KEY')
    )


#create collection

os.environ['QDRANT_COLLECTION_NAME']= 'my_collection'

vectors_config =qdrant_client.http.models.VectorParams(
    size=1536,
    distance= qdrant_client.http.models.Distance.COSINE
    
    )

client.recreate_collection(
    collection_name=os.getenv('QDRANT_COLLECTION_NAME'),
    vectors_config=vectors_config,
    )

os.environ['OPENAI_API_KEY']= ''
embeddings = OpenAIEmbeddings()

vector_store =Qdrant(
    client = client,
    collection_name=os.getenv('QDRANT_COLLECTION_NAME'),
    embeddings = embeddings,
    )

from langchain.text_splitter import CharacterTextSplitter

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator ="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

with open('law_cases.txt') as f:
    raw_text = f.read()
    
texts = get_chunks(raw_text)
    
vector_store.add_texts(texts)


#plug vector store into retrieval chain

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

qa= RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vector_store.as_retriever()
    )



query=input("Enter your Question")

response = qa.run(query)
print(response)

