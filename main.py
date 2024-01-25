from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")

pinecone.init(
    api_key=PINECONE_API_KEY, environment=PINECONE_ENV
)


def load():
    loader = PyPDFLoader(
        "/Users/qichen/Documents/Volunteer/LEXIFY/AI Projects/LexAutoNew/7-UP BOTTLING CO. LTD & ORS V. ABIOLA & SONS BOTTLING CO. LTD.pdf")
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="langchain-doc-index")


if __name__ == "__main__":
    load()


def run_llm(query):
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings, index_name="langchain-doc-index")
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="refine", retriever=docsearch.as_retriever())

    result = qa({"query": query})
    print(result)
    return result
