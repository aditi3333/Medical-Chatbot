import pickle
import os
from urllib import response
import torch
from copy import deepcopy
import colorama
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaTokenizer, set_seed, pipeline
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from huggingface_hub import snapshot_download
from langchain_openai import ChatOpenAI
DB_FAISS_PATH="vectorstore/db_faiss"

from langchain.prompts import PromptTemplate

def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db=FAISS.load_local(DB_FAISS_PATH,embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(CUSTOM_PROMPT_TEMPLATE):
    prompt=PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE,input_variables=["context","question"])
    return prompt

def build_qa_chain(llm, vectorstore, question):
   
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(question)}
    )


def main():
    # Raw model for testing purposes
    print("Loading model...")
    load_dotenv()
    llm = ChatOpenAI(
        model="llama3.1:8b",
        base_url="https://genai.science-cloud.hu/api/",
        api_key=os.environ.get("GENAI_API_KEY"),
        temperature=0
    )
    
    
    vectorstore = get_vectorstore()

    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() in ["exit", "quit"]:
            break
        
        system_prompt = (
            "  "
        )

        custom_prompt = (
            system_prompt + "\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"

        )

        
        qa_chain = build_qa_chain(llm, vectorstore, custom_prompt)
        response = qa_chain.invoke({'query': question})


        print("Answer:")
        print(response["result"])


if __name__ == "__main__":
    main()
