from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load raw PDF(s)
DATA_PATH="data/"

def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyMuPDFLoader)
    
    documents=loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)
print("Length of PDF pages: ", len(documents))

# Step 2: Create Chunks
def create_chunks(pdf_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(pdf_data)
    return text_chunks

text_chunks=create_chunks(pdf_data=documents)

# Step 3: Create Vector Embeddings 

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)
