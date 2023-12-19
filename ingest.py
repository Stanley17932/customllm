from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import VectorStore, FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = 'traIn_data/'
DB_FAISS_PATH = 'vectorstores/'

#create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf',loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device':'cpu'})
    
    # Print the size of the embeddings list
    print("Number of documents processed:", len(texts))
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == '__main__':
    create_vector_db()
