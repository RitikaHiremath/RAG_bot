from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_vectorstore(file_path: str) -> Chroma:
    # 1. Load document
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    docs = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # 3. Embed and store
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # small, fast, free
    )
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"✅ Indexed {len(chunks)} chunks from {file_path}")
    return vectorstore