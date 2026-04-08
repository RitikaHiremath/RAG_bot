from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage
from state import RAGState

llm = ChatGroq(model="llama-3.1-8b-instant")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def retrieve(state: RAGState) -> dict:
    """Find the most relevant chunks for the user's question."""
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    query = state["messages"][-1].content
    results = vectorstore.similarity_search(query, k=3)  # top 3 chunks
    context = "\n\n".join([doc.page_content for doc in results])
    return {"context": context}

def generate(state: RAGState) -> dict:
    """Answer the question using retrieved context."""
    system = SystemMessage(content=
        f"You are a helpful assistant. Answer the user's question using "
        f"ONLY the context below. If the answer isn't in the context, "
        f"say 'I don't find that in the document.'\n\n"
        f"Context:\n{state['context']}"
    )
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}