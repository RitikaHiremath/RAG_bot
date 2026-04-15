# RAG Bot — LangGraph-based Retrieval-Augmented Generation Pipeline

A conversational AI system built with LangGraph that enables natural language 
question answering over custom documents using retrieval-augmented generation.

## Architecture

The system is structured as a stateful LangGraph graph with discrete, 
single-responsibility nodes:

| File | Responsibility |
|---|---|
| `ingest.py` | Loads documents, generates embeddings, indexes into Chroma |
| `state.py` | Defines shared graph state passed between nodes |
| `nodes.py` | Retrieval node (semantic search) and generation node (LLM response) |
| `graph.py` | Assembles the LangGraph graph with node connections and routing |
| `main.py` | Entry point, runs the conversational loop |

## Key Design Decisions

- **LangGraph over a simple chain** — enables stateful conversation memory 
  and conditional routing between retrieval and direct generation
- **Chroma as local vector store** — fast embedding-based retrieval with no 
  external dependencies
- **Single-responsibility nodes** — each node can be tested and replaced 
  independently without touching the rest of the graph

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your GROQ API key.

Place your documents in the `docs/` folder, then:

```bash
python ingest.py    # index documents into Chroma
python main.py      # start the chat
```

## How It Works

1. `ingest.py` loads documents from `docs/`, splits them into chunks, 
   generates embeddings, and stores them in a local Chroma vector database
2. At query time, the retrieval node performs semantic search over the 
   vector store to find relevant chunks
3. Retrieved chunks are injected as context into the LLM prompt
4. The generation node produces a grounded response using the retrieved context
5. Conversation history is maintained in the graph state across turns

## Requirements

- Python 3.10+
- GROQ API key 