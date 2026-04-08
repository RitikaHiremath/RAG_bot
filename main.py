from dotenv import load_dotenv
load_dotenv()

import os
from graph import graph
from ingest import load_vectorstore

# Index your document first
doc_path = "docs/sample.txt"   # ← change to your file
if not os.path.exists("./chroma_db"):
    load_vectorstore(doc_path)
else:
    print("✅ Using existing vector store")

config = {"configurable": {"thread_id": "rag-session-1"}}
print("\nRAG Chatbot ready! Ask questions about your document.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        break

    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config
    )
    print(f"Bot: {result['messages'][-1].content}\n")