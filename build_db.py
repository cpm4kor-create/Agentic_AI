import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "agentic_kb")
KNOWLEDGE_FILE = "knowledge.txt"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def read_knowledge_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{file_path} not found")

    return path.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def build_database():
    print("Reading knowledge file...")
    text = read_knowledge_file(KNOWLEDGE_FILE)

    print("Splitting text into chunks...")
    chunks = chunk_text(text)

    print(f"Total chunks created: {len(chunks)}")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Creating embeddings...")
    embeddings = embedding_model.encode(chunks).tolist()

    print("Creating ChromaDB collection...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Old collection deleted.")
    except Exception:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    metadatas = [
        {
            "source": KNOWLEDGE_FILE,
            "chunk_id": i
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("Database created successfully.")
    print(f"ChromaDB folder: {CHROMA_PATH}")
    print(f"Collection name: {COLLECTION_NAME}")


if __name__ == "__main__":
    build_database()