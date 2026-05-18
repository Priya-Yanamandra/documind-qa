import chromadb
from typing import List, Dict, Any


class VectorStore:
    def __init__(self, persist_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_path)

    def reset_collection(self, name: str = "documents"):
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        return self.client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_or_create_collection(self, name: str = "documents"):
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        collection,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ):
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def query(
        self,
        collection,
        query_embedding: List[float],
        n_results: int = 4,
    ) -> Dict[str, Any]:
        count = collection.count()
        if count == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        n = min(n_results, count)
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
