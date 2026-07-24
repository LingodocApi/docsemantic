import math
import sys
from chroma_database import ChromaVectorDB
from local_embeddings import LocalEmbeddings

class SemanticSearchEngine:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print("Loading embedding model...")
        self.embedder = LocalEmbeddings(model_name)
        self.db = ChromaVectorDB()

    def search(self, query_text, top_k=3):
        if not query_text.strip():
            return []

        # Get embedding as flat list
        vec = self.embedder.encode(query_text)
        if hasattr(vec, 'tolist'):
            vec_list = vec.tolist()
        else:
            vec_list = list(vec)
        
        # Flatten if nested
        if isinstance(vec_list, list) and vec_list and isinstance(vec_list[0], list):
            vec_list = vec_list[0]

        # Query database
        results = self.db.collection.query(
            query_embeddings=[vec_list],
            n_results=top_k
        )
        
        formatted = []
        if results and results['documents']:
            docs = results['documents'][0]
            metas = results['metadatas'][0] if results['metadatas'] else [{}] * len(docs)
            dists = results['distances'][0] if results['distances'] else [0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                # Use exponential decay to convert distance to similarity
                similarity = math.exp(-dist) * 100
                score = max(0.0, min(100.0, similarity))
                formatted.append({
                    "content": doc,
                    "metadata": meta,
                    "score": round(score, 2)
                })
        return formatted

if __name__ == "__main__":
    engine = SemanticSearchEngine()
    results = engine.search("car maintenance", 3)
    print(f"Found {len(results)} results")
    for r in results:
        print(f"Score: {r['score']}%")
        print(f"Content: {r['content'][:100]}...")
        print()
