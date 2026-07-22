search.py from local_embeddings import LocalEmbeddingEngine
from chroma_database import ChromaVectorDB

def interactive_search():
    print("--- docsemantic Interactive Search Engine ---")
    embedder = LocalEmbeddingEngine()
    db = ChromaVectorDB()
    
    while True:
        query_text = input("Search query > ").strip()
        if query_text.lower() in ['exit', 'quit']:
            break
        query_vec = embedder.embed_text(query_text)
        results = db.query(query_vector=query_vec, top_k=2)
        print(results)

if __name__ == "__main__":
    interactive_search() 
