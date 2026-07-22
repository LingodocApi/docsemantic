import os
from local_embeddings import LocalEmbeddingEngine
from chroma_database import ChromaVectorDB

def load_and_index_documents():
    print("--- docsemantic Document Indexer ---")
    embedder = LocalEmbeddingEngine()
    db = ChromaVectorDB()
    
    # We will look for a folder named 'documents'
    folder_path = "./documents"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Created a './documents' folder. Put your text files inside it!")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            print(f"Reading {filename}...")
            
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Split the document into paragraph chunks
            chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
            
            for i, chunk in enumerate(chunks):
                vector = embedder.embed_text(chunk)
                db.add_document(
                    doc_id=f"{filename}_{i}",
                    vector=vector,
                    text=chunk
                )
    print("Finished indexing all text documents!")

if __name__ == "__main__":
    load_and_index_documents()
