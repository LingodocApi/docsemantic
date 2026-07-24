import os
import chromadb
from typing import List, Optional

class ChromaVectorDB:
    def __init__(self, storage_path="chroma_storage", collection_name="docsemantic_collection"):
        """Initializes the persistent ChromaDB client."""
        self.storage_path = storage_path
        self.collection_name = collection_name
        
        # Ensure storage directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.storage_path)
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None
        )
        print(f"Connected to ChromaDB at '{self.storage_path}' | Collection: '{self.collection_name}'")

    def add_documents(self, ids, embeddings, documents, metadatas=None):
        """Adds text chunks and their corresponding vector embeddings to the database."""
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Successfully added {len(ids)} chunks to the database.")
        except Exception as e:
            print(f"Error adding documents to database: {e}")

    def query_db(self, query_vector, n_results=3):
        """Queries the database using a vector and returns the top matches."""
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Error querying database: {e}")
            return None

    def query(self, query_vector, top_k=3):
        """Alternative query method that returns just the text."""
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )
            return results['documents'][0] if results and results['documents'] else []
        except Exception as e:
            print(f"Error querying database: {e}")
            return []

    def count(self):
        """Returns the number of documents in the collection."""
        return self.collection.count()

if __name__ == "__main__":
    db = ChromaVectorDB()
    print(f"Current document count: {db.count()}")
    