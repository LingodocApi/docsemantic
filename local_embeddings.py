import numpy as np
from sentence_transformers import SentenceTransformer

class LocalEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = 384

    def encode(self, texts, show_progress=True):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=show_progress, 
            convert_to_numpy=True
        )
        return embeddings

    def encode_query(self, query):
        return self.encode(query, show_progress=False)[0]

if __name__ == "__main__":
    embedder = LocalEmbeddings()
    test_texts = ["This is a test document", "Another example text"]
    embeddings = embedder.encode(test_texts)
    print(f"Generated {len(embeddings)} embeddings")
    