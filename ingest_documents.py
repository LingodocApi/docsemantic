import os
from pathlib import Path
from tqdm import tqdm
from local_embeddings import LocalEmbeddings
from chroma_database import ChromaVectorDB

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            for sep in ['. ', '\n\n', '\n', ' ']:
                last_sep = text.rfind(sep, start, end)
                if last_sep != -1 and last_sep > start + chunk_size // 2:
                    end = last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < len(text) else end
    
    return chunks

def ingest_documents(doc_folder="documents", chunk_size=500, overlap=50):
    """Ingest all documents from a folder into ChromaDB."""
    print("Starting document ingestion...")
    embedder = LocalEmbeddings()
    db = ChromaVectorDB()
    
    doc_path = Path(doc_folder)
    if not doc_path.exists():
        print(f"Documents folder '{doc_folder}' not found!")
        print(f"Creating '{doc_folder}' folder...")
        doc_path.mkdir(parents=True)
        print(f"Please add your .txt or .md files to the '{doc_folder}' folder.")
        return
    
    text_files = list(doc_path.glob("*.txt")) + list(doc_path.glob("*.md"))
    
    if not text_files:
        print(f"No .txt or .md files found in '{doc_folder}'")
        return
    
    print(f"Found {len(text_files)} document(s) to process")
    
    all_chunks = []
    all_metadata = []
    sources = []
    
    for file_path in tqdm(text_files, desc="Reading documents"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print(f"Skipping empty file: {file_path.name}")
                continue
            
            chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": file_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
                sources.append(file_path.name)
            
            print(f"Processed '{file_path.name}': {len(chunks)} chunks")
            
        except Exception as e:
            print(f"Error processing '{file_path.name}': {e}")
    
    if not all_chunks:
        print("No valid content found in any documents")
        return
    
    print(f"\nCreated {len(all_chunks)} total text chunks")
    print("Generating embeddings (this may take a moment)...")
    embeddings = embedder.encode(all_chunks, show_progress=True)
    
    ids = [f"chunk_{i:04d}" for i in range(len(all_chunks))]
    
    print("Storing in ChromaDB...")
    db.add_documents(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=all_chunks,
        metadatas=all_metadata
    )
    
    final_count = db.collection.count()
    print(f"\nIngest complete!")
    print(f"Total chunks in database: {final_count}")
    print(f"Documents processed: {len(set(sources))}")
    print(f"Source files: {', '.join(set(sources))}")

def clear_database():
    """Delete all documents from the database."""
    db = ChromaVectorDB()
    try:
        db.collection.delete_all()
        print("Database cleared successfully")
    except Exception as e:
        print(f"Error clearing database: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into DocSemantic")
    parser.add_argument("--clear", action="store_true", help="Clear database before ingestion")
    parser.add_argument("--folder", default="documents", help="Documents folder path")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap in characters")
    
    args = parser.parse_args()
    
    if args.clear:
        clear_database()
    
    ingest_documents(
        doc_folder=args.folder,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    