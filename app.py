import streamlit as st
from local_embeddings import LocalEmbeddingEngine
from chroma_database import ChromaVectorDB

# Set up the look of the webpage
st.set_page_config(page_title="docsemantic Search", page_icon="🔍", layout="centered")

st.title("🔍 docsemantic AI Search")
st.write("Type a query below to search through your indexed documents instantly.")

# Initialize our background tools
@st.cache_resource
def load_tools():
    return LocalEmbeddingEngine(), ChromaVectorDB()

embedder, db = load_tools()

# Create a clean text search bar
query = st.text_input("What are you looking for?", placeholder="Enter keywords or questions...")

if query:
    with st.spinner("Searching documents..."):
        # Convert search text to vector and query the DB
        query_vec = embedder.embed_text(query)
        results = db.query(query_vector=query_vec, top_k=3)
        
        st.subheader("Top Results Found:")
        
        if results:
            for idx, res in enumerate(results):
                with st.expander(f"Result #{idx + 1} (Score: {res.get('score', 'N/A')})", expanded=True):
                    st.write(res.get('text', 'No text content available.'))
        else:
            st.info("No matching text blocks found in the database.")
