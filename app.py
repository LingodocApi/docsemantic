import streamlit as st
from search import SemanticSearchEngine

st.set_page_config(page_title="DocSemantic", layout="wide")
st.title("🔍 DocSemantic")

# Initialize
@st.cache_resource
def load_engine():
    return SemanticSearchEngine()

engine = load_engine()

# Show database count
count = engine.db.collection.count()
st.sidebar.write(f"📊 {count} chunks in database")

# Search
query = st.text_input("Ask a question:")
results_count = st.slider("Results", 1, 10, 3)

if query:
    with st.spinner("Searching..."):
        results = engine.search(query, results_count)
        
        if results:
            st.success(f"Found {len(results)} results")
            for i, r in enumerate(results):
                with st.expander(f"Result {i+1} (Score: {r['score']}%)"):
                    st.write(f"**Source:** {r['metadata'].get('source', 'Unknown')}")
                    st.write(r['content'])
        else:
            st.warning("No results found. Try a different query.")

# Show document samples
if st.sidebar.button("Show all documents"):
    docs = engine.db.collection.get()
    if docs['documents']:
        st.sidebar.write("---")
        st.sidebar.write("All chunks:")
        for i, doc in enumerate(docs['documents']):
            st.sidebar.write(f"{i+1}. {doc[:80]}...")
            