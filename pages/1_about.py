import streamlit as st

# Page Config
st.set_page_config(
    page_title="About | MedGraph-RAG",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About MedGraph-RAG")
st.markdown("---")

# System Overview
st.header("🧬 What is MedGraph-RAG?")
st.markdown("""
**MedGraph-RAG** is an AI-powered medical research assistant that combines knowledge graphs with advanced reasoning to answer complex medical questions.

Unlike traditional retrieval systems, MedGraph-RAG can:
- **Traverse knowledge graphs** to find hidden connections between medical concepts
- **Reason across multiple sources** using Chain-of-Graph logic
- **Provide cited answers** with full source transparency
- **Evaluate its own responses** using RAGAS metrics
""")

st.markdown("---")

# Architecture
st.header("🏗️ System Architecture")
st.image("data/Architecture.png", caption="MedGraph-RAG System Architecture", use_container_width=True)

st.markdown("---")

# Technology Stack
st.header("🛠️ Technology Stack")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Core Components")
    st.markdown("""
    - **LangGraph**: Multi-agent reasoning framework
    - **Neo4j**: Graph database for medical knowledge
    - **Google Gemini**: Large language model for generation
    - **Streamlit**: Interactive web interface
    """)
    
with col2:
    st.subheader("Additional Tools")
    st.markdown("""
    - **RAGAS**: Answer faithfulness evaluation
    - **LangChain**: LLM orchestration
    - **Pydantic**: Data validation
    - **NetworkX**: Graph analysis
    """)

st.markdown("---")

# How It Works
st.header("⚙️ How It Works")

with st.expander("1. 📥 Question Input", expanded=True):
    st.markdown("""
    Users ask natural language questions about medical topics (e.g., Alzheimer's disease, treatments, biomarkers).
    """)

with st.expander("2. 🧠 Planning & Reasoning"):
    st.markdown("""
    The system creates a multi-step reasoning plan using LangGraph:
    - Identifies key entities and concepts
    - Determines what information is needed
    - Plans graph traversal strategy
    """)

with st.expander("3. 🔍 Hybrid Retrieval"):
    st.markdown("""
    Combines two retrieval methods:
    - **Vector Search**: Semantic similarity for finding relevant documents
    - **Graph Traversal**: Following relationships between medical entities
    """)

with st.expander("4. 🕸️ Knowledge Graph Reasoning"):
    st.markdown("""
    Traverses the Neo4j knowledge graph to:
    - Find direct relationships (e.g., "treats", "causes")
    - Discover multi-hop connections
    - Aggregate evidence from multiple paths
    """)

with st.expander("5. 💡 Answer Generation"):
    st.markdown("""
    Google Gemini synthesizes a comprehensive answer:
    - Grounded in retrieved evidence
    - Includes inline citations [1], [2], etc.
    - Provides follow-up suggestions
    """)

with st.expander("6. 🎯 Evaluation (Optional)"):
    st.markdown("""
    RAGAS metrics evaluate answer quality:
    - **Faithfulness**: Is the answer factually consistent with sources?
    - **Relevancy**: Does it address the question?
    """)

st.markdown("---")

# Use Cases
st.header("💡 Use Cases")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🔬 Research**
    - Literature review
    - Hypothesis generation
    - Finding connections
    """)

with col2:
    st.markdown("""
    **🏥 Clinical**
    - Treatment options
    - Drug interactions
    - Diagnostic support
    """)

with col3:
    st.markdown("""
    **📚 Education**
    - Medical training
    - Concept exploration
    - Knowledge synthesis
    """)

st.markdown("---")

# Footer
st.caption("Built with ❤️ for advancing medical AI research")
