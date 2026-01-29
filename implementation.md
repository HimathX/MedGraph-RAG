# MedGraph-RAG: specialized Hybrid Retrieval & Reasoning System

## 1. Executive Summary
MedGraph-RAG is a production-grade Question Answering system designed for complex medical queries. It overcomes the limitations of traditional RAG (Retrieval Augmented Generation) by integrating **Knowledge Graph (KG)** reasoning with **Vector Search**.

This architecture enables:
- **Multi-hop Reasoning**: Traversing relationships between entities (e.g., *Drug A -> targets -> Protein B -> causes -> Side Effect C*).
- **Global Context Awareness**: Understanding broad concepts via community detection.
- **High-Fidelity Citations**: Linking answers directly to specific document sections.

---

## 2. System Architecture

The system is built on a modular four-phase pipeline:

### Phase 1: High-Fidelity ETL (Extract, Transform, Load)
We move beyond simple text splitting. Medical documents (PDFs) are complex, containing headers, tables, and diverse layouts.
- **Ingestion**: Uses **Docling** (simulated interface) to parse PDFs into structured Markdown, preserving hierarchy.
- **Chunking**: Implements a `MarkdownHeaderTextSplitter`. This ensures that chunks are semantically bounded by their headers (e.g., "Clinical Trial Results"), rather than arbitrary character counts.

### Phase 2: Knowledge Graph Construction
We build a structural index alongside the vector index.
- **Entity Extraction**: An asynchronous pipeline uses a medical-grade LLM (Gemini 2.0 Flash) to extract triplets: `(Subject, Predicate, Object)`.
- **Normalization**: A critical step to resolve ambiguity (e.g., mapping "Aß", "Amyloid-beta", and "A-beta" to a single node `Amyloid beta`). We implement a **SapBERT**-based resolution layer.
- **Graph Schema**:
  - **Nodes**: `Document`, `Section`, `Chunk`, `Entity`.
  - **Edges**: `PART_OF` (hierarchy), `MENTIONS` (chunk-to-entity), `RELATED_TO` (entity-to-entity).

### Phase 3: Hybrid Retrieval & Reasoning Agent
The core brain is a **LangGraph** agent that orchestrates the retrieval process.
- **Vector Search**: Finds semantically similar text chunks.
- **Graph Traversal**: Queries the Neo4j database to find "2-hop" neighborhoods around key entities.
- **Agent Loop**:
  1.  **Plan**: Decomposes the user query into search tasks.
  2.  **Tool**: Executes Hybrid Search (Vector + Graph).
  3.  **Reflect**: Evaluates if the retrieved context is sufficient.
  4.  **Synthesize**: Generates the final answer with strict citation requirements.

### Phase 4: User Experience (The "Front End")
A premium, "Neo-Brutalist" Streamlit interface designed for trust and observability.
- **Interactive Evidence Cloud**: A physics-based graph visualization (using `streamlit-agraph`) that shows the user exactly *why* an answer was derived.
- **Source Inspector**: A side-by-side view allowing users to click citations `[1]` and immediately see the source document snippet.

---

## 3. Implementation Details

### A. Environment Setup
The project uses `uv` for blazing fast dependency management.
```toml
[project]
dependencies = [
    "langchain",
    "neo4j",
    "streamlit",
    "streamlit-agraph",
    "networkx",
    "pydantic"
]
```

### B. The Knowledge Graph Schema
```cypher
(:Document {title: "..."}) 
  <-[:PART_OF]- (:Section {title: "..."}) 
  <-[:PART_OF]- (:Chunk {text: "...", embedding: [...]})
  -[:MENTIONS]-> (:Entity {name: "...", type: "..."})
  -[:RELATED_TO {type: "..."}]-> (:Entity)
```

### C. The Reasoning Loop (LangGraph)
The `ReasoningAgent` class encapsulates the state machine:
```python
class AgentState(TypedDict):
    query: str
    plan: List[str]
    context: List[Document]
    reflection: str
```
The graph topology is cyclic: `Plan -> Tool -> Reflect -> (Tool OR Synthesis)`.

---

## 4. Deployment & Scalability
- **Database**: Neo4j Aura (supporting GDS for community detection).
- **LLM**: Google Gemini 2.0 Flash (high throughput, large context).
- **Frontend**: Streamlit (stateless, easy horizontal scaling).

## 5. Future Roadmap
- **Real SapBERT**: Replace mock normalizer with a HuggingFace inference endpoint.
- **GDS Integration**: Move community detection from client-side (NetworkX) to server-side (Neo4j GDS) for larger graphs.
- **RAGAS Evaluation**: Implement "Hallucination Guardrails" using RAGAS metrics.
