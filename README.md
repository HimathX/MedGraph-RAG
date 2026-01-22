# Medical Graph RAG — Medical Evidence & Hypothesis Generator

A compact project that analyzes a small collection of medical abstracts (e.g., PubMed) to discover non-obvious relationships between symptoms, treatments, and biological pathways using a graph-based RAG (Retrieval-Augmented Generation) approach.

---

## 🚀 Overview

- Purpose: Build a knowledge graph from medical abstracts and use it to support multi-hop reasoning and explainable evidence retrieval for research questions.
- Scope: Small-scale prototype (20–50 abstracts) focused on a specific disease area (e.g., Alzheimer’s, Type 2 Diabetes).

## 🔬 Graph Structure

- **Nodes:** `Condition` (e.g., Diabetes), `Drug` (e.g., Metformin), `Gene/Protein`, `SideEffect`, `Study`.
- **Edges:** `TREATS`, `ASSOCIATED_WITH`, `INHIBITS`, `REPORTED_IN`.

**Example triplet:**

```
(Drug:Lisinopril)-[:TREATS]->(Condition:Hypertension)
```

## 📊 Project Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 1: PDF INGESTION                      │
│  Docling/MinerU → Layout-Aware Markdown → Staging Environment   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│            PHASE 2: ENTITY EXTRACTION & NORMALIZATION           │
│  LLM Triplets → Entity Resolution (SapBERT) → Neo4j Ingestion   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│         PHASE 3: HYBRID RETRIEVAL & MULTI-HOP REASONING         │
│  Vector Similarity + Graph Traversal → LangGraph Reasoning      │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│        PHASE 4: EVALUATION & CITATION-BACKED UI (Streamlit)     │
│  RAGAS Evaluation → Precision Citations → Dashboard with Evidence│
└─────────────────────────────────────────────────────────────────┘
```

## ✨ The "Graph RAG" Advantage

- **Multi-hop reasoning:** Answer questions like "What biological pathways are affected by drugs that treat Condition X?" by traversing the graph.
- **Fact verification & explainability:** Every generated answer can be traced back to supporting nodes/edges and source studies.

## 🛠️ 2026 Implementation Steps

1. **Curate data**
   - Download 20–50 relevant abstracts (PDF or text) from sources like PubMed.
   - Organize metadata (title, authors, year, PMID).

2. **Entity extraction**
   - Use an LLM (or an NLP pipeline) to extract triplets: (Subject - Predicate - Object).
   - Example: "Lisinopril is used to manage hypertension" → `(Drug:Lisinopril)-[:TREATS]->(Condition:Hypertension)`.

3. **Build the knowledge graph**
   - Store triplets in a graph database. For prototypes: **Neo4j Aura (free tier)** or **FalkorDB** (open-source, low-latency).

4. **Retrieve & summarize**
   - **Local search:** Query facts about a single node (e.g., a drug or gene).
   - **Global search / community summaries:** Use libraries (e.g., Microsoft GraphRAG) to summarize communities and answer broader questions such as common side effects in a treatment category.

5. **Evaluate**
   - Use evaluation frameworks such as **RAGAS** to measure performance on multi-step medical questions and check for hallucinations or unsupported claims.

## ⚖️ Notes & Ethics

- This prototype is a research tool — **not clinical advice**. Always verify findings against primary literature and domain experts.
- Ensure patient data privacy and adhere to regulatory guidelines when using clinical datasets.

## 📌 Why this is a 2026 Trend

Specialized Medical Graph RAG frameworks are gaining traction for handling private clinical data and reducing hallucinations in healthcare AI. They enable verifiable, citation-backed reasoning rather than black-box outputs.

---

## Contributing & License

Contributions welcome. See the `LICENSE` file for license details.
