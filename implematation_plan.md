# Implementation Plan — PDF-Centric Knowledge Graph

**Timeline:** 4 weeks (prototype)

## Phase 1 — High-Fidelity PDF Ingestion (Week 1)

- **Advanced Parsing:** Use Docling or MinerU to convert the 15 full PDFs into layout-aware Markdown. This is critical for 2026 workflows to preserve complex tables and section hierarchies (e.g., distinguishing "Methods" from "Results").
- **Structural Mapping:** Instead of generic chunks, map the document hierarchy into the graph: `(Document) -> (Section) -> (Chunk)`.
- **Staging:** Store Markdown outputs and metadata (Title, Author, Year, DOI) in a structured staging environment.

## Phase 2 — Relation Extraction & Normalization (Week 2)

- **Entity Resolution:** Use SapBERT or a biomedical LLM to map extracted terms (e.g., "Hypertension") to a unified ID (e.g., MeSH).
- **LLM Triplets:** Design extraction prompts that output JSON triplets `(Subject, Predicate, Object)` based on the content of the full text.
- **Batch Ingestion:** Use Cypher MERGE in Neo4j to deduplicate nodes while creating edges that represent findings in the papers.

## Phase 3 — Hybrid Retrieval & Reasoning (Week 3)

- **Hybrid Retriever:** Implement a search engine that combines Vector similarity (finding relevant text) with Graph traversal (following relationships between entities).
- **Chain-of-Graph Reasoning:** Use LangGraph to create a multi-hop reasoning loop that "walks" the graph to answer complex "Why" or "How" questions found in the full PDFs.
- **Global Context:** Integrate Microsoft GraphRAG to generate community summaries for large clusters of related experimental data.

## Phase 4 — Evaluation & Citation UI (Week 4)

- **RAGAS Evaluation:** Measure faithfulness and answer relevance to ensure the system doesn't hallucinate outside the provided 15 PDFs.
- **Precision Citations:** Map every graph edge back to a specific PDF page or section.
- **Streamlit UI:** Prototype a dashboard showing the user's question, the visual reasoning path through the graph, and the final answer with hyperlinked citations.

## Deliverables

PDF-to-Graph ETL pipeline, normalized Neo4j graph, multi-hop retrieval engine, and a citation-backed UI.
