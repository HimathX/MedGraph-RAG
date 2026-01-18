# Implementation Plan — Short & Professional

**Timeline:** 4 weeks (prototype)

## Phase 1 — Env & Data (Week 1)

- Setup Python 3.11+ environment and install core deps: `langchain`, `neo4j`, `openai`/`Ollama`, `pydantic`.
- Scrape ~50 PubMed abstracts (Biopython Entrez) and capture metadata (PMID, DOI, date).
- Chunk and stage documents for processing.

## Phase 2 — Graph Construction (Week 2)

- Define graph schema (Cypher) for Neo4j/FalkorDB.
- Design LLM relation-extraction prompt to output JSON triplets (Subject, Predicate, Object).
- Batch-import triplets using Cypher `MERGE` to deduplicate nodes.

## Phase 3 — Retrieval & Reasoning (Week 3)

- Build a hybrid retriever (graph + vector) for context-rich search.
- Implement multi-hop query engine (chain-of-graph reasoning).
- Integrate Microsoft GraphRAG for community/global summaries.

## Phase 4 — Eval & UI (Week 4)

- Evaluate with RAGAS (faithfulness, relevance).
- Add citation mapping (edge → PMID).
- Prototype Streamlit UI showing question, reasoning path, answer, and citations.

**Deliverables:** Working ETL pipeline, populated graph, retrieval engine, evaluation report, and a minimal UI.

**Notes:** Keep privacy and ethical considerations front-and-center when using clinical literature or patient data.
