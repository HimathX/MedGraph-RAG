import os
import glob
import asyncio
import argparse
from pathlib import Path
from src.graph import Neo4jManager
from src.ingestion import MarkdownLoader
from src.extraction import RelationExtractor
from src.reasoning import ReasoningAgent
from src.llm import get_embeddings

async def ingest_data(data_dir: Path):
    """
    Iterates over all .md files in the data directory, parses them,
    and ingests them into the Neo4j graph with embeddings.
    """
    manager = Neo4jManager()
    extractor = RelationExtractor()
    embeddings = get_embeddings()
    
    try:
        # Find all markdown files
        md_files = list(data_dir.glob("**/*.md"))
        
        if not md_files:
            print(f"No markdown files found in {data_dir}")
            return

        print(f"Found {len(md_files)} markdown files. Starting ingestion...")

        for file_path in md_files:
            try:
                print(f"Processing: {file_path.name}")
                
                # 1. Parse Markdown
                loader = MarkdownLoader(file_path)
                document = loader.parse()
                
                # 2. Ingest into Neo4j
                doc_dict = document.model_dump()
                manager.add_document(doc_dict)
                print(f"✅ Ingested Document: {document.title}")

                # 3. Extract Triplets (Phase 2)
                print(f"   Extracting triplets from {len(document.sections)} sections...")
                triplets = await extractor.process_document(doc_dict)
                print(f"   Found {len(triplets)} triplets.")
                
                # 4. Ingest Triplets
                if triplets:
                    manager.add_triplets(triplets)
                    print(f"   ✅ Ingested {len(triplets)} triplets into graph.")
                
                # 5. Create Chunks with Embeddings (Phase 3)
                print(f"   Creating chunks with embeddings...")
                for section in document.sections:
                    if section.content and len(section.content) > 50:
                        # Create embedding
                        embedding = embeddings.embed_query(section.content)
                        
                        # Store chunk in Neo4j
                        chunk_query = """
                        CREATE (c:Chunk {
                            id: $chunk_id,
                            content: $content,
                            section_id: $section_id,
                            embedding: $embedding
                        })
                        """
                        with manager.driver.session() as session:
                            session.run(chunk_query, 
                                      chunk_id=section.id,
                                      content=section.content,
                                      section_id=section.id,
                                      embedding=embedding)
                print(f"   ✅ Created chunks with embeddings.")
                
            except Exception as e:
                print(f"❌ Failed to process {file_path.name}: {e}")
                await asyncio.sleep(0.1)

        # 6. Create Indexes
        print("\n>>> Creating Vector Index...")
        manager.create_vector_index(dimension=768)
        
        print("\n>>> Running Community Detection...")
        manager.create_community_index()

    finally:
        manager.close()

async def query_mode():
    """
    Interactive query mode using the Phase 3 Reasoning Agent.
    """
    print("\n" + "="*60)
    print("MedGraph-RAG Query Interface (Phase 3)")
    print("="*60)
    print("Using: Gemini 3 Pro + Hybrid Retrieval + Chain-of-Graph Reasoning")
    print("Type 'exit' or 'quit' to stop.\n")
    
    agent = ReasoningAgent()
    
    while True:
        try:
            query = input("\n🔍 Your Question: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
                
            if not query:
                continue
            
            print("\n⚙️  Running Chain-of-Graph Reasoning Agent...\n")
            result = await agent.run(query)
            
            print("\n" + "="*60)
            print("📊 REASONING TRACE:")
            print("="*60)
            print(f"Plan: {result.get('plan', [])}")
            print(f"Steps Executed: {result.get('current_step', 0)}")
            print(f"Reflection: {result.get('reflection', 'N/A')}")
            
            print("\n" + "="*60)
            print("💡 FINAL ANSWER:")
            print("="*60)
            print(result.get('answer', 'No answer generated.'))
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

async def main():
    parser = argparse.ArgumentParser(description="MedGraph-RAG: Phase 3 - Hybrid Retrieval & Reasoning")
    parser.add_argument("--mode", choices=["ingest", "query"], default="query",
                       help="Mode: 'ingest' to load data, 'query' to ask questions")
    parser.add_argument("--data-dir", type=str, default="data/markdowns",
                       help="Directory containing markdown files (for ingest mode)")
    
    args = parser.parse_args()
    
    if args.mode == "ingest":
        base_dir = Path(__file__).parent
        data_dir = base_dir / args.data_dir
        
        if not data_dir.exists():
            print(f"Data directory not found: {data_dir}")
            print("Checking 'data' root...")
            data_dir = base_dir / "data"
        
        await ingest_data(data_dir)
    else:
        await query_mode()

if __name__ == "__main__":
    asyncio.run(main())
