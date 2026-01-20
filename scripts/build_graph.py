import sys
import json
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.extraction import RelationExtractor
from src.graph import Neo4jManager

load_dotenv()

async def main():
    parser = argparse.ArgumentParser(description="Build Knowledge Graph from Staged Data")
    parser.add_argument("--dry-run", action="store_true", help="Run extraction but do not write to Neo4j")
    parser.add_argument("--files", nargs="+", help="Specific filenames to process (e.g. 2.md 3.md)")
    args = parser.parse_args()

    # Load Staged Data
    staging_file = Path("data/staging/structured_documents.json")
    if not staging_file.exists():
        print("Staged data not found. Please run scripts/stage_data.py first.")
        return

    with open(staging_file, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # Filter documents if requested
    if args.files:
        target_files = [f if f.lower().endswith(".md") else f"{f}.md" for f in args.files]
        print(f"Filtering for: {target_files}")
        documents = [
            d for d in documents 
            if Path(d["source_path"]).name in target_files
        ]
        
    if not documents:
        print("No documents found to process.")
        return

    print(f"Processing {len(documents)} documents...")

    # Initialize Components
    extractor = RelationExtractor()
    neo4j = None
    if not args.dry_run:
        try:
            neo4j = Neo4jManager()
            print("Connected to Neo4j.")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            print("Proceeding in DRY RUN mode.")
            args.dry_run = True

    # Process Loop
    for doc in documents:
        print(f"\nProcessing Document: {doc['title']} ({Path(doc['source_path']).name})")
        
        # 1. Add Document Node
        if not args.dry_run:
            neo4j.add_document(doc)
            print("  ✓ Document node created.")

        # 2. Extract Triplets
        print("  Extracting relations (this may take a moment)...")
        triplets = await extractor.process_document(doc)
        print(f"  Found {len(triplets)} triplets.")
        
        if triplets:
            if args.dry_run:
                print("  [Dry Run] Sample Triplets:")
                for t in triplets[:3]:
                    print(f"   - {t['head']} -> [{t['relation']}] -> {t['tail']}")
            else:
                # 3. Load Triplets
                try:
                    neo4j.add_triplets(triplets)
                    print(f"  ✓ Loaded {len(triplets)} relations into Neo4j.")
                except Exception as e:
                    print(f"  ✗ Failed to load triplets: {e}")

    if neo4j:
        neo4j.close()
    
    print("\nJob Complete.")

if __name__ == "__main__":
    asyncio.run(main())
