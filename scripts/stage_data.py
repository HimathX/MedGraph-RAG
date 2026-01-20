import json
import sys
from pathlib import Path
from typing import List

# Add parent directory to path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion import MarkdownLoader

def main():
    input_dir = Path("data/markdowns")
    output_dir = Path("data/staging")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    docs_output = output_dir / "structured_documents.json"
    metadata_output = output_dir / "metadata.json"
    
    markdown_files = list(input_dir.glob("*.md"))
    
    if not markdown_files:
        print("No markdown files found in data/markdowns/")
        return
        
    documents = []
    metadata_summary = []
    
    print(f"Found {len(markdown_files)} markdown files. Processing...")
    
    for file_path in markdown_files:
        try:
            print(f"Processing {file_path.name}...")
            loader = MarkdownLoader(file_path)
            doc = loader.parse()
            
            # Convert to dict for JSON serialization
            doc_dict = doc.model_dump()
            documents.append(doc_dict)
            
            summary = {
                "id": doc.id,
                "title": doc.title,
                "authors": doc.authors,
                "sections_count": len(doc.sections),
                "source": str(file_path)
            }
            metadata_summary.append(summary)
            print(f"  ✓ Processed: {doc.title[:40]}... ({len(doc.sections)} sections)")
            
        except Exception as e:
            print(f"  ✗ Failed to process {file_path.name}: {e}")
            
    # Save results
    print(f"\nSaving {len(documents)} documents to {docs_output}...")
    with open(docs_output, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
        
    print(f"Saving metadata summary to {metadata_output}...")
    with open(metadata_output, "w", encoding="utf-8") as f:
        json.dump(metadata_summary, f, indent=2, ensure_ascii=False)
        
    print("Done!")

if __name__ == "__main__":
    main()
