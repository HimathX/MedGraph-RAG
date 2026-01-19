import argparse
import json
from pathlib import Path
from datetime import datetime
from docling.document_converter import DocumentConverter


def convert_pdf_to_markdown(source: str, output_dir: Path, converter: DocumentConverter) -> dict:
    """
    Convert a single PDF to markdown and save it.
    
    Args:
        source: URL or local path to PDF file
        output_dir: Directory to save markdown files
        converter: DocumentConverter instance
        
    Returns:
        dict: Metadata about the conversion
    """
    try:
        print(f"Converting: {source}")
        result = converter.convert(source)
        doc = result.document
        markdown_content = doc.export_to_markdown()
        
        # Generate output filename
        if source.startswith("http"):
            # Extract filename from URL
            filename = source.split("/")[-1].replace(".pdf", ".md")
        else:
            # Use local file name
            filename = Path(source).stem + ".md"
        
        output_path = output_dir / filename
        
        # Save markdown file
        output_path.write_text(markdown_content, encoding="utf-8")
        print(f"✓ Saved: {output_path}")
        
        return {
            "source": source,
            "output_file": str(output_path),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        print(f"✗ Failed to convert {source}: {str(e)}")
        return {
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Convert PDF files to Markdown using Docling")
    parser.add_argument(
        "sources",
        nargs="*",
        help="PDF URLs or file paths to convert. If not provided, will process files from data/abstracts/"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("data/markdowns"),
        help="Output directory for markdown files (default: data/markdowns)"
    )
    parser.add_argument(
        "-l", "--log-file",
        type=Path,
        default=Path("data/markdowns/converted.json"),
        help="JSON file to log conversion results (default: data/markdowns/converted.json)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine sources to convert
    sources = args.sources
    if not sources:
        # Look for PDF files in data/abstracts/
        abstracts_dir = Path("data/abstracts")
        if abstracts_dir.exists():
            sources = [str(f) for f in abstracts_dir.glob("*.pdf")]
            if not sources:
                print("No PDF files found in data/abstracts/")
                return
        else:
            print("No sources provided and data/abstracts/ directory not found")
            print("Usage: python converter.py <PDF_URL_or_PATH> [<PDF_URL_or_PATH> ...]")
            return
    
    # Initialize converter
    converter = DocumentConverter()
    
    # Convert all sources
    results = []
    for source in sources:
        result = convert_pdf_to_markdown(source, args.output_dir, converter)
        results.append(result)
    
    # Save conversion log
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing log if it exists
    existing_results = []
    if args.log_file.exists():
        try:
            with open(args.log_file, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []
    
    # Append new results
    existing_results.extend(results)
    
    # Save updated log
    with open(args.log_file, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    print(f"\n{'='*50}")
    print(f"Conversion complete: {successful} successful, {failed} failed")
    print(f"Results logged to: {args.log_file}")


if __name__ == "__main__":
    main()