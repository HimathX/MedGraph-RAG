import os
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import requests
from Bio import Entrez

# ==================== CONFIGURATION ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NCBI Configuration
Entrez.email = "himathavenger@gmail.com"
Entrez.tool = "Alzheimer-Research-Agent/1.0"
Entrez.api_key = "ee18e3f99a6ba7500c20bb2d175d8de5e408"  # Your API key

SEARCH_QUERY = "Alzheimer's disease therapeutic pathways"
MAX_RESULTS = 100  # Start small for testing
SAVE_DIR = Path("data/abstracts")
METADATA_FILE = SAVE_DIR / "metadata.json"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiting
REQUEST_DELAY = 0.34 if Entrez.api_key else 0.5

# ==================== METADATA TRACKING ====================
def load_metadata() -> Dict:
    """Load existing download metadata."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {"downloaded": [], "failed": [], "no_pmc": [], "total_requests": 0}

def save_metadata(metadata: Dict):
    """Persist download tracking."""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

# ==================== PUBMED SEARCH ====================
def search_pubmed(query: str, limit: int) -> List[str]:
    """Search PubMed with PMC availability filter."""
    try:
        logger.info(f"Searching PubMed: '{query}' (limit: {limit})")
        
        # Filter for FREE full text in PMC
        pmc_filtered_query = f'({query}) AND ("pubmed pmc"[filter] OR "free full text"[filter])'
        
        handle = Entrez.esearch(
            db="pubmed",
            term=pmc_filtered_query,
            retmax=limit,
            usehistory="y"
        )
        results = Entrez.read(handle)
        handle.close()
        
        pmid_list = results["IdList"]
        logger.info(f"Found {len(pmid_list)} PMIDs with PMC availability")
        return pmid_list
        
    except Exception as e:
        logger.error(f"PubMed search failed: {e}")
        return []

# ==================== FETCH ARTICLE METADATA ====================
def fetch_article_metadata(pmid: str) -> Optional[Dict]:
    """Retrieve structured metadata (title, abstract, authors)."""
    try:
        handle = Entrez.efetch(
            db="pubmed",
            id=pmid,
            rettype="medline",
            retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()
        
        if not records['PubmedArticle']:
            return None
            
        article = records['PubmedArticle'][0]['MedlineCitation']['Article']
        
        # Extract abstract
        abstract = ""
        if 'Abstract' in article:
            abstract_texts = article['Abstract'].get('AbstractText', [])
            if isinstance(abstract_texts, list):
                abstract = ' '.join(str(text) for text in abstract_texts)
            else:
                abstract = str(abstract_texts)
        
        # Extract authors
        authors = []
        if 'AuthorList' in article:
            for author in article['AuthorList']:
                if 'LastName' in author and 'Initials' in author:
                    authors.append(f"{author['LastName']} {author['Initials']}")
        
        return {
            "pmid": pmid,
            "title": str(article.get('ArticleTitle', 'N/A')),
            "abstract": abstract,
            "authors": authors,
            "journal": str(article.get('Journal', {}).get('Title', 'N/A'))
        }
    except Exception as e:
        logger.warning(f"Could not fetch metadata for PMID {pmid}: {e}")
        return None

# ==================== LINK TO PMC ====================
def get_pmc_id(pmid: str) -> Optional[str]:
    """Convert PMID to PMCID using elink."""
    try:
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
        results = Entrez.read(handle)
        handle.close()
        
        if results and results[0].get("LinkSetDb"):
            for linkset in results[0]["LinkSetDb"]:
                if linkset["LinkName"] == "pubmed_pmc":
                    pmc_id = linkset["Link"][0]["Id"]
                    return pmc_id
        return None
    except Exception as e:
        logger.debug(f"elink failed for PMID {pmid}: {e}")
        return None

# ==================== PDF DOWNLOAD ====================
def download_pdf_from_pmc(pmcid: str, pmid: str) -> bool:
    """
    Download PDF from PMC using official endpoint.
    """
    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/pdf/"
    
    try:
        logger.info(f"Attempting PDF download for PMC{pmcid}...")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Research-Agent/1.0)",
            "Accept": "application/pdf"
        }
        
        response = requests.get(pdf_url, headers=headers, timeout=30, stream=True)
        
        if response.status_code == 200 and response.headers.get('Content-Type', '').startswith('application/pdf'):
            output_path = SAVE_DIR / f"PMC{pmcid}_{pmid}.pdf"
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = output_path.stat().st_size
            if file_size > 1000:  # At least 1KB
                logger.info(f"✅ Downloaded: {output_path.name} ({file_size:,} bytes)")
                return True
            else:
                logger.warning(f"File too small, likely not a valid PDF")
                output_path.unlink()
                return False
        else:
            logger.warning(f"PDF not available (status: {response.status_code})")
            return False
            
    except Exception as e:
        logger.warning(f"PDF download error for PMC{pmcid}: {e}")
        return False

# ==================== SAVE ABSTRACT AS TEXT ====================
def save_abstract_text(metadata: Dict, pmid: str, pmcid: Optional[str] = None):
    """Save abstract and metadata as text file."""
    try:
        filename = f"PMC{pmcid}_{pmid}.txt" if pmcid else f"PMID_{pmid}.txt"
        output_path = SAVE_DIR / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"PMID: {metadata['pmid']}\n")
            if pmcid:
                f.write(f"PMCID: PMC{pmcid}\n")
            f.write(f"Title: {metadata['title']}\n")
            f.write(f"Journal: {metadata['journal']}\n")
            f.write(f"Authors: {', '.join(metadata['authors'][:5])}\n")
            f.write(f"\nAbstract:\n{metadata['abstract']}\n")
        
        logger.info(f"📄 Saved abstract: {filename}")
        return True
    except Exception as e:
        logger.warning(f"Could not save abstract: {e}")
        return False

# ==================== MAIN EXECUTION ====================
def main():
    metadata = load_metadata()
    
    pmids = search_pubmed(SEARCH_QUERY, MAX_RESULTS)
    
    if not pmids:
        logger.error("No PMIDs found!")
        return
    
    for idx, pmid in enumerate(pmids, 1):
        time.sleep(REQUEST_DELAY)
        
        logger.info(f"\n[{idx}/{len(pmids)}] Processing PMID: {pmid}")
        
        # Fetch metadata
        article_meta = fetch_article_metadata(pmid)
        if not article_meta:
            logger.warning(f"  Could not fetch metadata for PMID {pmid}")
            metadata["failed"].append(pmid)
            continue
        
        logger.info(f"  Title: {article_meta['title'][:60]}...")
        
        time.sleep(REQUEST_DELAY)
        
        # Get PMC ID
        pmcid = get_pmc_id(pmid)
        
        if pmcid:
            logger.info(f"  Found PMCID: {pmcid}")
            time.sleep(REQUEST_DELAY)
            
            # Try to download PDF
            pdf_success = download_pdf_from_pmc(pmcid, pmid)
            
            # Always save abstract
            save_abstract_text(article_meta, pmid, pmcid)
            
            if pdf_success:
                metadata["downloaded"].append({
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": article_meta
                })
            else:
                metadata["failed"].append(pmid)
        else:
            logger.warning(f"  No PMCID found for PMID {pmid}")
            # Still save abstract
            save_abstract_text(article_meta, pmid)
            metadata["no_pmc"].append(pmid)
        
        metadata["total_requests"] += 1
        save_metadata(metadata)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Downloaded PDFs: {len(metadata['downloaded'])}")
    logger.info(f"Failed: {len(metadata['failed'])}")
    logger.info(f"No PMC: {len(metadata['no_pmc'])}")
    logger.info(f"Total processed: {len(pmids)}")
    logger.info(f"Metadata saved to: {METADATA_FILE}")

if __name__ == "__main__":
    main()