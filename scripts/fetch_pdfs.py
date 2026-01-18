
import os
import time
import requests
from Bio import Entrez

# --- Configuration ---
Entrez.email = "himath.nimpura@gmail.com"  # Required by NCBI
SEARCH_QUERY = "Alzheimer's disease therapeutic pathways"
MAX_RESULTS = 10  # Start small for your prototype
SAVE_DIR = "data/abstracts"

os.makedirs(SAVE_DIR, exist_ok=True)

def search_pubmed(query, limit):
    print(f"Searching PubMed for: {query}...")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=limit, usehistory="y")
    results = Entrez.read(handle)
    handle.close()
    return results["IdList"]

def get_pmc_id(pmid):
    """Link PubMed ID to PubMed Central ID for PDF access."""
    try:
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
        results = Entrez.read(handle)
        handle.close()
        if results[0]["LinkSetDb"]:
            return results[0]["LinkSetDb"][0]["Link"][0]["Id"]
    except Exception:
        return None
    return None

def download_pdf(pmcid, filename):
    # Updated 2026 URL pattern for PMC PDF service
    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/pdf/"
    
    # Enhanced headers to avoid 403 Forbidden
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Accept": "application/pdf",
        "Referer": "https://www.ncbi.nlm.nih.gov"
    }
    
    try:
        # Using a session can help manage cookies/permissions
        with requests.Session() as session:
            response = session.get(pdf_url, headers=headers, timeout=20, allow_redirects=True)
            
            if response.status_code == 200:
                with open(os.path.join(SAVE_DIR, f"{filename}.pdf"), "wb") as f:
                    f.write(response.content)
                return True
            else:
                # If 403 persists, it might require the PMC OAI service
                print(f"Status {response.status_code} for PMC{pmcid}. PMC may require an API key or Web-browser download.")
    except Exception as e:
        print(f"Error downloading PMC{pmcid}: {e}")
    return False

# --- Execution ---
pmids = search_pubmed(SEARCH_QUERY, MAX_RESULTS)
downloaded_count = 0

for pmid in pmids:
    pmcid = get_pmc_id(pmid)
    if pmcid:
        print(f"Found PMC ID: {pmcid} for PMID: {pmid}. Downloading...")
        if download_pdf(pmcid, pmid):
            print(f"Successfully saved {pmid}.pdf")
            downloaded_count += 1
            time.sleep(1) # Respect NCBI Rate Limits
    else:
        print(f"PMID {pmid} has no free PMC PDF available. Skipping.")

print(f"\nFinished! Downloaded {downloaded_count} PDFs to {SAVE_DIR}.")