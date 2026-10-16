import requests
import io
import re
from bs4 import BeautifulSoup
import pdfplumber

PMCID = "PMC11655950"

url = f"https://pmc.ncbi.nlm.nih.gov/articles/{PMCID}/pdf/"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

response = requests.get(url, headers=headers, timeout=30)
print("Status code:", response.status_code)
print("Final URL (after redirect):", response.url)
print("Content-Type:", response.headers.get("Content-Type"))

if response.status_code == 200 and "pdf" in response.headers.get("Content-Type", "").lower():
    with pdfplumber.open(io.BytesIO(response.content)) as pdf:
        pages_text = [page.extract_text() or "" for page in pdf.pages]
    text = " ".join(pages_text)
    text = re.sub(r"\s+", " ", text).strip()
    print("\n=== Extracted text (first 500 chars) ===")
    print(text[:500])
    print("\nTotal length:", len(text), "characters")
else:
    print("\nDid not get a PDF back.")
    print(response.text[:500])