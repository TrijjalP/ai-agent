import requests, json

response = requests.get(
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
    params={
        "query": "tuberculosis drug resistance",
        "format": "json",
        "pageSize": 3,
        "resultType": "core"
    },
    timeout=15
)
data = response.json()
results = data.get("resultList", {}).get("result", [])

for r in results:
    print("\nTitle:", r.get("title", "")[:60])
    print("pmcid:", r.get("pmcid"))
    print("inEPMC:", r.get("inEPMC"))
    print("isOpenAccess:", r.get("isOpenAccess"))
    print("fullTextUrlList:", json.dumps(r.get("fullTextUrlList"), indent=2))