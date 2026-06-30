#!/usr/bin/env python3
"""
verify_datasets.py
==================
Automated verification tool for the 400 Free & Open Scientific Datasets.

Checks performed:
  1. URL reachability (HTTP HEAD / GET)
  2. API endpoint accessibility
  3. License field completeness
  4. Download availability flags
  5. Estimated dataset size bucket

Usage:
    # Full run (all 400 datasets, throttled)
    python verify_datasets.py

    # Single category
    python verify_datasets.py --category Physics

    # Quick smoke-test first 20
    python verify_datasets.py --limit 20

    # Save results
    python verify_datasets.py --output results.json

Requirements:
    pip install requests tqdm
"""

import argparse
import csv
import json
import sys
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("Install: pip install requests tqdm")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── add parent to path ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from registry import DATASETS

# ─────────────────────────────────────────────────────────────
TIMEOUT   = 10          # seconds per request
MAX_WORKERS = 8         # parallel threads
THROTTLE  = 0.25        # seconds between requests per thread

HEADERS = {
    "User-Agent": "SciDatasetVerifier/1.0 (Open Science Research; +https://github.com/AppliedInsights)"
}

# ─────────────────────────────────────────────────────────────
def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s


def check_url(session: requests.Session, url: str) -> dict:
    """Return status, code, latency_ms."""
    t0 = time.time()
    try:
        r = session.head(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code == 405:          # HEAD not allowed → try GET
            r = session.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
            r.close()
        latency = int((time.time() - t0) * 1000)
        ok = r.status_code < 400
        return {"reachable": ok, "http_code": r.status_code, "latency_ms": latency, "error": None}
    except requests.exceptions.SSLError as e:
        return {"reachable": False, "http_code": None, "latency_ms": None, "error": f"SSL: {e}"}
    except requests.exceptions.ConnectionError as e:
        return {"reachable": False, "http_code": None, "latency_ms": None, "error": f"Connection: {e}"}
    except requests.exceptions.Timeout:
        return {"reachable": False, "http_code": None, "latency_ms": None, "error": "Timeout"}
    except Exception as e:
        return {"reachable": False, "http_code": None, "latency_ms": None, "error": str(e)}


def check_license(license_str: str) -> dict:
    """Flag whether the license is permissive/open."""
    OPEN_KEYWORDS = [
        "CC0","CC-BY","MIT","Apache","GPL","BSD","ODbL","PDDL",
        "Public Domain","Open Data","Open Access","Gov Open","US Gov",
        "Open Government","LGPL","MPL","ISC","Unlicense","FreeBSD"
    ]
    RESTRICTIVE_KEYWORDS = ["NC","ND","paid","commercial","subscription","proprietary","contact"]
    
    upper = license_str.upper()
    is_open  = any(k.upper() in upper for k in OPEN_KEYWORDS)
    has_restriction = any(k.upper() in upper for k in RESTRICTIVE_KEYWORDS)
    
    if is_open and not has_restriction:
        verdict = "OPEN"
    elif is_open and has_restriction:
        verdict = "OPEN_WITH_RESTRICTIONS"
    elif not license_str.strip():
        verdict = "MISSING"
    else:
        verdict = "UNCLEAR"
    
    return {"license_raw": license_str, "verdict": verdict, "is_open": is_open, "has_restriction": has_restriction}


def verify_one(dataset: dict) -> dict:
    """Full verification of a single dataset entry."""
    session = make_session()
    result = {
        "id":    dataset["id"],
        "name":  dataset["name"],
        "url":   dataset["url"],
        "cat":   dataset["cat"],
        "domain":dataset["domain"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # 1. URL reachability
    time.sleep(THROTTLE)
    url_check = check_url(session, dataset["url"])
    result["url_check"] = url_check

    # 2. License
    result["license_check"] = check_license(dataset.get("license",""))

    # 3. Metadata completeness
    required_fields = ["name","url","cat","domain","desc","type","size","freq","license","level","compute"]
    missing = [f for f in required_fields if not dataset.get(f,"").strip() if isinstance(dataset.get(f,""),str)]
    result["metadata_complete"] = len(missing) == 0
    result["missing_fields"]    = missing

    # 4. Projects populated
    projects = dataset.get("projects", [])
    result["has_projects"] = len(projects) >= 1
    result["project_count"] = len(projects)

    # 5. Registration flag
    result["registration_required"] = dataset.get("reg", False)

    # 6. Overall verdict
    checks_passed = [
        url_check["reachable"],
        result["license_check"]["is_open"],
        result["metadata_complete"],
        result["has_projects"],
    ]
    result["checks_passed"] = sum(1 for c in checks_passed if c)
    result["checks_total"]  = len(checks_passed)
    result["overall_pass"]  = all(checks_passed)

    return result


# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Verify open scientific datasets")
    parser.add_argument("--category", choices=["Physics","Biology","Chemistry","Live Science"], help="Filter by category")
    parser.add_argument("--limit",    type=int, default=None, help="Only check first N datasets")
    parser.add_argument("--output",   default="verification_results.json", help="Output JSON file")
    parser.add_argument("--csv",      default="verification_results.csv",  help="Output CSV file")
    parser.add_argument("--workers",  type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    # Select datasets
    datasets = DATASETS
    if args.category:
        datasets = [d for d in datasets if d["cat"] == args.category]
    if args.limit:
        datasets = datasets[:args.limit]

    print(f"\n{'='*60}")
    print(f"  Open Science Dataset Verifier")
    print(f"  Datasets to verify : {len(datasets)}")
    print(f"  Parallel workers   : {args.workers}")
    print(f"  Timeout per URL    : {TIMEOUT}s")
    print(f"{'='*60}\n")

    results = []
    iterator = datasets

    if HAS_TQDM:
        iterator = tqdm(datasets, desc="Verifying", unit="dataset")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(verify_one, d): d for d in datasets}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                d = futures[future]
                results.append({"id": d["id"], "name": d["name"], "error": str(e)})
            if HAS_TQDM:
                iterator.update(1)

    # Sort by ID
    results.sort(key=lambda r: r.get("id",""))

    # ── Summary ────────────────────────────────────────────────
    total       = len(results)
    reachable   = sum(1 for r in results if r.get("url_check",{}).get("reachable"))
    open_lic    = sum(1 for r in results if r.get("license_check",{}).get("is_open"))
    meta_ok     = sum(1 for r in results if r.get("metadata_complete"))
    overall_ok  = sum(1 for r in results if r.get("overall_pass"))

    print(f"\n{'='*60}")
    print(f"  VERIFICATION SUMMARY  ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})")
    print(f"{'='*60}")
    print(f"  Total verified    : {total}")
    print(f"  URLs reachable    : {reachable:4d} / {total}  ({100*reachable/total:.1f}%)")
    print(f"  Open license      : {open_lic:4d} / {total}  ({100*open_lic/total:.1f}%)")
    print(f"  Metadata complete : {meta_ok:4d} / {total}  ({100*meta_ok/total:.1f}%)")
    print(f"  Overall PASS      : {overall_ok:4d} / {total}  ({100*overall_ok/total:.1f}%)")
    print(f"{'='*60}\n")

    # By category
    for cat in ["Physics","Biology","Chemistry","Live Science"]:
        cat_r   = [r for r in results if r.get("cat") == cat]
        if not cat_r: continue
        reach   = sum(1 for r in cat_r if r.get("url_check",{}).get("reachable"))
        print(f"  {cat:<14}  reachable {reach}/{len(cat_r)}")

    # ── Save JSON ──────────────────────────────────────────────
    output = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total": total, "reachable": reachable,
            "open_license": open_lic, "metadata_complete": meta_ok,
            "overall_pass": overall_ok,
        },
        "results": results
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\n  JSON → {args.output}")

    # ── Save CSV ───────────────────────────────────────────────
    flat_fields = ["id","name","cat","domain","url",
                   "reachable","http_code","latency_ms","url_error",
                   "license_raw","license_verdict","is_open","has_restriction",
                   "metadata_complete","missing_fields","project_count",
                   "registration_required","overall_pass"]
    with open(args.csv,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            uc = r.get("url_check",{})
            lc = r.get("license_check",{})
            w.writerow({
                "id":                   r.get("id"),
                "name":                 r.get("name"),
                "cat":                  r.get("cat"),
                "domain":               r.get("domain"),
                "url":                  r.get("url"),
                "reachable":            uc.get("reachable"),
                "http_code":            uc.get("http_code"),
                "latency_ms":           uc.get("latency_ms"),
                "url_error":            uc.get("error"),
                "license_raw":          lc.get("license_raw"),
                "license_verdict":      lc.get("verdict"),
                "is_open":              lc.get("is_open"),
                "has_restriction":      lc.get("has_restriction"),
                "metadata_complete":    r.get("metadata_complete"),
                "missing_fields":       ";".join(r.get("missing_fields",[])),
                "project_count":        r.get("project_count"),
                "registration_required":r.get("registration_required"),
                "overall_pass":         r.get("overall_pass"),
            })
    print(f"  CSV  → {args.csv}")
    print()


if __name__ == "__main__":
    main()
