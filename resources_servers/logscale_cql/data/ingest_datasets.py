#!/usr/bin/env python3
"""Ingest all generated CSV datasets into a running LogScale container.

Reads every query_*.csv from the datasets directory and POSTs them to
LogScale's structured ingest API.  Optionally verifies each dataset by
running its reference CQL query from the manifest.

Usage:
    python ingest_datasets.py
    python ingest_datasets.py --datasets-dir datasets --logscale-url http://localhost:8080
    python ingest_datasets.py --verify
    python ingest_datasets.py --csv-glob "query_01.csv,query_02.csv"
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATASETS_DIR = SCRIPT_DIR / "datasets"
DEFAULT_LOGSCALE_URL = "http://localhost:8080"
CHUNK_SIZE = 500

_event_counter = 0


def _make_recent_timestamp() -> str:
    """Generate a recent ISO timestamp that the demo container will accept.

    Spreads events across the last hour so they have distinct times.
    """
    global _event_counter
    _event_counter += 1
    offset_ms = _event_counter * 10
    now_ms = int(time.time() * 1000) - offset_ms
    dt = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now_ms % 1000:03d}Z"


# ── LogScale discovery & repo management ─────────────────────────

def discover_repository(base_url: str) -> str:
    resp = requests.post(
        f"{base_url}/graphql",
        json={"query": "{ searchDomains { name } }"},
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    resp.raise_for_status()
    domains = resp.json()["data"]["searchDomains"]
    sandboxes = [d["name"] for d in domains if d["name"].startswith("sandbox_")]
    if sandboxes:
        return sandboxes[0]
    if domains:
        return domains[0]["name"]
    raise RuntimeError("No repositories found on the LogScale instance")


def ensure_repo_exists(base_url: str, repo_name: str) -> None:
    """Create a repository if it doesn't already exist."""
    resp = requests.post(
        f"{base_url}/graphql",
        json={"query": "{ searchDomains { name } }"},
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    existing = [d["name"] for d in resp.json()["data"]["searchDomains"]]
    if repo_name in existing:
        return
    mutation = (
        'mutation { createRepository(name: "%s") { repository { name } } }' % repo_name
    )
    resp = requests.post(
        f"{base_url}/graphql",
        json={"query": mutation},
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    resp.raise_for_status()
    print(f"  Created repository: {repo_name}")


def fetch_ingest_token(base_url: str, repo: str) -> str:
    query = (
        '{ searchDomain(name: "%s") '
        "{ ... on Repository { ingestTokens { token } } } }" % repo
    )
    resp = requests.post(
        f"{base_url}/graphql",
        json={"query": query},
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    resp.raise_for_status()
    tokens = resp.json()["data"]["searchDomain"]["ingestTokens"]
    if not tokens:
        raise RuntimeError(f"No ingest tokens found for repository '{repo}'")
    return tokens[0]["token"]


def detect_repo_for_csv(csv_path: Path) -> str | None:
    """Read the #repo column from a CSV and return the repo name, or None."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("#repo", "").strip()
            if val:
                return val
    return None


# ── CSV → structured events ──────────────────────────────────────

def csv_to_payloads(csv_path: Path) -> list[dict]:
    """Convert a CSV file into LogScale humio-structured payloads.

    Only ``host`` and ``#event_simpleName`` are sent as tags to stay within
    the demo container's tag-combination limit.  All other columns
    (including ``#repo``) are sent as attributes.
    """
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return []

    by_etype: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        etype = row.get("#event_simpleName", "unknown")

        attrs = {}
        for k, v in row.items():
            if not v or k == "timestamp" or k == "#event_simpleName":
                continue
            attrs[k] = v

        ts = _make_recent_timestamp()

        by_etype[etype].append({"timestamp": ts, "attributes": attrs})

    payloads = []
    for etype, events in by_etype.items():
        tags = {"host": "sandbox", "#event_simpleName": etype}
        for i in range(0, len(events), CHUNK_SIZE):
            payloads.append({"tags": tags, "events": events[i : i + CHUNK_SIZE]})

    return payloads


# ── Ingest ────────────────────────────────────────────────────────

def ingest_csv(base_url: str, token: str, csv_path: Path) -> int:
    """Ingest one CSV file. Returns the number of events ingested."""
    payloads = csv_to_payloads(csv_path)
    total = 0
    for payload in payloads:
        n = len(payload["events"])
        resp = requests.post(
            f"{base_url}/api/v1/ingest/humio-structured",
            json=[payload],
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Ingest failed ({resp.status_code}): {resp.text[:200]}"
            )
        total += n
    return total


# ── Verify via query job ─────────────────────────────────────────

def run_query(base_url: str, repo: str, cql: str, start: str = "365days") -> int:
    """Execute a CQL query and return the number of result rows."""
    job = requests.post(
        f"{base_url}/api/v1/repositories/{repo}/queryjobs",
        json={"queryString": cql, "start": start, "isLive": False},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=30,
    )
    job.raise_for_status()
    job_id = job.json().get("id")
    if not job_id:
        return -1

    for _ in range(60):
        r = requests.get(
            f"{base_url}/api/v1/repositories/{repo}/queryjobs/{job_id}",
            headers={"Accept": "application/json"},
            timeout=30,
        ).json()
        if r.get("done"):
            return len(r.get("events", []))
        time.sleep(0.5)

    return -1


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest generated CSV datasets into a LogScale container",
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=str(DEFAULT_DATASETS_DIR),
        help="Directory containing query_*.csv files (default: datasets/)",
    )
    parser.add_argument(
        "--logscale-url",
        type=str,
        default=DEFAULT_LOGSCALE_URL,
        help="LogScale base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After ingesting, run each query's reference CQL to verify results",
    )
    parser.add_argument(
        "--csv-glob",
        type=str,
        default=None,
        help="Comma-separated list of CSV filenames to ingest (default: all *.csv)",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help="Path to queries JSON file (default: queries_logscale.json next to datasets dir)",
    )
    args = parser.parse_args()

    base_url = args.logscale_url.rstrip("/")
    datasets_dir = Path(args.datasets_dir)

    # Discover default repo
    print("=== Ingest Datasets into LogScale ===\n")
    print(f"  LogScale URL:  {base_url}")
    print(f"  Datasets dir:  {datasets_dir.resolve()}\n")

    print("[..] Discovering default repository...")
    default_repo = discover_repository(base_url)
    print(f"  Default repo: {default_repo}")

    # Collect CSV files
    if args.csv_glob:
        csv_files = sorted(
            datasets_dir / name.strip()
            for name in args.csv_glob.split(",")
        )
    else:
        csv_files = sorted(datasets_dir.glob("*.csv"))

    if not csv_files:
        print("[FAIL] No CSV files found.")
        sys.exit(1)

    # Detect which repos are needed and create them
    csv_repo_map: dict[str, str] = {}
    needed_repos: set[str] = set()
    for csv_path in csv_files:
        repo_name = detect_repo_for_csv(csv_path)
        if repo_name:
            csv_repo_map[csv_path.name] = repo_name
            needed_repos.add(repo_name)

    if needed_repos:
        print(f"\n[..] Setting up {len(needed_repos)} additional repos: {sorted(needed_repos)}")
        for repo_name in sorted(needed_repos):
            ensure_repo_exists(base_url, repo_name)

    # Fetch ingest tokens for all repos
    repo_tokens: dict[str, str] = {}
    all_repos = {default_repo} | needed_repos
    print(f"\n[..] Fetching ingest tokens for {len(all_repos)} repos...")
    for repo_name in sorted(all_repos):
        token = fetch_ingest_token(base_url, repo_name)
        repo_tokens[repo_name] = token
        print(f"  {repo_name}: {token[:12]}...")

    print(f"\n[..] Ingesting {len(csv_files)} datasets...\n")

    # Load manifest for verification
    manifest_path = datasets_dir / "manifest.json"
    manifest_by_file = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest_by_file = {m["dataset"]: m for m in manifest if "dataset" in m}

    total_events = 0
    success = 0
    verify_pass = 0
    verify_total = 0
    verified_files: set[str] = set()

    # Phase 1: Ingest all CSVs
    ingested: list[tuple[str, str, dict]] = []
    for csv_path in csv_files:
        label = csv_path.name
        info = manifest_by_file.get(label, {})
        target_repo = csv_repo_map.get(label, default_repo)
        target_token = repo_tokens[target_repo]

        try:
            n = ingest_csv(base_url, target_token, csv_path)
            total_events += n
            success += 1
            print(f"  [{label}] {n} events -> {target_repo}")
            ingested.append((label, target_repo, info))
        except Exception as e:
            print(f"  [{label}] ERROR: {e}")

    # Phase 2: Verify (after all data has been indexed)
    if args.verify:
        print(f"\n[..] Waiting for indexing...")
        time.sleep(3)
        print(f"[..] Verifying {len(ingested)} datasets...\n")

        for label, target_repo, info in ingested:
            if not info.get("reference"):
                continue
            prompt = info.get("prompt", "")
            prompt_short = (prompt[:60] + "...") if len(prompt) > 60 else prompt

            try:
                rows = run_query(base_url, target_repo, info["reference"])
                verify_total += 1
                if rows > 0:
                    verify_pass += 1
                    verified_files.add(label)
                    print(f"  [{label}] PASS ({rows} rows) -> {target_repo}")
                elif rows == 0:
                    print(f"  [{label}] FAIL (0 rows) -> {target_repo}")
                else:
                    print(f"  [{label}] ERROR -> {target_repo}")
                if prompt_short:
                    print(f"           {prompt_short}")
            except Exception as e:
                print(f"  [{label}] ERROR: {e}")

    # Write verified-only queries file for benchmarking
    if args.verify:
        queries_path = Path(args.queries_file) if args.queries_file else datasets_dir.parent / "queries_logscale.json"
        if queries_path.exists():
            all_queries = json.loads(queries_path.read_text())
            verified = []
            for m in manifest_by_file.values():
                idx = m.get("index")
                if idx is not None and idx < len(all_queries) and m.get("dataset") in verified_files:
                    verified.append(all_queries[idx])
            verified_path = datasets_dir.parent / "queries_verified.json"
            verified_path.write_text(json.dumps(verified, indent=4))
            print(f"\n[OK] Wrote {len(verified)} verified Q&A pairs -> {verified_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Ingested {total_events} total events from {success}/{len(csv_files)} datasets")
    print(f"Repos used: {sorted(all_repos)}")
    if args.verify:
        print(f"CQL verification: {verify_pass}/{verify_total} queries returned results")

    # Final count check per repo
    print("\n[..] Verifying event counts per repo...")
    time.sleep(2)
    for repo_name in sorted(all_repos):
        try:
            job_id = requests.post(
                f"{base_url}/api/v1/repositories/{repo_name}/queryjobs",
                json={"queryString": "count()", "start": "365days", "isLive": False},
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=30,
            ).json().get("id")
            for _ in range(30):
                r = requests.get(
                    f"{base_url}/api/v1/repositories/{repo_name}/queryjobs/{job_id}",
                    headers={"Accept": "application/json"}, timeout=30,
                ).json()
                if r.get("done"):
                    events = r.get("events", [])
                    count = events[0].get("_count", 0) if events else 0
                    print(f"  {repo_name}: {count} events")
                    break
                time.sleep(0.5)
        except Exception:
            print(f"  {repo_name}: (error checking count)")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
