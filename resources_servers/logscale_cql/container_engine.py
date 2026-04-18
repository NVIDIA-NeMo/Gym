"""
Execute CQL queries against a live LogScale container via the Query Jobs REST API.

With AUTHENTICATION_METHOD=none, no tokens are needed for the query API.
Returns {success, n_rows, preview, exec_time_ms, error?} dicts.
"""

import re
import time
from typing import Optional

import pandas as pd
import requests


class LogScaleContainerEngine:
    """Thin client that submits CQL to a running LogScale instance and polls for results."""

    def __init__(self, base_url: str = "http://localhost:8080", repository: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.repository = repository or self._discover_sandbox()
        self._known_repos: set[str] | None = None

    def _get_known_repos(self) -> set[str]:
        if self._known_repos is None:
            try:
                resp = requests.post(
                    f"{self.base_url}/graphql",
                    json={"query": "{ searchDomains { name } }"},
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
                resp.raise_for_status()
                self._known_repos = {
                    d["name"] for d in resp.json()["data"]["searchDomains"]
                }
            except Exception:
                self._known_repos = set()
        return self._known_repos

    def _resolve_repo(self, cql: str) -> str:
        """If the CQL contains ``#repo=<name>`` and that repo exists, use it."""
        m = re.search(r'#repo\s*=\s*"?(\w+)"?', cql)
        if m:
            repo_name = m.group(1)
            if repo_name in self._get_known_repos():
                return repo_name
        return self.repository

    # ------------------------------------------------------------------
    # Repository discovery
    # ------------------------------------------------------------------

    def _discover_sandbox(self) -> str:
        """Find a sandbox repo with data via GraphQL (works without auth)."""
        resp = requests.post(
            f"{self.base_url}/graphql",
            json={"query": "{ searchDomains { name } }"},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        domains = resp.json().get("data", {}).get("searchDomains", [])
        sandboxes = [d["name"] for d in domains if d.get("name", "").startswith("sandbox_")]
        if not sandboxes:
            if domains:
                return domains[0]["name"]
            raise RuntimeError("No repositories found on the LogScale instance")
        if len(sandboxes) == 1:
            return sandboxes[0]
        for name in reversed(sandboxes):
            try:
                r = requests.post(
                    f"{self.base_url}/api/v1/repositories/{name}/queryjobs",
                    json={"queryString": "count()", "start": "7days", "isLive": False},
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    timeout=10,
                )
                if r.status_code != 200:
                    continue
                job_id = r.json().get("id")
                if not job_id:
                    continue
                for _ in range(20):
                    pr = requests.get(
                        f"{self.base_url}/api/v1/repositories/{name}/queryjobs/{job_id}",
                        headers={"Accept": "application/json"},
                        timeout=10,
                    ).json()
                    if pr.get("done"):
                        events = pr.get("events", [])
                        if events and int(events[0].get("_count", 0)) > 0:
                            return name
                        break
                    time.sleep(0.3)
            except Exception:
                continue
        return sandboxes[-1]

    # ------------------------------------------------------------------
    # Ingest token discovery (used by setup.sh helper)
    # ------------------------------------------------------------------

    def get_ingest_token(self) -> str:
        """Return the default ingest token for the current repository."""
        query = (
            '{ searchDomain(name: "%s") { ... on Repository { ingestTokens { token } } } }'
            % self.repository
        )
        resp = requests.post(
            f"{self.base_url}/graphql",
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        tokens = (
            resp.json()
            .get("data", {})
            .get("searchDomain", {})
            .get("ingestTokens", [])
        )
        if not tokens:
            raise RuntimeError(f"No ingest tokens found for repository {self.repository}")
        return tokens[0]["token"]

    # ------------------------------------------------------------------
    # Query validation (GraphQL validateQuery)
    # ------------------------------------------------------------------

    def validate_query(self, cql: str) -> dict:
        """Check whether *cql* is syntactically valid LogScale.

        Returns {"is_valid": bool, "diagnostics": [{"message": str, "severity": str}]}
        """
        import json as _json

        gql = (
            "{ validateQuery("
            "queryString: $qs, version: legacy, isLive: false"
            ") { isValid diagnostics { message severity } } }"
        )
        # Use GraphQL variables so we never have to manually escape the query string
        payload = {
            "query": "query ($qs: String!) " + gql,
            "variables": {"qs": cql},
        }

        try:
            resp = requests.post(
                f"{self.base_url}/graphql",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            vq = (result.get("data") or {}).get("validateQuery") or {}
            return {
                "is_valid": bool(vq.get("isValid")),
                "diagnostics": vq.get("diagnostics") or [],
            }
        except Exception as e:
            return {"is_valid": False, "diagnostics": [{"message": str(e), "severity": "error"}]}

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def execute(self, cql: str) -> dict:
        """Execute *cql* and return a result dict compatible with the Gym verify flow."""
        repo = self._resolve_repo(cql)
        t0 = time.time()
        try:
            job_id = self._create_job(cql, repo=repo)
            events = self._poll_job(job_id, repo=repo)
            elapsed = time.time() - t0
            preview = self._format_preview(events, max_rows=15)
            return {
                "success": True,
                "n_rows": len(events),
                "preview": preview,
                "exec_time_ms": round(elapsed * 1000),
            }
        except Exception as e:
            return {
                "success": False,
                "n_rows": 0,
                "preview": "",
                "exec_time_ms": round((time.time() - t0) * 1000),
                "error": str(e),
            }

    def _create_job(self, cql: str, start: str = "7days", repo: str | None = None) -> str:
        target = repo or self.repository
        resp = requests.post(
            f"{self.base_url}/api/v1/repositories/{target}/queryjobs",
            json={"queryString": cql, "start": start, "isLive": False},
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("id")
        if not job_id:
            raise RuntimeError(f"No job ID returned: {data}")
        return job_id

    def _poll_job(self, job_id: str, max_polls: int = 60, interval: float = 0.5, repo: str | None = None) -> list[dict]:
        target = repo or self.repository
        for _ in range(max_polls):
            resp = requests.get(
                f"{self.base_url}/api/v1/repositories/{target}/queryjobs/{job_id}",
                headers={"Accept": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("done"):
                return data.get("events", [])
            time.sleep(interval)
        raise RuntimeError(f"Query job {job_id} did not complete within {max_polls * interval}s")

    @staticmethod
    def _format_preview(events: list[dict], max_rows: int = 15) -> str:
        if not events:
            return "(no results)"
        skip = {
            "@rawstring", "@id", "@timestamp.nanos", "@timezone",
            "#repo", "#host", "@ingesttimestamp",
        }
        rows = events[:max_rows]
        cols: list[str] = []
        for e in rows:
            for k in e:
                if k not in skip and k not in cols:
                    cols.append(k)
        df = pd.DataFrame(rows, columns=cols)
        return df.to_string(index=False)
