# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare BEAVER benchmark rows: questions from Hugging Face, schema from the MySQL dumps.

Per-row output: ``{task_id, id, question, gt_sql, db_id, domain, category, detailed_category, evidence,
evidence_section, sql_context}``. ``task_id`` is BEAVER's own question id, ``id`` kept as its source copy: it already
names its domain (``dw_12``, ``dw_real_12``), so it is unique across domains and doesn't depend on row order. The prompt (``prompts/schema_only.yaml``) shows the question, its evidence when there is any, and the
``CREATE TABLE`` statements of the question's database (``sql_context``): the schema-only setup of auto-ontology-eval.

Sources, both gated on Hugging Face (MIT; accept the terms on each dataset page, then ``hf auth login``), pinned:

- questions: ``beaverbench/beaver-query`` at ``HF_QUERY_REVISION``, one parquet file per domain;
- schema: ``beaver_db.zip`` from ``beaverbench/beaver-table`` (see ``resources_servers/beaver_sql/setup_beaver_mysql``),
  shared with the grader's download cache.

Sampling follows BEAVER's own download script (``data/download_hf.py``): ``random.Random(77).sample(rows, n)`` over
a domain's rows in file order. For dw (5,787 questions) samples of up to 1,365 nest: the first 100 of a 300-question
sample are the 100-question sample, and the first 300 of a 400-question sample are the 300 (Python's sampler changes
method above 1,365). ``sample=0`` keeps every question.

Evidence is each question's ``domain_knowledge`` items, one per line, as auto-ontology-eval builds it. BEAVER's
leaderboard (Setting 0) gives no hints: prepare with ``include_evidence=False`` for numbers comparable with it.

Prompt construction and evidence adapted from auto-ontology-eval (https://github.com/NVIDIA/auto-ontology-eval,
``ontology_sql_eval/gym/tasks.py``, ``ddl.py`` and ``scripts/seed/seed_beaverbench.py`` at e727e4d6e, Apache-2.0).
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from resources_servers.beaver_sql.setup_beaver_mysql import DOMAIN_DB, create_table_statements, ensure_beaver_dumps


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
DEFAULT_CACHE_DIR = BENCHMARK_DIR.parent.parent / "resources_servers" / "beaver_sql" / ".beaver"

HF_QUERY_REPO = "beaverbench/beaver-query"
HF_QUERY_REVISION = "3c8eb607ee2b7709ad4b1d53243e04af1b9eda4b"  # pragma: allowlist secret
DOMAINS = tuple(DOMAIN_DB)
SAMPLE_SEED = 77


def output_fpath_for(domains: Sequence[str], sample: int, seed: int, include_evidence: bool) -> Path:
    """``data/beaver_<domains>_<n or all>_seed<seed>[_noevidence].jsonl``: one file per setting, so a cached file
    prepared with other settings is never reused."""
    size = str(sample) if sample > 0 else "all"
    suffix = "" if include_evidence else "_noevidence"
    return DATA_DIR / f"beaver_{'-'.join(domains)}_{size}_seed{seed}{suffix}.jsonl"


def load_questions(domain: str) -> List[Dict[str, Any]]:
    """A domain's questions, in file order, from the pinned revision."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_QUERY_REPO,
        filename=f"data/{domain}-00000-of-00001.parquet",
        repo_type="dataset",
        revision=HF_QUERY_REVISION,
    )
    return pq.read_table(path).to_pylist()


def sample_questions(rows: List[Dict[str, Any]], sample: int, seed: int = SAMPLE_SEED) -> List[Dict[str, Any]]:
    if sample <= 0 or sample >= len(rows):
        return rows
    return random.Random(seed).sample(rows, sample)


def evidence_text(domain_knowledge: Any) -> str:
    """BEAVER's ``domain_knowledge`` (a JSON list, stored as a string) as one item per line."""
    if isinstance(domain_knowledge, str):
        try:
            domain_knowledge = json.loads(domain_knowledge)
        except json.JSONDecodeError:
            return ""
    if not isinstance(domain_knowledge, list):
        return ""
    return "\n".join(str(item).strip() for item in domain_knowledge if str(item).strip())


def schema_dump(dump_path: Path) -> str:
    """A database's ``CREATE TABLE`` statements, verbatim, wrapped as a SQL dump (``BEGIN TRANSACTION;`` ...
    ``COMMIT;``), as auto-ontology-eval's schema-only arm shows them."""
    body = "\n\n".join(create_table_statements(dump_path))
    return f"BEGIN TRANSACTION;\n{body}\nCOMMIT;"


def build_rows(
    domain_rows: Dict[str, List[Dict[str, Any]]], schemas: Dict[str, str], include_evidence: bool
) -> List[Dict[str, Any]]:
    rows = []
    seen: Dict[str, str] = {}
    for domain, questions in domain_rows.items():
        for q in questions:
            task_id = str(q["id"])
            if task_id in seen:
                raise ValueError(f"BEAVER question id {task_id!r} appears in {seen[task_id]!r} and {domain!r}")
            seen[task_id] = domain
            db_id = str(q.get("db") or DOMAIN_DB[domain]).strip() or DOMAIN_DB[domain]
            evidence = evidence_text(q.get("domain_knowledge")) if include_evidence else ""
            rows.append(
                {
                    "task_id": task_id,
                    "id": task_id,
                    "question": q["question"],
                    "gt_sql": q["sql"],
                    "db_id": db_id,
                    "domain": domain,
                    # Metric names: "complex query" -> "complex_query".
                    "category": (q.get("category") or "").replace(" ", "_"),
                    "detailed_category": q.get("detailed_category") or "",
                    "evidence": evidence,
                    "evidence_section": f"\n\n### Evidence\n{evidence}" if evidence else "",
                    "sql_context": schemas[db_id],
                }
            )
    return rows


def prepare(
    domains: Sequence[str] = ("dw",),
    sample: int = 300,
    seed: int = SAMPLE_SEED,
    include_evidence: bool = True,
    cache_dir: Optional[str] = None,
) -> Path:
    """Write the prepared rows for ``domains`` and return the file's path.

    Args:
        domains: BEAVER query splits: ``dw`` (default), ``dw_real`` (runs on the dw database), ``nova``, ``neutron``.
        sample: Questions to sample per domain with ``seed`` (0 = all). Default 300, as auto-ontology-eval's runs.
        seed: Sampling seed; 77 is BEAVER's own.
        include_evidence: Show each question's domain knowledge (default). ``False`` matches the leaderboard's
            Setting 0.
        cache_dir: Download cache for ``beaver_db.zip``; defaults to the ``beaver_sql`` server's.
    """
    domains = list(domains)
    unknown = [d for d in domains if d not in DOMAINS]
    if unknown:
        raise ValueError(f"Unknown BEAVER domains {unknown}; expected some of {DOMAINS}")
    domain_rows = {d: sample_questions(load_questions(d), sample, seed) for d in domains}
    db_ids = sorted({DOMAIN_DB[d] for d in domains})
    dumps = ensure_beaver_dumps(Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR, db_ids)
    schemas = {db_id: schema_dump(path) for db_id, path in dumps.items()}
    rows = build_rows(domain_rows, schemas, include_evidence)
    out = output_fpath_for(domains, sample, seed, include_evidence)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} BEAVER rows to {out}")
    return out


if __name__ == "__main__":
    prepare()
