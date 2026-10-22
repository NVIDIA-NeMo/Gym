# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect K rollouts per BIRD task and write training data, for any ``--split`` / ``--dialect``.

Output records, one per task: ``{"id", "db_id", "context": [...], "completions": [{"completion": [...],
"reward": ..., "reference_logprob": ...}]}``. ``reference_logprob`` is the summed per-token logprob of the model's
generation (``None`` if the backend returned none).

``collect`` calls the policy endpoint (``env.yaml``'s ``policy_base_url`` / ``policy_api_key`` / ``policy_model_name``,
or ``--policy-*``) straight from this process with ``logprobs: true, top_logprobs: 1``. It does not use the
``vllm_model`` server, which hardcodes ``top_logprobs=0`` -- hosted inference APIs may not support that the way
self-hosted vLLM does. Prompts and rewards reuse the benchmark's own code: ``nemo_gym.prompt.apply_prompt_to_row`` and
the ``bird_sql`` server's ``extract_sql_from_response`` / ``eval_utils`` (SQLite, or Spark for ``--dialect spark``).

Every raw API response is persisted before any post-processing, so a later fix to the text reconstruction, reward or
logprob logic can be re-applied offline for free with ``reprocess``.

Subcommands (``python -m benchmarks.birdbench.collect <subcommand> --help``):

- ``collect``: prepare the data if missing, then collect rollouts. Writes ``raw_choices.jsonl`` and ``grouped.jsonl``
  into the output dir (default ``train_data/<split>_<dialect>/<model>/``).
- ``reprocess``: rebuild ``grouped.jsonl`` from ``raw_choices.jsonl``, with no API calls.
- ``group``: convert the rollouts of a ``gym eval run`` (e.g. from ``eval.sh``) into the same grouped format.
- ``select-hard``: keep the tasks that no collected rollout solved, for a follow-up ``collect --input-jsonl``.

Needs ``bm25s`` and ``nltk`` for ``prepare`` (see README.md). ``--dialect spark`` also needs a JDK and pyspark, so run
it with ``resources_servers/bird_sql/.venv``'s python.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp
import yaml
from tqdm.auto import tqdm

from benchmarks.birdbench import prepare as birdbench_prepare
from benchmarks.birdbench.prepare import DIALECTS, SPLITS, databases_dir_for, prompt_config_fpath_for
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config
from nemo_gym.responses_converter import ResponsesConverter
from resources_servers.bird_sql.app import extract_sql_from_response
from resources_servers.bird_sql.eval_utils import execute_and_compare, execute_and_compare_spark


logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).parent
REPO_ROOT = BENCHMARK_DIR.parent.parent
TRAIN_DATA_DIR = BENCHMARK_DIR / "train_data"

# Must match the `spark_warehouse_dir` that `eval.sh` gives the server for each split (train has its own db_ids).
SPARK_WAREHOUSE_DIRS = {
    "dev": REPO_ROOT / "resources_servers/bird_sql/.bird_sql/spark_warehouse",
    "train": REPO_ROOT / "resources_servers/bird_sql/.bird_sql/spark_warehouse_train",
}

_FLUSH_EVERY_N_ROWS = 250
_LOG_REWARD_EVERY_N_TASKS = 50
_MAX_REQUEST_RETRIES = 5
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# (db_id, gold_sql, generated_text) -> reward in {0.0, 1.0}
Scorer = Callable[[str, str, str], Awaitable[float]]


def make_scorer(dialect: str, databases_dir: Path, sql_concurrency: int, spark_warehouse_dir: Path) -> Scorer:
    """The same reward as ``BirdSqlResourcesServer.verify()``: no ```sql``` block, or any execution error, scores 0."""
    semaphore = asyncio.Semaphore(sql_concurrency)
    spark_sessions: Dict[str, Any] = {}
    if dialect == "spark":
        # Imported here: only the Spark dialect needs pyspark and a JVM.
        from resources_servers.bird_sql.setup_bird_spark import ensure_bird_spark

        spark_sessions = ensure_bird_spark(databases_dir, spark_warehouse_dir)

    async def score(db_id: str, gold_sql: str, generated_text: str) -> float:
        extracted_sql = extract_sql_from_response(generated_text)
        if extracted_sql is None:
            return 0.0
        db_path = databases_dir / db_id / f"{db_id}.sqlite"
        try:
            if dialect == "spark":
                match, _gold, _pred, err = await execute_and_compare_spark(
                    db_path, spark_sessions[db_id], gold_sql, extracted_sql, semaphore
                )
            else:
                match, _gold, _pred, err = await execute_and_compare(db_path, gold_sql, extracted_sql, semaphore)
        except Exception:
            logger.exception("Scoring failed for db_id=%s; counting it as reward 0", db_id)
            return 0.0
        return 1.0 if match and err is None else 0.0

    return score


# --------------------------------------------------------------------------------------
# Raw API choice -> text / logprob
# --------------------------------------------------------------------------------------


def _reconstruct_generated_text(choice: Dict[str, Any]) -> str:
    """The text the model generated, from a raw chat-completion ``choice``: ``reasoning</think>content`` plus the stop
    token, lining up token-for-token with ``choice["logprobs"]["content"]``.

    - No ``<think>`` opening tag: chat templates force it as a prefix the model continues from, so it is never
      generated and has no logprob (``_add_think_prefix_for_saving`` adds it back, for the saved text only).
    - No separators around ``</think>``: the generation stream has none.
    - ``</think>`` is only added if reasoning actually finished (non-empty ``content``, or ``finish_reason == "stop"``).
      A ``length`` cutoff mid-reasoning returns reasoning with empty ``content``; faking a ``</think>`` would present
      an unfinished generation as complete. This does not depend on the backend returning logprobs.
    - The stop token (e.g. ``<|im_end|>``, read from the last logprob entry) is appended only when
      ``finish_reason == "stop"``, so its absence marks a truncated generation.
    """
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    logprobs_content = (choice.get("logprobs") or {}).get("content") or []

    if reasoning:
        reasoning_finished = bool(content) or choice.get("finish_reason") == "stop"
        text = f"{reasoning}</think>{content}" if reasoning_finished else reasoning
    else:
        text = content

    if choice.get("finish_reason") == "stop" and logprobs_content:
        text += logprobs_content[-1].get("token", "")
    return text


def _add_think_prefix_for_saving(generated_text: str, choice: Dict[str, Any]) -> str:
    """Prefix ``<think>`` for the saved completion only -- never before SQL extraction or anything feeding
    ``reference_logprob`` (see ``_reconstruct_generated_text``)."""
    message = choice.get("message") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    return f"<think>{generated_text}" if reasoning else generated_text


def _sum_chat_completion_logprobs(choice: Dict[str, Any]) -> Optional[float]:
    logprobs_block = choice.get("logprobs")
    content = logprobs_block.get("content") if isinstance(logprobs_block, dict) else None
    if not isinstance(content, list) or not content:
        return None
    total = 0.0
    for entry in content:
        log_prob = entry.get("logprob")
        if not isinstance(log_prob, (int, float)):
            return None
        total += log_prob
    return total


async def _score_choice(choice: Dict[str, Any], db_id: str, gold_sql: str, scorer: Scorer) -> Dict[str, Any]:
    """One raw ``choice`` -> a scored completion. Pure post-processing (no network), which is what ``reprocess`` re-runs."""
    generated_text = _reconstruct_generated_text(choice)
    return {
        "completion": [{"role": "assistant", "content": _add_think_prefix_for_saving(generated_text, choice)}],
        "reward": await scorer(db_id, gold_sql, generated_text),
        "reference_logprob": _sum_chat_completion_logprobs(choice),
    }


# --------------------------------------------------------------------------------------
# Direct inference
# --------------------------------------------------------------------------------------


async def _call_policy_model(
    session: aiohttp.ClientSession,
    api_semaphore: asyncio.Semaphore,
    base_url: str,
    api_key: str,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[str],
    on_latency: Callable[[float], Awaitable[None]],
) -> Dict[str, Any]:
    """POST one chat completion and return ``choices[0]``.

    Retries 429/5xx, timeouts and connection errors with exponential backoff (a single blip should not sink an
    hours-long job); other errors, e.g. 400/401, raise at once. Latency is timed from acquiring ``api_semaphore``, so
    it excludes time spent queued.
    """
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 1,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = base_url.rstrip("/") + "/chat/completions"

    for attempt in range(1, _MAX_REQUEST_RETRIES + 1):
        is_last_attempt = attempt == _MAX_REQUEST_RETRIES
        try:
            async with api_semaphore:
                started_at = time.monotonic()
                async with session.post(
                    url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    if resp.ok:
                        data = await resp.json()
                        await on_latency(time.monotonic() - started_at)
                        return data["choices"][0]
                    body = (await resp.text())[:500]
                    if resp.status not in _RETRYABLE_STATUS_CODES or is_last_attempt:
                        logger.error("Policy model request failed with status %d: %s", resp.status, body)
                        resp.raise_for_status()
                    logger.warning(
                        "Policy model request failed (attempt %d/%d): %d %s. Retrying...",
                        attempt,
                        _MAX_REQUEST_RETRIES,
                        resp.status,
                        body,
                    )
        except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, asyncio.TimeoutError) as e:
            if is_last_attempt:
                raise
            logger.warning(
                "Policy model request failed (attempt %d/%d): %s. Retrying...", attempt, _MAX_REQUEST_RETRIES, e
            )
        await asyncio.sleep(min(2 ** (attempt - 1), 30))
    raise AssertionError("unreachable: the last attempt returns or raises")


def _load_env_yaml_defaults() -> Dict[str, Optional[str]]:
    env_yaml_fpath = REPO_ROOT / "env.yaml"
    data = yaml.safe_load(env_yaml_fpath.read_text()) or {} if env_yaml_fpath.exists() else {}
    return {key: data.get(key) for key in ("policy_base_url", "policy_api_key", "policy_model_name")}


class _FlushingWriter:
    """Line-oriented jsonl writer that fsyncs every ``_FLUSH_EVERY_N_ROWS`` rows, so a killed run leaves a readable file."""

    def __init__(self, fpath: Path):
        fpath.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(fpath, "w")
        self._lock = asyncio.Lock()
        self.count = 0

    async def write(self, record: Dict[str, Any]) -> None:
        async with self._lock:
            self._f.write(json.dumps(record) + "\n")
            self.count += 1
            if self.count % _FLUSH_EVERY_N_ROWS == 0:
                self.sync()

    def sync(self) -> None:
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        self.sync()
        self._f.close()


async def collect_direct(
    rows: List[Dict[str, Any]],
    prompt_config_fpath: Path,
    scorer: Scorer,
    grouped_output_fpath: Path,
    raw_choices_output_fpath: Path,
    num_repeats: int,
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[str],
    api_concurrency: int,
    base_url: str,
    api_key: str,
    model_name: str,
) -> None:
    prompt_cfg = load_prompt_config(str(prompt_config_fpath))
    api_semaphore = asyncio.Semaphore(api_concurrency)
    grouped_writer = _FlushingWriter(grouped_output_fpath)
    raw_writer = _FlushingWriter(raw_choices_output_fpath)
    pbar = tqdm(total=len(rows), desc="Collecting rollouts", unit="task")
    failed_completions = 0
    request_count = 0
    request_time_total = 0.0

    async def on_latency(elapsed: float) -> None:
        nonlocal request_count, request_time_total
        request_count += 1
        request_time_total += elapsed
        pbar.set_postfix_str(f"avg_response_s={request_time_total / request_count:.1f} (n={request_count})")

    async def collect_one(session: aiohttp.ClientSession, row: Dict[str, Any], context: List[Dict[str, str]]):
        """One repeat. Returns ``None`` instead of raising if it fails after retries: a single bad repeat should not
        take down the whole job."""
        try:
            choice = await _call_policy_model(
                session,
                api_semaphore,
                base_url,
                api_key,
                model_name,
                context,
                temperature,
                max_tokens,
                reasoning_effort,
                on_latency,
            )
        except Exception:
            logger.exception("Giving up on one repeat after retries exhausted; skipping it.")
            return None
        # Persisted before any post-processing: the generation you paid for is safe whatever happens below.
        await raw_writer.write(
            {"id": row.get("id"), "db_id": row["db_id"], "gt_sql": row["gt_sql"], "context": context, "choice": choice}
        )
        return await _score_choice(choice, row["db_id"], row["gt_sql"], scorer)

    async def process_row(session: aiohttp.ClientSession, row: Dict[str, Any]) -> None:
        nonlocal failed_completions
        context = apply_prompt_to_row(row, prompt_cfg)["responses_create_params"]["input"]
        results = await asyncio.gather(*[collect_one(session, row, context) for _ in range(num_repeats)])
        completions = [c for c in results if c is not None]
        failed_completions += len(results) - len(completions)
        # `id`/`db_id` let a record be matched back to its prepared row; `context` alone is ambiguous when two rows
        # share a byte-identical prompt but have different gold SQL.
        await grouped_writer.write(
            {"id": row.get("id"), "db_id": row["db_id"], "context": context, "completions": completions}
        )
        pbar.update(1)

    # A bounded pool of row workers, not one gather over every row x repeat: at this scale that would create
    # hundreds of thousands of coroutines contending for `api_semaphore` at once, make the latency metric measure queue
    # position, and hold every materialized prompt in memory.
    row_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
    for row in rows:
        row_queue.put_nowait(row)

    async def row_worker(session: aiohttp.ClientSession) -> None:
        while not row_queue.empty():
            await process_row(session, row_queue.get_nowait())

    try:
        connector = aiohttp.TCPConnector(limit=max(100, api_concurrency))
        async with aiohttp.ClientSession(connector=connector) as session:
            await asyncio.gather(*[row_worker(session) for _ in range(max(1, api_concurrency))])
    finally:
        grouped_writer.close()
        raw_writer.close()
        pbar.close()

    if failed_completions:
        print(
            f"WARNING: {failed_completions} of {len(rows) * num_repeats} requested completions failed even after "
            "retries and were dropped (see the warnings above); some tasks have fewer than num_repeats completions."
        )
    if request_count:
        print(
            f"Average policy-model response time: {request_time_total / request_count:.2f}s over {request_count} requests"
        )
    print(f"Wrote {grouped_writer.count} grouped records to {grouped_output_fpath}")
    print(f"Wrote {raw_writer.count} raw API responses to {raw_choices_output_fpath} (for `reprocess`)")


async def reprocess_raw_choices(raw_choices_fpath: Path, scorer: Scorer, grouped_output_fpath: Path) -> None:
    """Rebuild the grouped output from a saved ``raw_choices.jsonl`` offline: re-runs text reconstruction, scoring and
    logprob summing against the stored raw ``choice``s, with no API calls."""
    by_task: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    with open(raw_choices_fpath) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                by_task[record["id"]].append(record)

    writer = _FlushingWriter(grouped_output_fpath)
    recent_rewards: List[float] = []
    try:
        for i, task_id in enumerate(tqdm(sorted(by_task, key=str), desc="Reprocessing", unit="task"), start=1):
            records = by_task[task_id]
            completions = await asyncio.gather(
                *[_score_choice(r["choice"], r["db_id"], r["gt_sql"], scorer) for r in records]
            )
            await writer.write(
                {
                    "id": task_id,
                    "db_id": records[0]["db_id"],
                    "context": records[0]["context"],
                    "completions": completions,
                }
            )
            recent_rewards.extend(c["reward"] for c in completions)
            if i % _LOG_REWARD_EVERY_N_TASKS == 0:
                print(f"[{i} tasks] recent_avg_reward={sum(recent_rewards) / len(recent_rewards):.3f}")
                recent_rewards = []
    finally:
        writer.close()
    print(f"Reprocessed {len(by_task)} tasks -> {grouped_output_fpath} (no API calls made)")


# --------------------------------------------------------------------------------------
# Grouping a `gym eval run` rollouts file
# --------------------------------------------------------------------------------------


def _item_text(item: Dict[str, Any]) -> str:
    """One output item's share of the raw completion text. A reasoning item becomes ``<think>...</think>``, using
    ``ResponsesConverter._wrap_reasoning_in_think_tags`` -- the inverse of how ``vllm_model`` parses a raw completion
    into these items."""
    if item.get("type") == "reasoning":
        parts = item.get("content") or item.get("summary") or []
        reasoning_text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        return ResponsesConverter._wrap_reasoning_in_think_tags([reasoning_text]) if reasoning_text else ""
    if item.get("type") == "message" and item.get("role") == "assistant":
        return "".join(p.get("text", "") for p in (item.get("content") or []) if p.get("type") == "output_text")
    return ""


def group_gym_rollouts(rollouts_fpath: Path) -> List[Dict[str, Any]]:
    """Group the per-repeat rows of a ``gym eval run`` rollouts file by task into the grouped format.

    Each row carries its own ``responses_create_params.input`` (the context) and ``response`` / ``reward``. The full
    completion (reasoning plus answer) is rebuilt from the output items, because ``response["output_text"]`` is a
    computed property that is neither serialized nor includes reasoning. ``reference_logprob`` is only set when
    ``return_token_id_information`` reached the policy model, which makes the assistant item carry
    ``generation_log_probs``.
    """
    by_task: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    with open(rollouts_fpath) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                by_task[row["_ng_task_index"]].append(row)

    grouped = []
    for task_index in sorted(by_task):
        rows = sorted(by_task[task_index], key=lambda r: r["_ng_rollout_index"])
        completions = []
        for row in rows:
            output_items = row["response"].get("output") or []
            assistant_items = [item for item in output_items if item.get("role") == "assistant"]
            log_probs = assistant_items[-1].get("generation_log_probs") if assistant_items else None
            completions.append(
                {
                    "completion": [{"role": "assistant", "content": "".join(_item_text(i) for i in output_items)}],
                    "reward": row["reward"],
                    "reference_logprob": sum(log_probs) if log_probs is not None else None,
                }
            )
        grouped.append(
            {
                "id": rows[0].get("id"),
                "db_id": rows[0].get("db_id"),
                "context": rows[0]["responses_create_params"]["input"],
                "completions": completions,
            }
        )
    return grouped


# --------------------------------------------------------------------------------------
# Hard examples
# --------------------------------------------------------------------------------------


def select_hard_examples(
    prepared_jsonl: Path, grouped_jsonls: List[Path], output_jsonl: Path, reward_threshold: float, prompt_config: Path
) -> None:
    """Re-emit the prepared rows (same schema, ``sql_context`` included) that no collected rollout solved.

    A row is hard if, pooled over every grouped file, no completion reached ``reward_threshold``; rows that were never
    collected count as hard too. Grouped records are matched to rows by ``id``; records without one (from ``group``)
    fall back to matching the materialized prompt, which is ambiguous only if two rows have identical prompts.
    Feed the output to ``collect --input-jsonl``.
    """
    prompt_cfg = load_prompt_config(str(prompt_config))
    with open(prepared_jsonl) as f:
        prepared_rows = [json.loads(line) for line in f]
    print(f"Loaded {len(prepared_rows)} prepared rows from {prepared_jsonl}")

    def context_key(context: List[Dict[str, Any]]) -> str:
        return json.dumps(context, sort_keys=True)

    row_index_by_id = {row["id"]: i for i, row in enumerate(prepared_rows)}
    row_index_by_context = {
        context_key(apply_prompt_to_row(row, prompt_cfg)["responses_create_params"]["input"]): i
        for i, row in enumerate(prepared_rows)
    }

    best_reward_by_row: Dict[int, float] = {}
    total_records = unmatched = 0
    for grouped_jsonl in grouped_jsonls:
        with open(grouped_jsonl) as f:
            for line in f:
                record = json.loads(line)
                total_records += 1
                if record.get("id") is not None:
                    i = row_index_by_id.get(record["id"])
                else:
                    i = row_index_by_context.get(context_key(record["context"]))
                if i is None:
                    unmatched += 1
                    continue
                record_best = max((c.get("reward", 0.0) for c in record.get("completions") or []), default=-1.0)
                best_reward_by_row[i] = max(best_reward_by_row.get(i, -1.0), record_best)
    if unmatched:
        print(
            f"WARNING: {unmatched}/{total_records} grouped records matched no prepared row -- check that "
            "--prepared-jsonl and --grouped-jsonl come from the same collection run."
        )

    hard_rows = [row for i, row in enumerate(prepared_rows) if best_reward_by_row.get(i, -1.0) < reward_threshold]
    never_collected = len(prepared_rows) - len(best_reward_by_row)
    if never_collected:
        print(f"{never_collected} prepared rows had no matching grouped record at all -- counted as hard.")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w") as f_out:
        for row in hard_rows:
            f_out.write(json.dumps(row) + "\n")
    print(f"Wrote {len(hard_rows)} hard rows (no reward>={reward_threshold} rollout yet) to {output_jsonl}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--dialect", choices=DIALECTS, default="sqlite")
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Extracted BIRD train/ dir (train.json + train_databases/). Default: downloaded on first use.",
    )
    parser.add_argument(
        "--sql-concurrency",
        type=int,
        default=16,
        help="Concurrent SQL executions. Too many threads contending on SQLite's progress handler can starve the GIL "
        "(0/32 correct at 32 vs 31/32 at 16, same data); use ~4 for --dialect spark (4 local cores).",
    )


def _output_dir(args: argparse.Namespace, model_name: Optional[str] = None) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    safe_model = re.sub(r"[/: ]", "_", model_name or "model")
    return TRAIN_DATA_DIR / f"{args.split}_{args.dialect}" / safe_model


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Prepare data if missing, then collect rollouts from a policy endpoint")
    _add_data_args(collect)
    collect.add_argument("--num-repeats", type=int, default=8, help="Rollouts per task")
    collect.add_argument("--temperature", type=float, default=1.0)
    collect.add_argument("--max-tokens", type=int, default=16000)
    collect.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["low", "medium", "xhigh"],
        help="Sent as `reasoning_effort` if set, else omitted (backend default). Lower effort also shortens how long "
        "each request holds an --api-concurrency slot.",
    )
    collect.add_argument("--limit", type=int, default=None, help="Only the first N tasks (e.g. a smoke test)")
    collect.add_argument("--api-concurrency", type=int, default=64, help="Concurrent policy-model requests")
    collect.add_argument(
        "--input-jsonl", default=None, help="Collect from this prepared jsonl instead (e.g. hard rows)"
    )
    collect.add_argument("--output-dir", default=None)
    for name in ("base-url", "api-key", "model-name"):
        collect.add_argument(f"--policy-{name}", default=None, help="Default: env.yaml")

    reprocess = sub.add_parser("reprocess", help="Rebuild grouped.jsonl from raw_choices.jsonl, offline")
    _add_data_args(reprocess)
    reprocess.add_argument("--output-dir", required=True, help="The collect run's dir (holds raw_choices.jsonl)")
    reprocess.add_argument("--grouped-output", default=None, help="Default: <output-dir>/grouped_reprocessed.jsonl")

    group = sub.add_parser("group", help="Convert a `gym eval run` rollouts.jsonl to the grouped format")
    group.add_argument("--rollouts", required=True)
    group.add_argument("--output", required=True)

    hard = sub.add_parser("select-hard", help="Keep the prepared rows no collected rollout solved")
    hard.add_argument("--dialect", choices=DIALECTS, default="sqlite", help="Selects the prompt used to match records")
    hard.add_argument("--prepared-jsonl", required=True)
    hard.add_argument("--grouped-jsonl", nargs="+", required=True, help="Pooled: hard only if unsolved in all of them")
    hard.add_argument("--output-jsonl", required=True)
    hard.add_argument("--reward-threshold", type=float, default=1.0)
    return parser


def _run_collect(args: argparse.Namespace) -> None:
    env_defaults = _load_env_yaml_defaults()
    base_url = args.policy_base_url or env_defaults["policy_base_url"]
    api_key = args.policy_api_key or env_defaults["policy_api_key"]
    model_name = args.policy_model_name or env_defaults["policy_model_name"]
    missing = [
        name
        for name, value in (
            ("policy_base_url", base_url),
            ("policy_api_key", api_key),
            ("policy_model_name", model_name),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing {missing}: set them in env.yaml or pass --policy-base-url/-api-key/-model-name")

    input_jsonl = (
        Path(args.input_jsonl) if args.input_jsonl else birdbench_prepare.output_fpath_for(args.dialect, args.split)
    )
    if not input_jsonl.exists():
        if args.input_jsonl:
            raise FileNotFoundError(f"--input-jsonl not found: {input_jsonl}")
        birdbench_prepare.prepare(dialect=args.dialect, split=args.split, train_dir=args.train_dir)
    with open(input_jsonl) as f:
        rows = [json.loads(line) for line in f]
    if args.limit is not None:
        rows = rows[: args.limit]

    output_dir = _output_dir(args, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    async def run() -> None:
        scorer = make_scorer(
            args.dialect,
            databases_dir_for(args.split, args.train_dir),
            args.sql_concurrency,
            SPARK_WAREHOUSE_DIRS[args.split],
        )
        await collect_direct(
            rows=rows,
            prompt_config_fpath=prompt_config_fpath_for(args.dialect),
            scorer=scorer,
            grouped_output_fpath=output_dir / "grouped.jsonl",
            raw_choices_output_fpath=output_dir / "raw_choices.jsonl",
            num_repeats=args.num_repeats,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            api_concurrency=args.api_concurrency,
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
        )

    asyncio.run(run())


def _run_reprocess(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    grouped_output = Path(args.grouped_output) if args.grouped_output else output_dir / "grouped_reprocessed.jsonl"

    async def run() -> None:
        scorer = make_scorer(
            args.dialect,
            databases_dir_for(args.split, args.train_dir),
            args.sql_concurrency,
            SPARK_WAREHOUSE_DIRS[args.split],
        )
        await reprocess_raw_choices(output_dir / "raw_choices.jsonl", scorer, grouped_output)

    asyncio.run(run())


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_argparser().parse_args()
    if args.command == "collect":
        _run_collect(args)
    elif args.command == "reprocess":
        _run_reprocess(args)
    elif args.command == "group":
        grouped = group_gym_rollouts(Path(args.rollouts))
        with open(args.output, "w") as f_out:
            for record in grouped:
                f_out.write(json.dumps(record) + "\n")
        print(f"Wrote {len(grouped)} grouped records to {args.output}")
    else:
        select_hard_examples(
            Path(args.prepared_jsonl),
            [Path(p) for p in args.grouped_jsonl],
            Path(args.output_jsonl),
            args.reward_threshold,
            prompt_config_fpath_for(args.dialect),
        )


if __name__ == "__main__":
    main()
    # Exit immediately instead of letting interpreter shutdown join every `asyncio.to_thread` worker, including ones
    # abandoned by a SQL timeout (the timeout only stops us waiting; a runaway query on a multi-GB train table keeps
    # running in its thread). All outputs have already been flushed and fsynced by now.
    os._exit(0)
