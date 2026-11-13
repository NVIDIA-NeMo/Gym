# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resumable, one-prefill-per-question GPQA likelihood evaluation using vLLM."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import time
from pathlib import Path


def audit_likelihood_parity(optimized: list[list[float]], forced: list[list[float]]) -> dict:
    """Compare numerical likelihoods without demanding identical argmax near ties."""
    assert len(optimized) == len(forced) and optimized
    differences = []
    near_ties = []
    for i, (left, right) in enumerate(zip(optimized, forced, strict=True)):
        assert len(left) == len(right) == 4
        assert all(math.isfinite(x) for x in left + right)
        delta = max(abs(a - b) for a, b in zip(left, right, strict=True))
        assert delta < 0.08, ("likelihood parity failed", i, left, right, delta)
        differences.extend(abs(a - b) for a, b in zip(left, right, strict=True))
        a = max(range(4), key=lambda j: left[j])
        b = max(range(4), key=lambda j: right[j])
        if a != b:
            # A change of argmax is numerically ambiguous only within the
            # measured error bound between these two BF16 kernel shapes.
            assert left[a] - left[b] <= 2 * delta + 1e-7
            assert right[b] - right[a] <= 2 * delta + 1e-7
            near_ties.append(
                dict(
                    question=i,
                    optimized=left,
                    forced=right,
                    optimized_margin=left[a] - left[b],
                    forced_margin=right[b] - right[a],
                    max_logprob_difference=delta,
                )
            )
    return dict(
        passed=True,
        forced_likelihood_comparisons=len(differences),
        max_logprob_difference=max(differences),
        numerical_near_ties=near_ties,
    )


def read_results(path: Path, expected: dict, identity: str) -> dict:
    """Ignore only an interrupted final JSONL record; never accept stale scores."""
    rows = {}
    if not path.exists():
        return rows
    with path.open("r+b") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                handle.truncate(start)
                break
            row = json.loads(line)
            assert row["identity"] == identity and row["id"] in expected and row["id"] not in rows
            assert len(row["choice_logprobs"]) == 4 and all(math.isfinite(s) for s in row["choice_logprobs"])
            selected = max(range(4), key=lambda i: row["choice_logprobs"][i])
            assert row["prediction"] == "ABCD"[selected]
            assert row["correct"] == (row["prediction"] == expected[row["id"]]["answer"])
            rows[row["id"]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("model")
    parser.add_argument("language")
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    model = manifest["models"][args.model]
    data_path = run / "data" / f"{args.language}.jsonl"
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == manifest["languages"][args.language]["data_sha256"]
    for filename, field in [
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "index_sha256"),
        ("tokenizer.json", "tokenizer_sha256"),
    ]:
        assert hashlib.sha256((Path(model["path"]) / filename).read_bytes()).hexdigest() == model[field]
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    expected = {r["id"]: r for r in rows}
    assert len(expected) == len(rows) == manifest["languages"][args.language]["rows"]
    work = run / "results" / args.model / args.language
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = hashlib.sha256(
        (run / "manifest.json").read_bytes() + args.model.encode() + args.language.encode()
    ).hexdigest()
    output_path = work / "scores.jsonl"
    completed = read_results(output_path, expected, identity)
    todo = [r for r in rows if r["id"] not in completed]
    status = dict(
        model=args.model,
        language=args.language,
        job_id=os.environ.get("SLURM_JOB_ID"),
        state="starting",
        completed=len(completed),
        total=len(rows),
        started=time.time(),
        scorer_path=str(Path(__file__).resolve()),
        scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )

    def save_status() -> None:
        # Lustre can transiently return ENOENT during metadata failover even
        # when the directory existed moments earlier. Keep status writes from
        # aborting an otherwise valid scoring batch, and publish atomically so
        # readers never observe a partial JSON document.
        temporary = work / f".status-{os.environ.get('SLURM_JOB_ID', os.getpid())}.tmp"
        error = None
        for attempt in range(10):
            try:
                work.mkdir(parents=True, exist_ok=True)
                temporary.write_text(json.dumps(status, indent=2) + "\n")
                temporary.replace(work / "status.json")
                return
            except FileNotFoundError as caught:
                error = caught
                time.sleep(min(0.25 * 2**attempt, 5.0))
        raise error

    save_status()
    try:
        if todo:
            cache = Path("/tmp") / f"gpqa-pre-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
            cache.mkdir(exist_ok=True)
            os.environ.update(
                HF_HUB_OFFLINE="1",
                VLLM_NO_USAGE_STATS="1",
                OMP_NUM_THREADS="1",
                NCCL_RAS_ENABLE="0",
                TOKENIZERS_PARALLELISM="false",
            )
            for key, suffix in {
                "TMPDIR": "tmp",
                "VLLM_CACHE_ROOT": "vllm",
                "TORCHINDUCTOR_CACHE_DIR": "torch",
                "TRITON_CACHE_DIR": "triton",
                "HF_MODULES_CACHE": "modules",
            }.items():
                path = cache / suffix
                path.mkdir(exist_ok=True)
                os.environ[key] = str(path)
            from vllm import LLM, SamplingParams

            llm = LLM(
                model=model["path"],
                tensor_parallel_size=model["tensor_parallel_size"],
                distributed_executor_backend="mp",
                dtype="bfloat16",
                trust_remote_code=True,
                max_model_len=manifest["max_model_len"],
                gpu_memory_utilization=0.85,
                max_num_seqs=16,
                max_num_batched_tokens=4096,
                enable_chunked_prefill=True,
                enable_prefix_caching=False,
                enforce_eager=True,
                disable_custom_all_reduce=True,
                mamba_ssm_cache_dtype="float32",
                generation_config="vllm",
                seed=42,
            )
            tokenizer = llm.get_tokenizer()
            for row in rows:
                ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
                assert ids == row["prompt_token_ids"]
                for letter, token in zip("ABCD", row["choice_token_ids"], strict=True):
                    assert tokenizer.encode(row["prompt"] + " " + letter, add_special_tokens=False) == ids + [token]

            def score(batch: list[dict]) -> list[list[float]]:
                if scoring_method == "forced_continuation":
                    return forced_score(batch)
                # Unit temperature and no filtering preserve the raw distribution
                # in the installed vLLM specific-token-logprobs implementation.
                params = [
                    SamplingParams(
                        temperature=1.0,
                        top_p=1.0,
                        top_k=-1,
                        max_tokens=1,
                        logprob_token_ids=row["choice_token_ids"],
                        seed=42,
                        ignore_eos=True,
                        detokenize=False,
                    )
                    for row in batch
                ]
                predictions = llm.generate(
                    [{"prompt_token_ids": row["prompt_token_ids"]} for row in batch], params, use_tqdm=False
                )
                result = []
                for row, prediction in zip(batch, predictions, strict=True):
                    values = prediction.outputs[0].logprobs[0]
                    scores = [values[token].logprob for token in row["choice_token_ids"]]
                    assert all(math.isfinite(x) and x <= 0 for x in scores)
                    result.append(scores)
                return result

            def forced_score(batch: list[dict]) -> list[list[float]]:
                prompts = [
                    {"prompt_token_ids": row["prompt_token_ids"] + [token]}
                    for row in batch
                    for token in row["choice_token_ids"]
                ]
                predictions = llm.generate(
                    prompts,
                    SamplingParams(
                        temperature=1.0,
                        top_p=1.0,
                        top_k=-1,
                        max_tokens=1,
                        prompt_logprobs=0,
                        seed=42,
                        ignore_eos=True,
                        detokenize=False,
                    ),
                    use_tqdm=False,
                )
                values = []
                for i, prediction in enumerate(predictions):
                    row = batch[i // 4]
                    token = row["choice_token_ids"][i % 4]
                    assert prediction.prompt_token_ids == row["prompt_token_ids"] + [token]
                    value = prediction.prompt_logprobs[-1][token].logprob
                    assert math.isfinite(value) and value <= 0
                    if i % 4 == 0:
                        values.append([])
                    values[-1].append(value)
                return values

            method_path = work / "scoring_method.json"
            scoring_method = json.loads(method_path.read_text())["method"] if method_path.exists() else "one_pass"
            status["state"] = "smoke"
            save_status()
            smoke_rows = todo[:5]
            smoke_scores = score(smoke_rows)
            # Independently verify optimized next-token scoring against forced
            # continuation likelihoods, including the actual token boundary.
            forced_prompts = [
                {"prompt_token_ids": row["prompt_token_ids"] + [token]}
                for row in smoke_rows[:2]
                for token in row["choice_token_ids"]
            ]
            forced = llm.generate(
                forced_prompts,
                SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=1,
                    prompt_logprobs=0,
                    seed=42,
                    ignore_eos=True,
                    detokenize=False,
                ),
                use_tqdm=False,
            )
            forced_scores = []
            for i, result in enumerate(forced):
                token = smoke_rows[i // 4]["choice_token_ids"][i % 4]
                value = result.prompt_logprobs[-1][token].logprob
                if i % 4 == 0:
                    forced_scores.append([])
                forced_scores[-1].append(value)
            (work / "smoke_scores.json").write_text(
                json.dumps(dict(optimized=smoke_scores[: len(forced_scores)], forced=forced_scores), indent=2) + "\n"
            )
            try:
                smoke_report = audit_likelihood_parity(smoke_scores[: len(forced_scores)], forced_scores)
                fallback_reason = "BF16 argmax near tie" if smoke_report["numerical_near_ties"] else None
            except AssertionError as error:
                smoke_report = dict(optimized_parity_error=str(error))
                fallback_reason = "Optimized likelihoods exceeded reference tolerance"
            if fallback_reason and scoring_method == "one_pass":
                assert not completed, "Cannot change scoring backend after partial results have been saved"
                scoring_method = "forced_continuation"
                smoke_scores = forced_score(smoke_rows)
            smoke_report.update(passed=True, scoring_method=scoring_method, fallback_reason=fallback_reason)
            method_path.write_text(json.dumps({"method": scoring_method}) + "\n")
            smoke_report["rows"] = len(smoke_rows)
            (work / "smoke.json").write_text(json.dumps(smoke_report, indent=2) + "\n")
            batches = [(smoke_rows, smoke_scores)]
            status["state"] = "evaluating"
            save_status()
            with output_path.open("a") as handle:

                def persist(batch: list[dict], scores: list[list[float]]) -> None:
                    for row, values in zip(batch, scores, strict=True):
                        prediction = "ABCD"[max(range(4), key=lambda i: values[i])]
                        result = dict(
                            id=row["id"],
                            identity=identity,
                            choice_logprobs=values,
                            scoring_method=scoring_method,
                            prediction=prediction,
                            answer=row["answer"],
                            correct=prediction == row["answer"],
                        )
                        handle.write(json.dumps(result) + "\n")
                        completed[row["id"]] = result
                    handle.flush()
                    os.fsync(handle.fileno())
                    status["completed"] = len(completed)
                    save_status()
                    print(f"{args.model}/{args.language}: {len(completed)}/{len(rows)}", flush=True)

                persist(*batches[0])
                for start in range(len(smoke_rows), len(todo), 16):
                    batch = todo[start : start + 16]
                    persist(batch, score(batch))
        audited = read_results(output_path, expected, identity)
        assert len(audited) == len(rows)
        correct = sum(x["correct"] for x in audited.values())
        report = dict(
            model=args.model,
            language=args.language,
            num_fewshot=5,
            rows=len(rows),
            correct=correct,
            accuracy=correct / len(rows),
            accuracy_pct=100 * correct / len(rows),
            identity=identity,
        )
        (work / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        (work / "DONE").touch()
        status["state"] = "completed"
        print(json.dumps(report), flush=True)
    except BaseException as error:
        status.update(state="failed", error=repr(error))
        raise
    finally:
        status["updated"] = time.time()
        save_status()


if __name__ == "__main__":
    main()
