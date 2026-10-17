# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Write every committed example artifact under ``data/`` deterministically — no model calls.

``example.jsonl`` holds five synthetic rows in the exact shape the preparer emits. The
benchmark's tasks are third-party theorems from 23 Lean repositories and are not
redistributed here. The five theorems below are written for this file: elementary facts
about ``Nat``, ``Bool`` and ``List`` that need no imports, each placed at the root of a
repository the Lean image contains so that a smoke run against them compiles inside
that repository's own toolchain. Prompts are built with upstream's ``PromptBuilder``
from the pinned checkout, exactly as for the real corpus, and the rows are written in
the collated form the tracked file carries (``task_source`` instead of ``agent_ref``).

The other four artifacts reproduce what one smoke run of ``gym eval run`` over these
five rows recorded, rebuilt from the prompts plus ``EXAMPLE_ROLLOUTS`` — the verified
proof, its hand-written reasoning, the Lean wall time and the token counts of each
task. Every service identifier, provider route and timestamp is a fixed placeholder
(``resp-verisoftbench-example-<id>``, ``example-model``, ``created_at`` 0.0), so the
committed files carry nothing that names the collecting service and regenerate
byte-identically; ``tests/test_contracts.py`` asserts both. Aggregate metrics go
through the server's own ``compute_metrics``/``get_key_metrics`` (in process, no
servers), so the file shows the benchmark's ``any_of_k/*`` headline.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import orjson
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import (  # noqa: E402
    VeriSoftBenchResourcesServer,
    VeriSoftBenchResourcesServerConfig,
    VeriSoftBenchVerifyResponse,
)
from prepare_verisoftbench import AGENT_REF, DEFAULT_UPSTREAM_DIR, PromptSource, prepare  # noqa: E402
from upstream import SYNTHETIC_EXAMPLE_REL_PATH, ensure_checkout  # noqa: E402

from nemo_gym.config_types import DatasetConfig  # noqa: E402
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming  # noqa: E402
from nemo_gym.reward_profile import compute_aggregate_metrics  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402
from nemo_gym.train_data_utils import (  # noqa: E402
    DatasetMetrics,
    aggregate_other_metrics,
    compute_sample_metrics,
    postprocess_other_metrics,
)


SERVER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = SERVER_DIR / "data"
AGENT_NAME = AGENT_REF["name"]

EXAMPLES = [
    ("IntroEffects", "add_zero_right", "(n : Nat) : n + 0 = n", ":= by\n  simp", "Semantics"),
    (
        "LeanExprEvaluator",
        "bool_and_comm",
        "(a b : Bool) : (a && b) = (b && a)",
        ":= by\n  cases a <;> cases b <;> rfl",
        "Compiler",
    ),
    (
        "Lentil",
        "length_append",
        "(xs ys : List Nat) : (xs ++ ys).length = xs.length + ys.length",
        ":= by\n  simp",
        "Framework",
    ),
    ("lean-hoare", "two_mul_eq_add", "(n : Nat) : 2 * n = n + n", ":= by\n  omega", "Type systems"),
    (
        "LeroyCompilerVerificationCourse",
        "reverse_reverse",
        "(xs : List Nat) : xs.reverse.reverse = xs",
        ":= by\n  simp",
        "Applied verif.",
    ),
]

# Sampling knobs of the recorded smoke run (`gym eval run --temperature 0.7
# --max-output-tokens 8192`); they appear in the materialized inputs and in every
# rollout's request parameters.
RUN_TEMPERATURE = 0.7
RUN_MAX_OUTPUT_TOKENS = 8192

# Fixed placeholders for every service field of the recorded responses. The committed
# example files must never carry a provider's response identifiers, model route or
# clock; these stand in for all of them and keep regeneration byte-stable.
EXAMPLE_MODEL = "example-model"
EXAMPLE_CREATED_AT = 0.0

# One verified round-0 attempt per task, in the completion order the smoke run recorded
# them: the proof and its reasoning (the model turn is rebuilt from these two below),
# the Lean verification wall time and the request/response token counts. None of the
# five proofs needed auxiliary lemmas or a repair round.
EXAMPLE_ROLLOUTS = [
    {
        "task_id": 100005,
        "proof": "\nby simp\n",
        "reasoning": "This is a straightforward theorem about list reversal. The statement says that reversing a "
        "list twice gives back the original list. This is a well-known property of list reversal and should be "
        "provable using the standard library lemma `List.reverse_reverse`.\n\nLooking at the theorem, we need to "
        "prove that `xs.reverse.reverse = xs` for any list of natural numbers. The standard library should have "
        "`List.reverse_reverse` which states exactly this for any type.\n\nThe proof should be a simple application "
        "of `List.reverse_reverse` or we can use `simp` which should know about this property.",
        "lean_time_s": 16.19400000013411,
        "input_tokens": 2962,
        "output_tokens": 162,
    },
    {
        "task_id": 100002,
        "proof": "\nby\n  cases a <;> cases b <;> rfl\n",
        "reasoning": "This is a simple theorem about boolean commutativity. We need to prove that `a && b = b && a` "
        "for all booleans `a` and `b`.\n\nThe simplest approach is to case split on both `a` and `b` and verify each "
        "case:\n- When `a = true, b = true`: `true && true = true && true` ✓\n- When `a = true, b = false`: "
        "`true && false = false = false && true` ✓\n- When `a = false, b = true`: `false && true = false = "
        "true && false` ✓\n- When `a = false, b = false`: `false && false = false && false` ✓\n\nWe can use `decide` "
        "or `cases` tactics, or simply use `Bool.and_comm` if it exists in the library.",
        "lean_time_s": 20.605999998748302,
        "input_tokens": 2967,
        "output_tokens": 232,
    },
    {
        "task_id": 100001,
        "proof": "\nby rfl\n",
        "reasoning": "This is a basic theorem about natural number addition. We need to prove that `n + 0 = n` for "
        "any natural number `n`.\n\nIn Lean 4, this is actually definitionally true because addition on natural "
        "numbers is defined by recursion on the second argument:\n- `n + 0 = n` (base case)\n- `n + succ m = "
        "succ (n + m)` (recursive case)\n\nSo `n + 0` reduces to `n` by the definition of addition. This means we "
        "can prove this by `rfl` (reflexivity).\n\nAlternatively, we can use `Nat.add_zero` which is a standard "
        "library lemma that states exactly this.",
        "lean_time_s": 19.843000000342727,
        "input_tokens": 2963,
        "output_tokens": 185,
    },
    {
        "task_id": 100003,
        "proof": "\nby simp [List.length_append]\n",
        "reasoning": "This is a basic theorem about the length of appended lists. The theorem states that the length "
        "of the concatenation of two lists equals the sum of their individual lengths.\n\nThis is a standard "
        "property that should be provable by induction on the first list `xs`, or by using existing lemmas from the "
        "standard library.\n\nIn Lean 4's standard library, there should be a lemma `List.length_append` that states "
        "exactly this property. Let me try to use that directly, or if needed, prove it by induction.\n\nThe proof "
        "by induction would be:\n- Base case: `[] ++ ys = ys`, so `([] ++ ys).length = ys.length = 0 + ys.length`\n"
        "- Inductive case: `(x :: xs) ++ ys = x :: (xs ++ ys)`, so the length is `1 + (xs ++ ys).length = 1 + "
        "xs.length + ys.length = (x :: xs).length + ys.length`\n\nIn Lean 4, `List.length_append` is the standard "
        "lemma for this. Let me try using `simp` which should apply the relevant lemmas automatically.",
        "lean_time_s": 22.75,
        "input_tokens": 2974,
        "output_tokens": 307,
    },
    {
        "task_id": 100004,
        "proof": "\nby omega\n",
        "reasoning": "The theorem states that `2 * n = n + n` for natural numbers. This is a basic arithmetic fact. "
        "In Lean 4's standard library, multiplication by 2 is defined such that `2 * n = n + n` essentially by "
        "definition or can be easily derived.\n\nLet me think about how to prove this:\n- `2 * n` can be expanded as "
        "`(1 + 1) * n = 1 * n + 1 * n = n + n`\n- Or we can use the fact that `2 * n = n + n` directly via "
        "`Nat.two_mul` or similar lemmas\n\nThe simplest approach would be to use `omega` which handles linear "
        "arithmetic, or `ring` for ring operations, or simply `simp` with the right lemmas.\n\nActually, for "
        "natural numbers, `Nat.two_mul` states exactly this: `2 * n = n + n`. So we can use `exact Nat.two_mul n` "
        "or `simp [Nat.two_mul]`.\n\nLet me try `omega` as it should easily handle this linear arithmetic identity.",
        "lean_time_s": 24.277000000700355,
        "input_tokens": 2967,
        "output_tokens": 282,
    },
]


def example_rows() -> list[dict]:
    rows = []
    for i, (repo, name, signature, proof, category) in enumerate(EXAMPLES, start=1):
        thm_name = f"GymExample.{name}"
        stmt = f"theorem {thm_name} {signature}"
        rows.append(
            {
                "id": 100_000 + i,
                "thm_name": thm_name,
                "thm_stmt": stmt,
                "lean_root": repo,
                # The harness's named out-of-corpus exception: no repository contains
                # this file, so these rows are deliberately absent from the repo index.
                "rel_path": SYNTHETIC_EXAMPLE_REL_PATH,
                "imports": [],
                "used_lib_defs": [],
                "used_repo_defs": [],
                "lib_lemmas": [],
                "repo_lemmas": [],
                "used_local_defs": [],
                "used_local_lemmas": [],
                "local_ctx": "",
                "target_theorem": stmt + " :=",
                "ground_truth_proof": proof,
                "nesting_depth": 1,
                "transitive_dep_count": 0,
                "subset_aristotle": False,
                "category": category,
            }
        )
    return rows


def collate_rows(prepared: list[dict]) -> list[dict]:
    """The transformation ``gym dataset collate`` applies: the routing stamp becomes ``task_source``."""
    collated = []
    for row in prepared:
        row = dict(row)
        row.pop("agent_ref", None)
        row["task_source"] = AGENT_NAME
        collated.append(row)
    return collated


def example_jsonl_bytes(collated: list[dict]) -> bytes:
    # Collation serializes with the stdlib encoder (ASCII escapes), unlike the orjson lines below.
    return "".join(json.dumps(row) + "\n" for row in collated).encode("utf-8")


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(orjson.dumps(row) + b"\n" for row in rows)


def materialized_input_rows(collated: list[dict]) -> list[dict]:
    """The rows exactly as the recorded run materialized them before dispatch."""
    rows = []
    for task_index, row in enumerate(collated):
        row = dict(row)
        row["responses_create_params"] = dict(row["responses_create_params"]) | {
            "temperature": RUN_TEMPERATURE,
            "max_output_tokens": RUN_MAX_OUTPUT_TOKENS,
        }
        row["agent_ref"] = {"name": AGENT_NAME}
        row["_ng_task_index"] = task_index
        row["_ng_rollout_index"] = 0
        rows.append(row)
    return rows


def _response(task_id: int, generation: str, input_tokens: int, output_tokens: int) -> dict[str, Any]:
    """The recorded model response with every service field a fixed placeholder.

    The non-placeholder constants (``temperature`` 0.0, ``parallel_tool_calls`` false,
    the empty details fields) are what the recorded responses carried; they are part of
    the byte-stable shape, not choices made here.
    """
    return NeMoGymResponse.model_validate(
        {
            "id": f"resp-verisoftbench-example-{task_id}",
            "created_at": EXAMPLE_CREATED_AT,
            "model": EXAMPLE_MODEL,
            "object": "response",
            "output": [
                {
                    "id": f"msg-verisoftbench-example-{task_id}",
                    "content": [{"annotations": [], "text": generation, "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "metadata": {},
            "parallel_tool_calls": False,
            "temperature": 0.0,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "text": {"format": None, "verbosity": None},
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": 0, "audio_tokens": None, "text_tokens": input_tokens},
                "output_tokens": output_tokens,
                "output_tokens_details": {"reasoning_tokens": 0, "text_tokens": output_tokens},
                "total_tokens": input_tokens + output_tokens,
                "cost": None,
            },
            "store": None,
            "provider_specific_fields": {"citations": None, "thinking_blocks": None},
        }
    ).model_dump(mode="json")


def rollout_rows(collated: list[dict]) -> list[dict]:
    """The agent's verify responses as ``gym eval run`` saved them, one per task.

    Each row is rebuilt through the same models that produced it: the request through
    ``NeMoGymResponseCreateParamsNonStreaming`` (defaults filled exactly as dispatch
    fills them), the verdict through the server's ``VeriSoftBenchVerifyResponse``, and
    the agent's summary fields spliced in where its ``extra="allow"`` response model
    placed them. All five synthetic proofs verified on round 0.
    """
    task_index_by_id = {row["verifier_metadata"]["id"]: i for i, row in enumerate(collated)}
    rows = []
    for rollout in EXAMPLE_ROLLOUTS:
        task_id = rollout["task_id"]
        example_row = collated[task_index_by_id[task_id]]
        metadata = example_row["verifier_metadata"]
        generation = (
            f"<reasoning>\n{rollout['reasoning']}\n</reasoning>\n\n<lean4_proof>{rollout['proof']}</lean4_proof>"
        )
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            {
                "input": [
                    {"role": m["role"], "content": m["content"]}
                    for m in example_row["responses_create_params"]["input"]
                ],
                "temperature": RUN_TEMPERATURE,
                "max_output_tokens": RUN_MAX_OUTPUT_TOKENS,
            }
        ).model_dump(mode="json")
        response = _response(task_id, generation, rollout["input_tokens"], rollout["output_tokens"])
        verdict = VeriSoftBenchVerifyResponse.model_validate(
            {
                "responses_create_params": params,
                "response": response,
                "reward": 1.0,
                "status": "verified",
                "repair_round": 0,
                "verified": True,
                "harness_failure": 0.0,
                "task_id": task_id,
                "lean_root": metadata["lean_root"],
                "category": metadata["category"],
                "proof": rollout["proof"],
                "aux_lemmas": "",
                "reasoning": rollout["reasoning"],
                "lean_time_s": rollout["lean_time_s"],
            }
        ).model_dump(mode="json")
        attempt = {
            "repair_round": 0,
            "input": params,
            "response": response,
            "generation": generation,
            "status": "verified",
            "reward": 1.0,
            "proof": rollout["proof"],
            "aux_lemmas": "",
            "err_msg": None,
            "lean_time_s": rollout["lean_time_s"],
        }
        # The agent validates the server's verdict into its extra="allow" response model
        # and sets its four summary fields, so the saved row carries the base verify
        # fields first, the summary next, and the server's verdict fields after.
        row: dict[str, Any] = {
            key: verdict.pop(key)
            for key in (
                "responses_create_params",
                "response",
                "reward",
                "mask_sample",
                "failure_kind",
                "failure_reason",
            )
        }
        row.update({"solved_without_repair": 1.0, "repair_rounds_used": 0, "total_rounds": 1, "attempts": [attempt]})
        row.update(verdict)
        row.update(
            {
                "_ng_task_index": task_index_by_id[task_id],
                "_ng_rollout_index": 0,
                "agent_ref": {"name": AGENT_NAME},
                "task_source": AGENT_NAME,
            }
        )
        rows.append(row)
    return rows


def _metrics_server() -> VeriSoftBenchResourcesServer:
    """The server for its metrics methods only: no upstream checkout, no Lean container."""
    config = VeriSoftBenchResourcesServerConfig(
        host="0.0.0.0",
        port=0,
        entrypoint="",
        name="verisoftbench",
        # Points nowhere on purpose; aggregation never touches the harness, and a missing
        # one is a recorded status, not an error worth reporting here.
        upstream_dir="/nonexistent",
        require_container_at_startup=False,
    )
    app_logger = logging.getLogger("app")
    old_level = app_logger.level
    app_logger.setLevel(logging.CRITICAL)
    try:
        return VeriSoftBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    finally:
        app_logger.setLevel(old_level)


def aggregate_metrics_bytes(rollouts: list[dict]) -> bytes:
    """Aggregate the rollouts through the server's own metrics path, as `gym eval run` does.

    The heavyweight fields are stripped exactly as rollout collection strips them before
    calling ``/aggregate_metrics`` (``response.usage`` survives for the token means), so
    the file carries the benchmark's ``any_of_k/*`` headline over the five tasks.
    """
    stripped = []
    for row in rollouts:
        entry = {k: v for k, v in row.items() if k not in ("response", "responses_create_params")}
        usage = (row.get("response") or {}).get("usage")
        if usage is not None:
            entry["response"] = {"usage": usage}
        stripped.append(entry)
    server = _metrics_server()
    metrics = compute_aggregate_metrics(
        stripped, compute_metrics_fn=server.compute_metrics, get_key_metrics_fn=server.get_key_metrics
    )
    agent_entry = {
        "agent_ref": {"name": AGENT_NAME},
        "agent_metrics": metrics.agent_metrics,
        "key_metrics": metrics.key_metrics,
        "group_level_metrics": metrics.group_level_metrics,
        "repeat_level_metrics": metrics.repeat_level_metrics,
    }
    return orjson.dumps([agent_entry], option=orjson.OPT_INDENT_2)


def example_metrics_bytes(example_jsonl: bytes) -> bytes:
    """The sidecar the ``example_validation`` collate gate computes over ``example.jsonl``."""
    dataset_entries = yaml.safe_load((SERVER_DIR / "configs" / "verisoftbench.yaml").read_text())[AGENT_NAME][
        "responses_api_agents"
    ][AGENT_NAME]["datasets"]
    [dataset_config] = [DatasetConfig.model_validate(d) for d in dataset_entries if d["type"] == "example"]
    metrics = DatasetMetrics()
    other_metrics: dict[str, Any] = {}
    for line in example_jsonl.decode("utf-8").splitlines():
        sample_metrics, offending = compute_sample_metrics(line)
        if offending:
            raise ValueError("example row does not validate as a run request")
        metrics.add(sample_metrics)
        aggregate_other_metrics(other_metrics, json.loads(line))
    postprocess_other_metrics(metrics, other_metrics)
    aggregate = metrics.aggregate().model_dump(mode="json", by_alias=True)
    return json.dumps(dataset_config.model_dump(mode="json", exclude={"agent"}) | aggregate, indent=4).encode("utf-8")


def main() -> None:
    upstream_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_UPSTREAM_DIR
    ensure_checkout(upstream_dir)
    collated = collate_rows(prepare(example_rows(), PromptSource(upstream_dir), limit=None))
    rollouts = rollout_rows(collated)
    example_jsonl = example_jsonl_bytes(collated)
    artifacts = {
        "example.jsonl": example_jsonl,
        "example_metrics.json": example_metrics_bytes(example_jsonl),
        "example_rollouts.jsonl": jsonl_bytes(rollouts),
        "example_rollouts_materialized_inputs.jsonl": jsonl_bytes(materialized_input_rows(collated)),
        "example_rollouts_aggregate_metrics.json": aggregate_metrics_bytes(rollouts),
    }
    for name, payload in artifacts.items():
        (DATA_DIR / name).write_bytes(payload)
        print(f"Wrote {DATA_DIR / name} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()
