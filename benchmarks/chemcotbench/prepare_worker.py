# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build upstream ChemCoT prompts and source properties in the isolated chemistry runtime."""

import argparse
import json
import math
import os
import re
import sys
import warnings
from collections.abc import Callable
from contextlib import redirect_stdout
from functools import lru_cache
from importlib import import_module
from typing import Any, Protocol


SUBTASKS = {
    "mol_edit": ("add_v2", "delete_v2", "substitute_v2"),
    "rxn_pred": (
        "forward",
        "byproduct",
        "nepp",
        "retro",
        "rcr_catalyst",
        "rcr_reagent",
        "rcr_solvent",
        "condition_ranking",
        "yield_pred",
    ),
    "mol_und": ("fg_detect", "ring_count", "murcko_scaffold", "ring_sys_scaffold", "smiles_equivalent"),
    "mol_opt": (
        "logp",
        "qed",
        "solubility",
        "drd",
        "gsk",
        "jnk",
        "logp_qed",
        "logp_solubility",
        "qed_solubility",
        "drd_logp",
        "drd_solubility",
        "gsk_logp",
    ),
}
DROP_KEYS = ("formal_cot_trace", "raw_output", "raw_output_steps")


class PromptBuilder(Protocol):
    """The upstream prompt interface shared by all chemistry task families."""

    @property
    def system_prompt(self) -> str | Callable[[dict[str, Any]], str]: ...

    def build_user_prompt(self, record: dict[str, Any]) -> str: ...


def format_entry(record: dict[str, Any], family: str, subtask: str, prompt_builder: PromptBuilder) -> dict[str, Any]:
    upstream_record = {key: value for key, value in record.items() if key not in DROP_KEYS}
    return {
        "id": record["anonymous_sample_id"],
        "task_family": family,
        "subtask": subtask,
        "subset_for_metrics": record["reporting_task"],
        "system_prompt": (
            prompt_builder.system_prompt(record)
            if callable(prompt_builder.system_prompt)
            else prompt_builder.system_prompt
        ),
        "problem": prompt_builder.build_user_prompt(record),
        "upstream_record": upstream_record,
    }


def recover_equivalence_inputs(record: dict[str, Any]) -> dict[str, Any] | None:
    # Match original SMILES inputs, not the suffix of CANONICAL_SMILES outputs.
    values = re.findall(r'(?<![A-Za-z_])SMILES\("([^\"]+)"\)', record["raw_output"])
    if len(values) < 2:
        if record["anonymous_sample_id"] != "mol_und.smiles_equivalent.0097":
            raise ValueError(f"Missing equivalence input pair: {record['anonymous_sample_id']}")
        warnings.warn("Skipping mol_und.smiles_equivalent.0097: truncated trace lacks the second input")
        return None
    record = dict(record)
    record.update(smiles=values[0], smiles_a=values[0], smiles_b=values[1])
    record[record["source_subtask"]] = values[1]
    return record


class OptimizationPromptBuilder:
    def __init__(self, subtask: str) -> None:
        from evaluation.mol_opt.prompt import build_system_prompt
        from evaluation.mol_opt.utils import MULTI_SUBTASK_TO_PROPS

        self.subtask = subtask
        self.is_multi = subtask in MULTI_SUBTASK_TO_PROPS
        self.system_prompt = build_system_prompt(subtask, self.is_multi)

    def build_user_prompt(self, record: dict[str, Any]) -> str:
        from evaluation.mol_opt.prompt import build_user_prompt

        return build_user_prompt(self.subtask, self.is_multi, record)


def populate_source_properties(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from evaluation.mol_opt.utils import MULTI_SUBTASK_TO_PROPS, get_oracle

    oracle_for = lru_cache(maxsize=None)(get_oracle)
    for record in records:
        if not record.get("src"):
            raise ValueError(f"Missing source molecule: {record['anonymous_sample_id']}")
        for prop in MULTI_SUBTASK_TO_PROPS.get(record["subtask"], [record["subtask"]]):
            oracle = oracle_for(prop)
            # TDC catches model errors and substitutes zero. Fail preparation instead
            # of silently writing a false source property value into the question.
            evaluate = getattr(oracle, "evaluator_func", oracle)
            value = float(evaluate(record["src"]))
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite source {prop}: {record['anonymous_sample_id']}")
            record[f"src_{prop}"] = value
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    os.environ["CHEMCOT_DATA_DIR"] = args.data_dir
    from evaluation.core.released_data import load_released_records

    count = 0
    with redirect_stdout(sys.stderr), open(args.output, "w", encoding="utf-8") as stream:
        for family, subtasks in SUBTASKS.items():
            cls = (
                OptimizationPromptBuilder
                if family == "mol_opt"
                else import_module(f"evaluation.{family}.prompt_builder").PromptBuilder
            )
            for subtask in subtasks:
                print(f"Preparing ChemCoTBench: {family}/{subtask}", flush=True)
                builder = cls(subtask)
                records = load_released_records(family, subtask)
                if family == "mol_opt":
                    records = populate_source_properties(records)
                for record in records:
                    if subtask == "smiles_equivalent":
                        record = recover_equivalence_inputs(record)
                        if record is None:
                            continue
                    row = format_entry(record, family, subtask, builder)
                    converted = {
                        "responses_create_params": {
                            "input": [
                                {"role": "system", "content": row.pop("system_prompt")},
                                {"role": "user", "content": row.pop("problem")},
                            ]
                        },
                        "verifier_metadata": row,
                    }
                    stream.write(json.dumps(converted, ensure_ascii=False, allow_nan=False) + "\n")
                    count += 1
    if count != 5219:
        raise ValueError(f"Unexpected ChemCoTBench release size: {count}")


if __name__ == "__main__":
    main()
