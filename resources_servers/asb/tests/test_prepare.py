# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selector expansion, and the reconciliation that fixes ASB's denominators."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction

import pytest

from benchmarks.asb import prepare
from benchmarks.asb import upstream_spec as spec


UPSTREAM = prepare.UPSTREAM_DIR

pytestmark = pytest.mark.skipif(
    not (UPSTREAM / "data" / "agent_task.jsonl").exists(),
    reason="pinned ASB checkout absent; run `python -m benchmarks.asb.prepare fetch`",
)


def test_upstream_hashes_match_pinned_revision():
    prepare.verify_upstream_hashes(UPSTREAM)


def test_corrupt_upstream_file_is_still_html_and_never_parsed():
    """``agent_task_pot_all.jsonl`` is a failed GitHub fetch committed as data.

    If upstream ever fixes it, this fails on purpose: the fix must be read before the
    file is trusted, because nothing else would notice it becoming parseable.
    """
    corrupt = UPSTREAM / "data" / "agent_task_pot_all.jsonl"
    assert corrupt.read_bytes().lstrip()[:5].lower() == b"<html"
    assert "agent_task_pot_all.jsonl" not in spec.DATA_FILE_SHA256


@pytest.mark.parametrize("condition", prepare.published_conditions())
def test_every_published_condition_is_400_rows(condition):
    """The denominator behind every published ASB cell.

    ``main_attacker.py`` defaults ``task_num`` to 1 and the DPI/OPI/MP/mixed drivers never
    override it, giving 10 agents x 1 task x 40 agent-matched tools. ``config/POT.yml``
    sets 2 over a 5-agent task file, which lands on 400 as well.
    """
    rows = list(prepare.expand_selectors(upstream_dir=UPSTREAM, **condition))
    assert len(rows) == spec.PUBLISHED_ROWS_PER_CONDITION


def test_published_percentages_are_consistent_with_these_denominators():
    """Published cells quantize to 1/N for the N this adapter uses.

    This is the evidence the denominators were recovered correctly rather than assumed.
    A defense-table entry must be a whole number of rows out of 400; a main-table DPI
    entry a whole number out of 5 x 400; Mixed out of 3 x 400. Expanding the full task
    list instead (2,040/condition) satisfies none of them.
    """
    defense_cells = ["91.00", "98.75", "33.75", "64.25", "43.25", "99.75", "55.50"]
    for cell in defense_cells:
        assert (Fraction(cell) / 100 * 400).denominator == 1, cell

    dpi_cells = ["87.10", "96.75", "25.20", "51.10", "85.65", "98.40", "60.35"]
    for cell in dpi_cells:
        assert (Fraction(cell) / 100 * 400 * 5).denominator == 1, cell

    # Gemma2-9B's Mixed cell resolves only on a denominator divisible by 3.
    assert (Fraction("92.17") / 100 * 400 * 3).denominator != 1  # 92.17 is rounded
    assert round(float(Fraction(1106, 1200)) * 100, 2) == 92.17

    # And the full-task-list expansion is ruled out for the defense table.
    assert (Fraction("91.00") / 100 * 2040).denominator != 1


def test_memory_corpus_sizes_confirm_the_400_row_write_run():
    """Upstream's shipped memory stores hold ~400 records each.

    Each was written by a DPI run under one attack type, so its size is an independent
    witness of that run's row count -- and it agrees with ``task_num=1``.
    """
    corpus = prepare.extract_memory_corpus(UPSTREAM)
    assert set(corpus) == set(spec.ATTACK_TYPES)
    for attack_type, records in corpus.items():
        assert 390 <= len(records) <= 470, (attack_type, len(records))


def test_dpi_row_appends_the_exact_injection_string():
    rows = list(
        prepare.expand_selectors(
            upstream_dir=UPSTREAM, injection_method="direct_prompt_injection", attack_type="combined_attack"
        )
    )
    row = rows[0]
    metadata = row["verifier_metadata"]
    expected = spec.build_attack_prompts(
        attacker_instruction=metadata["attacker_tool"]["instruction"],
        attacker_tool=metadata["attacker_tool"]["name"],
    )["combined_attack"]
    assert metadata["task_input"].endswith(expected)
    assert metadata["task_input"] == f"{metadata['task']} {expected}"
    # The injection is in the user turn the model actually receives.
    assert row["responses_create_params"]["input"][-1]["content"] == metadata["task_input"]


def test_clean_condition_withholds_the_attacker_tool():
    """Clean rows must not expose the attacker tool, or ASR would be measurable there."""
    rows = list(
        prepare.expand_selectors(upstream_dir=UPSTREAM, injection_method="clean", attack_type="combined_attack")
    )
    metadata = rows[0]["verifier_metadata"]
    assert metadata["attacker_tool_available"] is False
    names = [entry["function"]["name"] for entry in metadata["tools_schema"]]
    assert metadata["attacker_tool"]["name"] not in names
    assert metadata["task_input"] == metadata["task"]


def test_opi_row_leaves_the_task_clean_but_arms_the_observation():
    rows = list(
        prepare.expand_selectors(
            upstream_dir=UPSTREAM, injection_method="observation_prompt_injection", attack_type="naive"
        )
    )
    metadata = rows[0]["verifier_metadata"]
    assert metadata["task_input"] == metadata["task"], "OPI must not touch the user prompt"
    assert metadata["injects_into_observations"] is True
    assert metadata["attacker_tool"]["name"] in metadata["observation_injection"]


def test_attacker_and_normal_tool_schemas_keep_upstreams_asymmetry():
    """Normal tools carry ``parameters: None``; the attacker tool omits the key entirely.

    Both are rendered into the planning prompt verbatim, so normalizing them would change
    what the model reads.
    """
    rows = list(
        prepare.expand_selectors(
            upstream_dir=UPSTREAM, injection_method="direct_prompt_injection", attack_type="naive"
        )
    )
    schemas = rows[0]["verifier_metadata"]["tools_schema"]
    assert all("parameters" in entry["function"] for entry in schemas[:-1])
    assert schemas[-1]["function"].get("parameters", "absent") == "absent"


def test_pot_backdoor_appends_trigger_and_plants_the_poisoned_example():
    rows = list(
        prepare.expand_selectors(
            upstream_dir=UPSTREAM,
            injection_method="pot_backdoor",
            attack_type="naive",
            trigger=spec.DEFAULT_POT_TRIGGER,
        )
    )
    row = rows[0]
    metadata = row["verifier_metadata"]
    assert metadata["task_input"].endswith(f"{spec.DEFAULT_POT_TRIGGER}.")
    system_prompt = row["responses_create_params"]["input"][1]["content"]
    assert spec.DEFAULT_POT_TRIGGER in system_prompt
    assert metadata["attacker_tool"]["name"] in system_prompt


def test_pot_clean_omits_the_trigger_so_the_backdoor_should_not_fire():
    """PoT Clean is the control: same planted plan, no trigger in the task."""
    rows = list(
        prepare.expand_selectors(
            upstream_dir=UPSTREAM,
            injection_method="pot_clean",
            attack_type="naive",
            trigger=spec.DEFAULT_POT_TRIGGER,
        )
    )
    metadata = rows[0]["verifier_metadata"]
    assert metadata["task_input"] == metadata["task"]
    assert not metadata["task_input"].endswith(f"{spec.DEFAULT_POT_TRIGGER}.")


def test_delimiters_defense_wraps_task_and_amends_the_system_prompt():
    rows = list(
        prepare.expand_selectors(
            upstream_dir=UPSTREAM,
            injection_method="direct_prompt_injection",
            attack_type="combined_attack",
            defense_type="delimiters_defense",
        )
    )
    row = rows[0]
    metadata = row["verifier_metadata"]
    # The wrap encloses the task but not the injection appended after it.
    assert metadata["task_input"].startswith("<start>")
    assert "<end>" in metadata["task_input"]
    assert row["responses_create_params"]["input"][0]["content"].endswith(spec.DELIMITERS_SYSTEM_SUFFIX)


def test_memory_condition_moves_plan_instruction_to_the_user_role():
    """Under read_db upstream re-roles the plan instruction so memory can answer it."""
    rows = list(prepare.expand_selectors(upstream_dir=UPSTREAM, injection_method="memory_attack", attack_type="naive"))
    messages = rows[0]["responses_create_params"]["input"]
    assert messages[1]["role"] == "user"
    assert rows[0]["verifier_metadata"]["reads_memory"] is True


def test_row_ids_are_unique_across_the_whole_matrix():
    seen = set()
    for condition in prepare.published_conditions():
        for row in prepare.expand_selectors(upstream_dir=UPSTREAM, **condition):
            assert row["asb_id"] not in seen
            seen.add(row["asb_id"])
    assert len(seen) == 27 * spec.PUBLISHED_ROWS_PER_CONDITION


def test_expansion_is_deterministic():
    """Same inputs, same rows -- the property that makes pinning meaningful."""
    kwargs = {"injection_method": "direct_prompt_injection", "attack_type": "naive"}
    first = [json.dumps(r, sort_keys=True) for r in prepare.expand_selectors(upstream_dir=UPSTREAM, **kwargs)]
    second = [json.dumps(r, sort_keys=True) for r in prepare.expand_selectors(upstream_dir=UPSTREAM, **kwargs)]
    assert first == second


def test_pull_defaults_to_a_pinned_revision_not_a_branch(monkeypatch, tmp_path):
    """A branch name is not a pin. The published repo's main already moved once.

    Recorded here rather than left to review: the failure it prevents is silent, since a
    moved branch still yields a working dataset and only shows up as an unexplained change
    in a later run's numbers.
    """
    from benchmarks.asb import prepare

    seen = {}

    def fake_download(**kwargs):
        seen.update(kwargs)
        (tmp_path / "all.jsonl").write_text("", encoding="utf-8")
        (tmp_path / "manifest.json").write_text(
            json.dumps({"content_hash": hashlib.sha256(b"").hexdigest(), "rows_total": 0}), encoding="utf-8"
        )
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    prepare.pull_from_hub(prepare.PUBLISHED_HF_REPO, data_dir=tmp_path)
    assert seen["revision"] == prepare.PUBLISHED_HF_REVISION
    assert len(prepare.PUBLISHED_HF_REVISION) == 40, "a pin is a full commit sha, not a branch or short sha"


def test_pull_rejects_rows_that_do_not_match_their_own_manifest(monkeypatch, tmp_path):
    """Scoring against a substituted or truncated copy is worse than failing to pull one."""
    from benchmarks.asb import prepare

    def fake_download(**kwargs):
        (tmp_path / "all.jsonl").write_text(json.dumps({"asb_id": "x"}) + "\n", encoding="utf-8")
        (tmp_path / "manifest.json").write_text(
            json.dumps({"content_hash": "0" * 64, "rows_total": 1}), encoding="utf-8"
        )
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    with pytest.raises(ValueError, match="do not match the manifest"):
        prepare.pull_from_hub(prepare.PUBLISHED_HF_REPO, data_dir=tmp_path)


def test_pull_rejects_a_short_download(monkeypatch, tmp_path):
    """A partial snapshot hashes differently, but state the row count too so the cause is obvious."""
    from benchmarks.asb import prepare

    rows = [{"asb_id": "a"}, {"asb_id": "b"}]
    kept = rows[:1]

    def fake_download(**kwargs):
        (tmp_path / "all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        content = hashlib.sha256("".join(json.dumps(r, sort_keys=True) for r in kept).encode()).hexdigest()
        (tmp_path / "manifest.json").write_text(
            json.dumps({"content_hash": content, "rows_total": len(rows)}), encoding="utf-8"
        )
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    with pytest.raises(ValueError, match="expected 2 rows, pulled 1"):
        prepare.pull_from_hub(prepare.PUBLISHED_HF_REPO, data_dir=tmp_path)
