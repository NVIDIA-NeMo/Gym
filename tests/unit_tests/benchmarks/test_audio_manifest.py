# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral checks for the helpers shared by the audio benchmarks."""

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import audio_manifest as am


def _source(**fields) -> dict:
    return {
        "id": "utt-1",
        "messages": [{"role": "user", "content": "Transcribe.", "audio": {"path": "/data/a.wav", "duration": 1.5}}],
        "expected_answer": "hello",
        **fields,
    }


def _contract(**fields) -> dict:
    return {"request": "greedy", "prompt_override": {}, **fields}


def test_request_presets_and_overrides() -> None:
    greedy = am.request_controls({"request": "greedy"})
    assert (greedy["temperature"], greedy["top_p"], greedy["max_output_tokens"], greedy["seed"]) == (0.0, 1.0, 256, 0)
    assert greedy["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}

    reasoning = am.request_controls({"request": {"preset": "reasoning", "max_output_tokens": 1024}})
    assert reasoning["max_output_tokens"] == 1024
    assert reasoning["temperature"] == 0.6
    assert reasoning["extra_body"]["thinking_token_budget"] == 17408

    custom = am.request_controls({"request": {"preset": "greedy", "extra_body": {"top_k": 5}}})
    assert custom["extra_body"]["top_k"] == 5 and custom["extra_body"]["min_p"] == 0.0
    # Presets are never mutated by an override.
    assert "top_k" not in am.REQUEST_PRESETS["greedy"]["extra_body"]
    assert am.request_controls({"request": "greedy"}) == greedy


def test_build_row_stores_request_prompt_and_source() -> None:
    source = _source()
    snapshot = json.loads(json.dumps(source))
    contract = _contract(prompt_override={"system_prompt": "", "user_prompt": "Verbatim."})
    row = am.build_row(source, profile="demo", contract=contract, index=3)

    assert source == snapshot, "the source row must not be modified"
    params = row["responses_create_params"]
    assert params["input"] == [{"role": "user", "content": "Verbatim."}]
    assert params["metadata"]["audio_path"] == "/data/a.wav"
    assert (params["temperature"], params["top_p"], params["max_output_tokens"]) == (0.0, 1.0, 256)
    assert json.loads(params["metadata"]["extra_body"])["seed"] == 0
    assert row["benchmark_profile"] == "demo" and row["sample_id"] == "demo:3"
    assert row["task_id"] == row["sample_id"]
    assert row["ng_source_row"] == snapshot
    assert (
        row["ng_source_row_sha256"]
        == hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert row["id"] == "utt-1" and row["expected_answer"] == "hello" and "messages" not in row


def test_the_row_is_stable_under_a_jsonl_round_trip_and_its_digest_matches_the_stored_source() -> None:
    """An integer mapping key is a string once written to JSONL: hashing the in-memory dict would then mismatch."""
    source = _source(per_speaker={2: "b", 10: "a"})
    row = am.build_row(source, profile="demo", contract=_contract(), index=1)
    assert row["ng_source_row"]["per_speaker"] == {"2": "b", "10": "a"}
    read_back = json.loads(json.dumps(row))
    assert (
        am.source_hash(read_back["ng_source_row"]) == read_back["ng_source_row_sha256"] == row["ng_source_row_sha256"]
    )
    assert source["per_speaker"] == {2: "b", 10: "a"}, "the caller's source row is not modified"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_source_row_that_cannot_be_hashed_reliably_fails_at_prepare_time(bad: float) -> None:
    with pytest.raises(ValueError, match="Out of range float|not JSON compliant"):
        am.build_row(_source(score=bad), profile="demo", contract=_contract(), index=1)


def test_prompt_override_system_handling() -> None:
    messages = [{"role": "system", "content": "before"}, {"role": "user", "content": "q", "audio": {"path": "/a.wav"}}]
    keep = am.build_row({"messages": messages}, profile="p", contract=_contract(), index=1)
    assert [m["role"] for m in keep["responses_create_params"]["input"]] == ["system", "user"]

    replaced = am.build_row(
        {"messages": messages}, profile="p", contract=_contract(prompt_override={"system_prompt": "after"}), index=1
    )
    assert replaced["responses_create_params"]["input"][0] == {"role": "system", "content": "after"}

    dropped = am.build_row(
        {"messages": messages}, profile="p", contract=_contract(prompt_override={"system_prompt": ""}), index=1
    )
    assert [m["role"] for m in dropped["responses_create_params"]["input"]] == ["user"]

    only_user = [{"role": "user", "content": "q", "audio": {"path": "/a.wav"}}]
    inserted = am.build_row(
        {"messages": only_user}, profile="p", contract=_contract(prompt_override={"system_prompt": "s"}), index=1
    )
    assert inserted["responses_create_params"]["input"][0] == {"role": "system", "content": "s"}


def test_user_prompt_override_requires_one_text_user_message() -> None:
    two_users = [{"role": "user", "content": "a", "audio": {"path": "/a.wav"}}, {"role": "user", "content": "b"}]
    contract = _contract(prompt_override={"user_prompt": "x"})
    with pytest.raises(ValueError):
        am.build_row({"messages": two_users}, profile="p", contract=contract, index=1)
    blocks = [{"role": "user", "content": [{"type": "text", "text": "a"}], "audio": {"path": "/a.wav"}}]
    with pytest.raises(ValueError, match="string user message"):
        am.build_row({"messages": blocks}, profile="p", contract=contract, index=1)


def test_multiple_audio_clips_keep_order_and_repeats() -> None:
    messages = [{"role": "user", "content": "Compare.", "audios": [{"path": "/one.wav"}, {"path": "/one.wav"}]}]
    row = am.build_row({"messages": messages}, profile="p", contract=_contract(), index=1)
    metadata = row["responses_create_params"]["metadata"]
    assert "audio_path" not in metadata
    assert json.loads(metadata["audio_paths"]) == ["/one.wav", "/one.wav"]


def test_row_without_audio_is_rejected() -> None:
    with pytest.raises(ValueError, match="no audio"):
        am.build_row({"messages": [{"role": "user", "content": "q"}]}, profile="p", contract=_contract(), index=4)


def test_load_profiles_caches_and_notices_changes(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"profiles": {"a": {}}}))
    assert list(am.load_profiles(path)["profiles"]) == ["a"]
    assert am.load_profiles(path) is am.load_profiles(path)
    path.write_text(json.dumps({"profiles": {"a": {}, "b": {}}}))
    assert list(am.load_profiles(path)["profiles"]) == ["a", "b"]
    path.write_text("{}")
    with pytest.raises(ValueError, match="Invalid audio benchmark profiles"):
        am.load_profiles(path)


def test_audio_prefix(tmp_path: Path) -> None:
    assert am.audio_prefix(tmp_path) == str(tmp_path.resolve())
    assert am.audio_prefix(tmp_path, "/data/x/") == "/data/x"


def test_download_verifies_sha256_and_reuses_a_good_file(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    target = tmp_path / "out" / "copy.bin"

    assert am.download(source.as_uri(), target, digest) == target
    assert target.read_bytes() == b"payload"
    source.unlink()  # a verified file is not downloaded again
    assert am.download(source.as_uri(), target, digest) == target

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        am.download(target.as_uri(), tmp_path / "bad.bin", "0" * 64)
    assert not (tmp_path / "bad.bin").exists() and not (tmp_path / "bad.bin.tmp").exists()


def test_write_jsonl_is_atomic_and_stamps_the_code(tmp_path: Path) -> None:
    code = tmp_path / "prepare.py"
    code.write_text("v1")
    output = tmp_path / "data" / "bench.jsonl"

    assert not am.is_current(output, code_files=[code])
    am.write_jsonl(output, iter([{"a": 1}, {"a": "é"}]), code_files=[code])
    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == [{"a": 1}, {"a": "é"}]
    assert am.is_current(output, code_files=[code])

    code.write_text("v2")  # the code that produced the rows changed
    assert not am.is_current(output, code_files=[code])

    am.write_jsonl(output, [{"a": 1}], code_files=[code], options={"max_samples_per_profile": 1})
    assert not am.is_current(output, code_files=[code]), "non-default options are never reported as current"


def test_write_jsonl_rejects_empty_output_without_touching_existing_data(tmp_path: Path) -> None:
    output = tmp_path / "bench.jsonl"
    output.write_text('{"keep": true}\n')
    with pytest.raises(ValueError, match="No rows"):
        am.write_jsonl(output, iter(()), code_files=[])
    assert output.read_text() == '{"keep": true}\n'
    assert not (tmp_path / "bench.jsonl.tmp").exists()


def test_write_rows_stamps_only_the_options_that_were_set_and_currency_check_follows(tmp_path: Path) -> None:
    code = tmp_path / "prepare.py"
    code.write_text("v1")
    output = tmp_path / "bench.jsonl"
    is_current = am.currency_check([code])

    am.write_rows(output, [{"a": 1}], code_files=[code], audio_root=None, max_samples_per_profile=None)
    assert json.loads(output.with_name("bench.jsonl.stamp").read_text())["options"] == {}
    assert is_current(output)
    am.write_rows(output, [{"a": 1}], code_files=[code], audio_root="/data/x", max_samples_per_profile=None)
    assert json.loads(output.with_name("bench.jsonl.stamp").read_text())["options"] == {"audio_root": "/data/x"}
    assert not is_current(output)


def test_write_rows_stamps_hydra_containers_as_plain_json(tmp_path: Path) -> None:
    # `+prepare_script_args.profiles=[alpha]` reaches prepare() as an OmegaConf ListConfig, which json cannot encode.
    from omegaconf import OmegaConf

    code = tmp_path / "prepare.py"
    code.write_text("v1")
    output = tmp_path / "bench.jsonl"
    overrides = OmegaConf.create({"profiles": ["alpha"], "nested": {"languages": ["de", "fr"]}})

    am.write_rows(output, [{"a": 1}], code_files=[code], profiles=overrides.profiles, nested=overrides.nested)

    assert json.loads(output.with_name("bench.jsonl.stamp").read_text())["options"] == {
        "profiles": ["alpha"],
        "nested": {"languages": ["de", "fr"]},
    }


def test_profile_rows_index_each_profile_from_one_for_shared_or_mapped_sources(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"profiles": {"a": _contract(), "b": _contract()}}))
    sources = [_source(id="x"), _source(id="y")]

    shared = list(am.profile_rows(path, sources))
    assert [row["sample_id"] for row in shared] == ["a:1", "a:2", "b:1", "b:2"]
    assert [row["task_id"] for row in shared] == [row["sample_id"] for row in shared]
    mapped = list(am.profile_rows(path, {"b": iter(sources[:1]), "a": sources}))
    assert [row["sample_id"] for row in mapped] == ["b:1", "a:1", "a:2"]
    assert shared[0]["ng_source_row_sha256"] == am.source_hash(sources[0])


def test_select_normalizes_narrowing_arguments() -> None:
    valid = ("Alpha", "beta", "gamma")
    assert am.select(None, valid, "x") == valid
    assert am.select("gamma, Alpha,gamma", valid, "x") == ("Alpha", "gamma")
    assert am.select(["gamma", "Alpha"], valid, "x", keep_order=True) == ("gamma", "Alpha")
    assert am.select("ALPHA,Beta", valid, "x", ignore_case=True) == ("Alpha", "beta")
    assert am.select("card", valid, "x", {"card": ("beta",)}) == ("beta",)
    for bad in ("", [], "delta", "alpha"):
        with pytest.raises(ValueError, match=r"Empty or unknown x .*valid: Alpha, beta, gamma"):
            am.select(bad, valid, "x")


def test_select_accepts_hydra_values_and_returns_plain_tuples() -> None:
    # `+prepare_script_args.languages=[gamma,Alpha]` arrives as a ListConfig, `=beta` and `=card` as str.
    from omegaconf import ListConfig, OmegaConf

    valid = ("Alpha", "beta", "gamma")
    listed = OmegaConf.create(["gamma", "Alpha"])
    assert isinstance(listed, ListConfig)
    for selected in (
        am.select(listed, valid, "x"),
        am.select(listed, valid, "x", keep_order=True),
        am.select("beta", valid, "x"),
        am.select("card", valid, "x", {"card": OmegaConf.create(["beta"])}),
    ):
        assert type(selected) is tuple and all(type(name) is str for name in selected)
    assert am.select(listed, valid, "x") == ("Alpha", "gamma")
    assert am.select(listed, valid, "x", keep_order=True) == ("gamma", "Alpha")
    with pytest.raises(ValueError, match="Empty or unknown x"):
        am.select(OmegaConf.create([]), valid, "x")
