# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for `python -m nemo_gym.sweep`, the surface a user actually types.

The library tests in test_sweep_manifest.py call the functions directly, so they miss everything
argparse and main() do: exit codes, which errors are caught and printed as one-liners versus
escaping as a traceback, and whether the flags are wired to the arguments they claim.
"""

import json

import pytest
import yaml

from nemo_gym.sweep.__main__ import main


def _env(tmp_path, *, label="alpha", rows=4, agent="alpha_agent", nickname="sweep", config="alpha.yaml"):
    """A manifest whose configs and data exist, so the CLI can be driven end to end."""
    # the agent must be a top-level key: validate parses the config directly rather than through
    # Gym's resolver, so nesting it would not count as declaring it
    (tmp_path / config).write_text(
        yaml.safe_dump({agent: {"responses_api_agents": {"simple_agent": {"entrypoint": "app.py"}}}})
    )
    (tmp_path / "alpha.jsonl").write_text(
        "".join(
            json.dumps({"agent_ref": {"type": "responses_api_agents", "name": agent}, "i": i}) + "\n"
            for i in range(rows)
        )
    )
    manifest = {
        "nickname": nickname,
        "gym_eval_run": {"num_repeats": 2},
        "entries": [
            {
                "label": label,
                "agent": agent,
                "configs": [config],
                "data": str(tmp_path / "alpha.jsonl"),
            }
        ],
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return path


def test_validate_returns_zero_on_a_good_manifest(tmp_path, capsys):
    assert main(["validate", str(_env(tmp_path)), "--repo-root", str(tmp_path)]) == 0
    assert "OK" in capsys.readouterr().out


def test_validate_reports_a_bad_manifest_as_one_line_not_a_traceback(tmp_path, capsys):
    """A typo'd top-level key is the likeliest mistake against an extra="forbid" schema, and it
    used to escape main() as a raw pydantic traceback."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("nickname: t\nenteries: []\n")
    assert main(["validate", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "enteries" in err


def test_validate_reports_a_missing_manifest(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "nope.yaml")]) == 1
    assert "not found" in capsys.readouterr().err


def test_validate_skip_data_still_catches_a_missing_data_file(tmp_path, capsys):
    """--skip-data is a fast config check; a typo'd path silently passing is what it would hide."""
    path = _env(tmp_path)
    doc = yaml.safe_load(path.read_text())
    doc["entries"][0]["data"] = str(tmp_path / "gone.jsonl")
    path.write_text(yaml.safe_dump(doc))
    assert main(["validate", str(path), "--repo-root", str(tmp_path), "--skip-data"]) == 1
    assert "data not found" in capsys.readouterr().err


def test_materialize_writes_the_resume_gate_and_reports_counts(tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["materialize", str(_env(tmp_path)), "--repo-root", str(tmp_path), "--out-dir", str(out)]) == 0
    sweep = out / "sweep"
    # both files must exist for --resume to take its fast path
    assert (sweep / "rollouts_materialized_inputs.jsonl").is_file()
    assert (sweep / "rollouts.jsonl").is_file()
    assert json.loads((sweep / "sweep_report.json").read_text())["total_materialized_rows"] == 8
    assert "8" in capsys.readouterr().out


def test_materialize_limit_per_entry_caps_rows(tmp_path):
    out = tmp_path / "out"
    main(
        [
            "materialize",
            str(_env(tmp_path)),
            "--repo-root",
            str(tmp_path),
            "--out-dir",
            str(out),
            "--limit-per-entry",
            "2",
        ]
    )
    rows = [json.loads(x) for x in (out / "sweep" / "rollouts_materialized_inputs.jsonl").read_text().splitlines()]
    assert len({r["_ng_task_index"] for r in rows}) == 2


def test_materialize_refuses_to_overwrite_without_the_flag(tmp_path, capsys):
    out = tmp_path / "out"
    path = _env(tmp_path)
    assert main(["materialize", str(path), "--repo-root", str(tmp_path), "--out-dir", str(out)]) == 0
    assert main(["materialize", str(path), "--repo-root", str(tmp_path), "--out-dir", str(out)]) == 1
    assert "already exists" in capsys.readouterr().err


def test_shard_merge_split_round_trip_through_the_cli(tmp_path, capsys):
    """The three commands a sharded run drives, exercised the way a launcher drives them."""
    out = tmp_path / "out"
    main(["materialize", str(_env(tmp_path)), "--repo-root", str(tmp_path), "--out-dir", str(out)])
    sweep = out / "sweep"

    assert main(["shard", str(sweep), "--num-shards", "2"]) == 0
    assert "dealt" in capsys.readouterr().out

    # stand in for collection: every shard claims its own inputs
    for shard in sorted(p for p in (sweep / "shards").glob("shard_*") if p.is_dir()):
        (shard / "rollouts.jsonl").write_text((shard / "rollouts_materialized_inputs.jsonl").read_text())

    assert main(["merge", str(sweep / "shards"), "--output", str(sweep / "rollouts.jsonl")]) == 0
    assert "merged 8" in capsys.readouterr().out

    assert main(["split", str(sweep)]) == 0
    assert (sweep / "by_label" / "alpha" / "rollouts.jsonl").is_file()


def test_split_on_the_wrong_directory_fails_instead_of_writing_nothing(tmp_path, capsys):
    """Pointing at <out-dir> rather than <out-dir>/<nickname> used to exit 0 having done nothing."""
    (tmp_path / "sweep_report.json").write_text(json.dumps({"entries": {}}))
    (tmp_path / "rollouts_materialized_inputs.jsonl").write_text("")
    assert main(["split", str(tmp_path)]) == 1
    assert "wrong directory" in capsys.readouterr().err


def test_shard_rejects_a_nonsense_count(tmp_path, capsys):
    out = tmp_path / "out"
    main(["materialize", str(_env(tmp_path)), "--repo-root", str(tmp_path), "--out-dir", str(out)])
    assert main(["shard", str(out / "sweep"), "--num-shards", "0"]) == 1
    assert capsys.readouterr().err


def test_container_config_unions_the_paths_of_several_manifests(tmp_path, capsys):
    first = _env(tmp_path)
    second_dir = tmp_path / "b"
    second_dir.mkdir()
    second = _env(second_dir, label="beta", agent="beta_agent", nickname="two", config="beta.yaml")
    target = tmp_path / "cc.yaml"
    assert main(["container-config", str(first), str(second), "--out", str(target)]) == 0
    doc = yaml.safe_load(target.read_text())
    # dummy values so the config resolves at build time without secrets
    assert "policy_base_url" in doc and "nv_inference_api_key" in doc
    assert len(doc["config_paths"]) >= 2
    assert "config paths" in capsys.readouterr().out


def test_build_emits_a_runnable_config(tmp_path):
    out = tmp_path / "out"
    assert main(["build", str(_env(tmp_path)), "--repo-root", str(tmp_path), "--out-dir", str(out)]) == 0
    assert (out / "sweep" / "sweep_config.yaml").is_file()


def test_no_subcommand_is_an_argparse_error(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_validate_prints_warnings_for_entries_sharing_an_agent(tmp_path, capsys):
    """Sharing an agent means sharing num_repeats -- a warning, not an error, and the shipped
    manifest trips it, so the printing path is load-bearing."""
    path = _env(tmp_path)
    doc = yaml.safe_load(path.read_text())
    doc["entries"].append({**doc["entries"][0], "label": "beta"})
    path.write_text(yaml.safe_dump(doc))
    assert main(["validate", str(path), "--repo-root", str(tmp_path)]) == 0
    assert "warn:" in capsys.readouterr().out


def test_validate_reports_a_missing_config(tmp_path, capsys):
    path = _env(tmp_path)
    doc = yaml.safe_load(path.read_text())
    doc["entries"][0]["configs"] = ["gone.yaml"]
    path.write_text(yaml.safe_dump(doc))
    assert main(["validate", str(path), "--repo-root", str(tmp_path)]) == 1
    assert "config not found" in capsys.readouterr().err


def test_validate_reports_a_missing_gym_env_start_config_path(tmp_path, capsys):
    """These are only caught here; otherwise gym env start fails inside the container."""
    path = _env(tmp_path)
    doc = yaml.safe_load(path.read_text())
    doc["gym_env_start"] = {"config_paths": ["nowhere/vllm_model.yaml"]}
    path.write_text(yaml.safe_dump(doc))
    assert main(["validate", str(path), "--repo-root", str(tmp_path)]) == 1
    assert "gym_env_start" in capsys.readouterr().err


def test_validate_reports_unparseable_data(tmp_path, capsys):
    path = _env(tmp_path)
    (tmp_path / "alpha.jsonl").write_text("{ not json\n")
    assert main(["validate", str(path), "--repo-root", str(tmp_path)]) == 1
    assert capsys.readouterr().err


def test_validate_rejects_a_manifest_that_is_not_a_mapping(tmp_path, capsys):
    bad = tmp_path / "list.yaml"
    bad.write_text("- just\n- a list\n")
    assert main(["validate", str(bad)]) == 1
    assert "must be a mapping" in capsys.readouterr().err


def test_validate_rejects_malformed_yaml(tmp_path, capsys):
    bad = tmp_path / "broken.yaml"
    bad.write_text("nickname: [unclosed\n")
    assert main(["validate", str(bad)]) == 1
    assert "not valid YAML" in capsys.readouterr().err


def test_build_reports_an_agent_ref_override(tmp_path, capsys):
    """The override is recorded rather than applied silently, since it changes which verifier
    scores the rows."""
    path = _env(tmp_path)
    doc = yaml.safe_load(path.read_text())
    doc["entries"][0]["agent_ref_override"] = "alpha_agent"  # same name, so it still validates
    path.write_text(yaml.safe_dump(doc))
    assert main(["build", str(path), "--repo-root", str(tmp_path), "--out-dir", str(tmp_path / "o")]) == 0
    assert "overrides applied" in capsys.readouterr().out


def test_shard_and_merge_report_carried_and_duplicate_rows(tmp_path, capsys):
    """Resharding a half-collected sweep prints what it carried; merging a rerun shard prints
    what it dropped. Both are the reassurance a user reads after a failure."""
    out = tmp_path / "out"
    main(["materialize", str(_env(tmp_path)), "--repo-root", str(tmp_path), "--out-dir", str(out)])
    sweep = out / "sweep"
    main(["shard", str(sweep), "--num-shards", "2"])
    for shard in sorted(p for p in (sweep / "shards").glob("shard_*") if p.is_dir()):
        (shard / "rollouts.jsonl").write_text((shard / "rollouts_materialized_inputs.jsonl").read_text())
    capsys.readouterr()

    main(["shard", str(sweep), "--num-shards", "4"])  # reshard over collected work
    out_text = capsys.readouterr().out
    assert "absorbed" in out_text and "carried" in out_text

    # a shard that claims rows another already has: merge must drop the duplicates
    shards = sorted(p for p in (sweep / "shards").glob("shard_*") if p.is_dir())
    body = "".join((s / "rollouts_materialized_inputs.jsonl").read_text() for s in shards)
    for s in shards:
        (s / "rollouts.jsonl").write_text(body)
    main(["merge", str(sweep / "shards"), "--output", str(sweep / "rollouts.jsonl")])
    assert "duplicate" in capsys.readouterr().out


def test_split_names_labels_that_collected_nothing(tmp_path, capsys):
    """A label at zero means that lane failed rather than scored badly -- the silence is the
    finding, so it has to be printed."""
    out = tmp_path / "out"
    main(["materialize", str(_env(tmp_path)), "--repo-root", str(tmp_path), "--out-dir", str(out)])
    assert main(["split", str(out / "sweep")]) == 0
    assert "no rollouts for" in capsys.readouterr().out
