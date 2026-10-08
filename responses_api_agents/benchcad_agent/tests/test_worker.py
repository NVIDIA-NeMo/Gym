# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from responses_api_agents.benchcad_agent.worker import export_records, final_text, score_qa


@pytest.fixture
def upstream():
    default = Path(__file__).resolve().parents[3] / "benchmarks/benchcad/.cache/upstream"
    root = Path(os.environ.get("BENCHCAD_TEST_ROOT", default)).resolve()
    if not (root / "QA/scoring/qa_score.py").is_file():
        pytest.skip("Run BenchCAD preparation or set BENCHCAD_TEST_ROOT to the pinned upstream checkout")
    return root


@pytest.mark.parametrize("tag", ["think", "thinking"])
def test_reasoning_does_not_leak_into_answer(tag):
    assert final_text(f"<{tag}>[99]\nreasoning</{tag}>\n[1, 2]") == "[1, 2]"


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("[10, 3, -5, 0]", 1.0),
        ("[5, 2, -10, 0]", 0.5),
        ("[10, 3, 5, 0]", 0.75),
        ("```json\n[10, 3, -5, 0]\n```", 1.0),
        ("[NaN, 3, -5, 0]", 0.0),
        ("[Infinity, 3, -5, 0]", 0.0),
        ("[10]", 0.0),
        ("", 0.0),
    ],
)
def test_upstream_numeric_scoring(upstream, answer, expected):
    pairs = [
        {"answer": 10, "type": "dim"},
        {"answer": 3, "type": "count"},
        {"answer": -5, "type": "ratio"},
        {"answer": 0, "type": "bool"},
    ]
    assert score_qa(upstream, answer, pairs)["reward"] == expected


def test_qa_export_preserves_prompt_and_withholds_answers(upstream, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(upstream))
    destination = export_records(upstream, source=upstream / "QA/test_data", output=tmp_path, task="code_qa", limit=2)
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        task = json.loads((tmp_path / "code_qa" / row["record_id"] / "task.json").read_text())
        assert "qa_pairs" not in row
        assert "answer" not in row
        prompt = row["responses_create_params"]["input"][0]["content"]
        assert "CadQuery code:" in prompt
        assert all(pair["question"] in prompt for pair in task["qa_pairs"])
        assert task["images"] == []


def test_export_rejects_path_traversal(upstream, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(upstream))
    source = tmp_path / "source"
    source.mkdir()
    (source / "records.jsonl").write_text(json.dumps({"record_id": "../escape"}))
    with pytest.raises(ValueError, match="Unsafe"):
        export_records(upstream, source=source, output=tmp_path / "out", task="code_qa", limit=None)


@pytest.mark.asyncio
async def test_worker_reaps_timed_out_process(tmp_path, monkeypatch):
    from responses_api_agents.benchcad_agent import runtime

    worker = tmp_path / "sleep.py"
    worker.write_text("import time\ntime.sleep(20)\n")
    monkeypatch.setattr(runtime, "WORKER", worker)
    with pytest.raises(TimeoutError):
        await runtime.run_worker(Path(sys.executable), tmp_path, timeout=0.05)


@pytest.mark.asyncio
async def test_worker_non_utf8_diagnostics(tmp_path, monkeypatch):
    from responses_api_agents.benchcad_agent import runtime

    worker = tmp_path / "fail.py"
    worker.write_text("import os\nos.write(2, b'failure: \\xff')\nraise SystemExit(3)\n")
    monkeypatch.setattr(runtime, "WORKER", worker)
    with pytest.raises(RuntimeError, match="exited 3: failure"):
        await runtime.run_worker(Path(sys.executable), tmp_path)


@pytest.mark.asyncio
async def test_worker_clears_camera_ablation(tmp_path, monkeypatch):
    from responses_api_agents.benchcad_agent import runtime

    worker = tmp_path / "env.py"
    worker.write_text("import os\nprint(os.environ.get('BENCH_VIEW_HINT', 'canonical'))\n")
    monkeypatch.setattr(runtime, "WORKER", worker)
    monkeypatch.setenv("BENCH_VIEW_HINT", "perturb")
    assert (await runtime.run_worker(Path(sys.executable), tmp_path)).strip() == "canonical"


def test_runtime_rejects_wrong_revision(tmp_path, monkeypatch):
    from responses_api_agents.benchcad_agent import runtime

    monkeypatch.setattr(runtime.subprocess, "check_output", Mock(return_value=b"wrong-revision"))
    install = Mock()
    monkeypatch.setattr(runtime.subprocess, "run", install)
    with pytest.raises(ValueError, match="checkout must be"):
        runtime.ensure_runtime(tmp_path)
    install.assert_not_called()


def test_runtime_pins_checkout_and_separates_cad_python(tmp_path, monkeypatch):
    from responses_api_agents.benchcad_agent import runtime

    root = tmp_path / "upstream"
    monkeypatch.setattr(runtime.subprocess, "check_output", Mock(return_value=runtime.UPSTREAM_REVISION.encode()))
    install = Mock()
    monkeypatch.setattr(runtime.subprocess, "run", install)
    python = runtime.ensure_runtime(root)
    commands = [call.args[0] for call in install.call_args_list]
    assert commands[0] == ["git", "clone", runtime.UPSTREAM_URL, str(root)]
    assert commands[1] == ["git", "checkout", runtime.UPSTREAM_REVISION]
    assert commands[2] == ["uv", "venv", "--python", "3.12", str(root / ".venv")]
    assert commands[3][:5] == ["uv", "pip", "install", "--python", str(python)]
    assert "--override" in commands[3]


def test_cli_patch_never_executes_prediction(upstream, tmp_path, monkeypatch, capsys):
    from responses_api_agents.benchcad_agent import worker

    answer = tmp_path / "answer.txt"
    answer.write_text("```python\nimport cadquery as cq\nraise RuntimeError('must not execute')\n```")
    output = tmp_path / "patched.py"
    monkeypatch.setattr(
        sys, "argv", ["worker", "--root", str(upstream), "patch", "--answer", str(answer), "--output", str(output)]
    )
    worker.main()
    assert json.loads(capsys.readouterr().out) == {"has_code": True}
    assert "/workspace/prediction.step" in output.read_text()
    answer.write_text("No program")
    worker.main()
    assert json.loads(capsys.readouterr().out) == {"has_code": False}


def test_cli_qa_scoring(upstream, tmp_path, monkeypatch, capsys):
    from responses_api_agents.benchcad_agent import worker

    (tmp_path / "task.json").write_text(json.dumps({"task": "code_qa", "qa_pairs": [{"answer": 10}]}))
    answer = tmp_path / "answer.txt"
    answer.write_text("[5]")
    monkeypatch.setattr(
        sys, "argv", ["worker", "--root", str(upstream), "score", "--answer", str(answer), "--task-dir", str(tmp_path)]
    )
    worker.main()
    assert json.loads(capsys.readouterr().out)["reward"] == 0.5


@pytest.mark.parametrize("task,expected", [("vision2code", 0.8), ("codeedit", 0.5)])
def test_cli_uses_native_edit_normalization(upstream, tmp_path, monkeypatch, capsys, task, expected):
    from responses_api_agents.benchcad_agent import worker

    monkeypatch.syspath_prepend(str(upstream))
    from benchcad_core.scoring import iou

    score = Mock(return_value=0.8)
    monkeypatch.setattr(iou, "iou_step_vs_step", score)
    (tmp_path / "task.json").write_text(json.dumps({"task": task, "baseline_iou": 0.6}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker",
            "--root",
            str(upstream),
            "score",
            "--answer",
            str(tmp_path / "answer"),
            "--task-dir",
            str(tmp_path),
            "--step",
            str(tmp_path / "generated.step"),
        ],
    )
    worker.main()
    assert json.loads(capsys.readouterr().out)["reward"] == expected
    score.assert_called_once_with(tmp_path / "generated.step", tmp_path / "reference.step")


@pytest.mark.parametrize("task", ["vision2code", "codeedit", "vision_qa"])
def test_export_keeps_reference_geometry_private(upstream, tmp_path, monkeypatch, task):
    from responses_api_agents.benchcad_agent import worker

    monkeypatch.syspath_prepend(str(upstream))
    from benchcad_core.scoring import exec_cq

    source = tmp_path / "source"
    source.mkdir()
    (source / "input.py").write_text("import cadquery as cq\nresult = cq.Workplane('XY').box(1,2,3)")
    (source / "reference.py").write_text("PRIVATE_REFERENCE_CODE")
    (source / "image.png").write_bytes(b"canonical image bytes")
    record = {
        "record_id": "part",
        "instruction": "Widen the part",
        "orig_code_path": "input.py",
        "gt_code_path": "reference.py",
        "gt_step_path": "reference.step",
        "iou": 0.3,
        "step_path": "reference.step",
        "image_path": "image.png",
        "qa_pairs": [{"question": "Size?", "answer": 2}],
    }
    (source / "records.jsonl").write_text(json.dumps(record))
    # Rendering is exercised with real CAD in the native smoke; this test checks
    # that the exact renderer output is copied rather than a dataset preview.
    if task == "vision2code":
        (source / "reference.step").write_bytes(b"PRIVATE_STEP")
        builder = Mock(return_value=("system", "user", [source / "image.png"]))
        real_loader = worker.load_module
        monkeypatch.setattr(
            worker,
            "load_module",
            lambda root, path: (
                type("Prompt", (), {"build": staticmethod(builder)})
                if path == "Vision2Code/pipeline/prompt.py"
                else real_loader(root, path)
            ),
        )

    def execute(code, destination):
        assert code == "PRIVATE_REFERENCE_CODE"
        destination.write_bytes(b"PRIVATE_STEP")

    monkeypatch.setattr(exec_cq, "execute_cq_to_step", execute)
    output = tmp_path / "output"
    path = export_records(upstream, source=source, output=output, task=task, limit=None)
    raw = path.read_text()
    assert "PRIVATE_REFERENCE_CODE" not in raw
    assert "PRIVATE_STEP" not in raw
    if task in {"vision2code", "vision_qa"}:
        assert (output / task / "part/view_0.png").read_bytes() == b"canonical image bytes"
        assert "/workspace/view_0.png" in raw
    else:
        assert (output / task / "part/reference.step").read_bytes() == b"PRIVATE_STEP"


@pytest.mark.parametrize("task", ["vision2code", "codeedit", "qa"])
def test_upstream_downloads_are_revision_pinned(upstream, tmp_path, monkeypatch, task):
    import huggingface_hub

    from responses_api_agents.benchcad_agent import worker

    single = Mock()
    snapshot = Mock()
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", single)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)

    def converter(path, *, run_name):
        assert path.endswith(".py") and run_name == "__main__"
        huggingface_hub.hf_hub_download("BenchCAD/BenchCAD", "data.parquet")
        huggingface_hub.snapshot_download("BenchCAD/BenchCAD")
        assert sys.argv[sys.argv.index("--limit") + 1] == "1"
        assert ("--max-shards" in sys.argv) is (task == "vision2code")

    monkeypatch.setattr(worker.runpy, "run_path", converter)
    monkeypatch.setattr(
        sys,
        "argv",
        ["worker", "--root", str(upstream), "download", "--task", task, "--output", str(tmp_path), "--limit", "1"],
    )
    worker.main()
    assert single.call_args.kwargs["revision"] == worker.DATASET_REVISION
    assert snapshot.call_args.kwargs["revision"] == worker.DATASET_REVISION
