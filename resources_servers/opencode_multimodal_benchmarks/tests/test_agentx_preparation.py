# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise source conversion, portable bundling and planning as one pipeline."""

import base64
import hashlib
import json

import pytest
from PIL import Image
from pydantic import TypeAdapter

from benchmarks.opencode_multimodal_benchmarks import agentx_prepare, agentx_source
from benchmarks.opencode_multimodal_benchmarks.agentx_media import read_image
from benchmarks.opencode_multimodal_benchmarks.bundle_agentx import bundle
from benchmarks.opencode_multimodal_benchmarks.canonical import fingerprint
from benchmarks.opencode_multimodal_benchmarks.data import sha256_file
from nemo_gym.data import TaskDataValidator
from nemo_gym.eval.eval_plan import PlanBenchmark, PlanGym, build_eval_plan
from resources_servers.opencode_multimodal_benchmarks.task_data import TaskData


@pytest.fixture
def source_release(tmp_path, monkeypatch):
    root = tmp_path / "release"
    (root / "files").mkdir(parents=True)
    records, references, tasks = {}, {}, {}
    for identifier, color in [("2", "red"), ("522", "blue")]:
        image = root / "files" / f"{identifier}.png"
        Image.new("RGB", (4, 4), color).save(image)
        record = {"dialogs": [{"role": "user", "content": "Is the color red or blue?"}], "files": [{"type": "image"}]}
        reference = {"query_type": "Objective", "final_answer": {"value": color, "justification": "visible color"}}
        records[identifier], references[identifier] = record, [reference]
        task = agentx_source.reference_metadata(identifier, record, reference)
        tasks[identifier] = {
            "reference_sha256": fingerprint(agentx_source.reference_identity(task)),
            "modality": "Image",
            "media": [[image.name, sha256_file(image)]],
        }
    for name, value in [("dataset.json", records), ("data.json", references)]:
        (root / name).write_text(json.dumps(value))
    manifest = {
        "samples": 2,
        "tasks": tasks,
        "source_repo": "synthetic",
        "source_revision": "fixture",
        "additional_generative_ids": [],
        "source_sha256": {name: sha256_file(root / name) for name in ("dataset.json", "data.json")},
    }
    monkeypatch.setattr(agentx_source, "manifest", lambda _: manifest)
    monkeypatch.setattr(agentx_prepare, "manifest", lambda _: manifest)
    monkeypatch.chdir(tmp_path)
    return root


@pytest.fixture
def source_video_release(source_release):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    root = source_release
    video = root / "files" / "600.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (16, 16))
    if not writer.isOpened():
        pytest.skip("MP4 encoder unavailable")
    try:
        for value in (10, 40, 80, 120, 160, 200):
            writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    # Establish fixture identities by sequentially decoding known indices, independently of sample_video.
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    frames = []
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FPS) == 2.0
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 6
        for index in range(6):
            ok, frame = capture.read()
            assert ok
            if index in (0, 2, 5):
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                assert ok
                frames.append(encoded.tobytes())
    finally:
        capture.release()
    record = {"dialogs": [{"role": "user", "content": "Does it become lighter?"}], "files": [{"type": "video"}]}
    reference = {"query_type": "Objective", "final_answer": {"value": "yes", "justification": "brightness rises"}}
    task = agentx_source.reference_metadata("600", record, reference)
    for name, value in (("dataset.json", record), ("data.json", [reference])):
        path = root / name
        records = json.loads(path.read_text())
        records["600"] = value
        path.write_text(json.dumps(records))
    canonical = agentx_prepare.manifest("AgentX")
    canonical["samples"] = 3
    canonical["source_sha256"] = {name: sha256_file(root / name) for name in ("dataset.json", "data.json")}
    canonical["tasks"]["600"] = {
        "reference_sha256": fingerprint(agentx_source.reference_identity(task)),
        "modality": "Video",
        "media": [[video.name, sha256_file(video)]],
        "video_sampling": {"fps": 2.0, "frame_count": 6, "indices": [0, 2, 5]},
        "video_frames_sha256": [hashlib.sha256(frame).hexdigest() for frame in frames],
    }
    return root, frames


def test_video_source_bundle_roundtrip_preserves_frames_and_sampling(source_video_release, tmp_path):
    root, expected_frames = source_video_release
    output = agentx_prepare.prepare(source_root=str(root))
    original = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["task_id"] for row in original] == ["2", "522", "600"]
    video = original[-1]
    assert video["verifier_metadata"]["video_sampling"] == {"fps": 2.0, "frame_count": 6, "indices": [0, 2, 5]}
    content = video["responses_create_params"]["input"][0]["content"]
    assert [read_image(part["image_url"])[0] for part in content[1:]] == expected_frames
    portable = tmp_path / "portable"
    receipt = bundle(output, portable, sha256=sha256_file(output))
    # Remove the source/prepared-media paths from their original location before importing.
    root.rename(tmp_path / "unavailable-source")
    prepared = agentx_prepare.prepare(
        dataset=str(portable / "agentx.jsonl"), sha256=receipt["dataset_sha256"], media_root=str(portable / "media")
    )
    assert [json.loads(line) for line in prepared.read_text().splitlines()] == original


@pytest.mark.parametrize("limit", [None, 1])
@pytest.mark.parametrize("mutation", ["replace", "reorder", "duplicate", "fps", "indices", "frame_count"])
def test_portable_video_tampering_is_rejected_before_limit(source_video_release, tmp_path, limit, mutation):
    root, _ = source_video_release
    output = agentx_prepare.prepare(source_root=str(root))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    video = rows[-1]  # Outside the requested smoke limit.
    task = video["verifier_metadata"]
    content = video["responses_create_params"]["input"][0]["content"]
    if mutation == "replace":
        content[1] = rows[0]["responses_create_params"]["input"][0]["content"][1]
    elif mutation == "reorder":
        content[1], content[2] = content[2], content[1]
    elif mutation == "duplicate":
        content[2] = content[1]
    else:
        task["video_sampling"][mutation] = {"fps": 4.0, "indices": [0, 3, 5], "frame_count": 7}[mutation]
        content[0]["text"] = agentx_source.policy_prompt(task)
    # Caller-controlled frame digests and the outer checksum must not establish canonical identity.
    task["video_frames_sha256"] = [
        hashlib.sha256(read_image(part["image_url"])[0]).hexdigest() for part in content[1:]
    ]
    modified = tmp_path / "modified.jsonl"
    modified.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="video frames or sampling differ from pinned source"):
        agentx_prepare.prepare(dataset=str(modified), sha256=sha256_file(modified), limit=limit)


def test_source_conversion_rejects_decoder_drift(source_video_release, monkeypatch):
    root, _ = source_video_release
    sample = agentx_source.sample_video

    def changed_frames(path, directory):
        frames, sampling = sample(path, directory)
        frames[1].write_bytes(frames[0].read_bytes())
        return frames, sampling

    monkeypatch.setattr(agentx_source, "sample_video", changed_frames)
    with pytest.raises(ValueError, match="video frames or sampling differ from pinned source"):
        agentx_source.convert_source(root, root / "prepared")


def test_source_bundle_roundtrip_preserves_bytes_ids_and_strict_schema(source_release, tmp_path):
    output = agentx_prepare.prepare(source_root=str(source_release))
    original = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["task_id"] for row in original] == ["2", "522"]
    validator = TaskDataValidator("opencode_multimodal_benchmarks", TypeAdapter(TaskData), str(output))
    for index, row in enumerate(original):
        validator.validate_row(index, row)
        url = row["responses_create_params"]["input"][0]["content"][1]["image_url"]
        assert url.startswith("data:image/png;base64,")
        assert read_image(url)[0] == (source_release / "files" / f"{row['task_id']}.png").read_bytes()
        assert "visible color" not in json.dumps(row["responses_create_params"])
    assert validator.report.clean, validator.report.summary()
    portable = tmp_path / "portable"
    receipt = bundle(output, portable, sha256=sha256_file(output))
    prepared = agentx_prepare.prepare(
        dataset=str(portable / "agentx.jsonl"), sha256=receipt["dataset_sha256"], media_root=str(portable / "media")
    )
    assert [json.loads(line) for line in prepared.read_text().splitlines()] == original
    params = dict(
        benchmark=PlanBenchmark(name="agentx", config_paths=[]),
        taskset="visual_tasks",
        gym=PlanGym(version="test", commit=None),
    )
    assert build_eval_plan(original, **params) == build_eval_plan(original[::-1], **params)
    selected = build_eval_plan(original, selected_task_ids=["522"], **params)
    assert [task.task_id.task_id for task in selected.tasks] == ["522"]


def test_embedded_media_tampering_and_conflicting_task_id_are_rejected(source_release):
    rows = agentx_source.convert_source(source_release, source_release / "prepared")
    rows[0]["task_id"] = "wrong"
    with pytest.raises(ValueError, match="task_id"):
        agentx_prepare.validate_rows(rows)
    rows[0]["task_id"] = "2"
    first, second = [row["responses_create_params"]["input"][0]["content"][1] for row in rows]
    first["image_url"] = second["image_url"]
    with pytest.raises(ValueError, match="differs from public source"):
        agentx_prepare.validate_rows(rows)


@pytest.mark.parametrize(
    "value",
    [
        "data:image/png;base64,!bad!",
        "data:image/png,abc",
        "data:image/png;base64," + base64.b64encode(b"not an image").decode(),
    ],
)
def test_embedded_media_must_be_valid_image_bytes(value):
    with pytest.raises((ValueError, OSError)):
        read_image(value)


@pytest.mark.parametrize("limit", [None, 1])
@pytest.mark.parametrize(
    "mutation",
    [
        "append_answer",
        "extra_text",
        "extra_message",
        "instructions",
        "prompt",
        "previous_response_id",
        "conversation",
        "text",
    ],
)
def test_portable_bundle_rejects_modified_policy_prompt_even_with_matching_checksum(
    source_release, tmp_path, mutation, limit
):
    output = agentx_prepare.prepare(source_root=str(source_release))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    row = rows[-1]  # Validate even when this row is outside the requested smoke limit.
    params = row["responses_create_params"]
    leaked = "The reference answer is " + row["verifier_metadata"]["answer"]
    if mutation == "append_answer":
        params["input"][0]["content"][0]["text"] += "\n" + leaked
    elif mutation == "extra_text":
        params["input"][0]["content"].append({"type": "input_text", "text": leaked})
    elif mutation == "extra_message":
        params["input"].append({"role": "user", "content": [{"type": "input_text", "text": leaked}]})
    elif mutation == "text":
        from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

        params["text"] = {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string", "const": row["verifier_metadata"]["answer"]}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            }
        }
        # The model request schema accepts this; canonical preparation must reject it.
        NeMoGymResponseCreateParamsNonStreaming.model_validate(params)
    else:
        params[mutation] = leaked
    modified = tmp_path / "modified.jsonl"
    modified.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="canonical policy"):
        agentx_prepare.prepare(dataset=str(modified), sha256=sha256_file(modified), limit=limit)


@pytest.mark.parametrize("null_text", [False, True])
def test_portable_bundle_preserves_sampling_with_omitted_or_null_text(source_release, tmp_path, null_text):
    output = agentx_prepare.prepare(source_root=str(source_release))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    for row in rows:
        params = row["responses_create_params"]
        assert "text" not in params
        if null_text:
            params["text"] = None
        params.update(temperature=0.5, top_p=0.9, max_output_tokens=128)
    portable = tmp_path / "sampling.jsonl"
    portable.write_text("".join(json.dumps(row) + "\n" for row in rows))
    imported = agentx_prepare.prepare(dataset=str(portable), sha256=sha256_file(portable))
    assert [json.loads(line) for line in imported.read_text().splitlines()] == rows


def test_canonical_question_can_contain_reference_answer(source_release):
    rows = agentx_source.convert_source(source_release, source_release / "prepared")
    assert all(row["verifier_metadata"]["answer"] in row["verifier_metadata"]["question"] for row in rows)
    agentx_prepare.validate_rows(rows)


def test_portable_bundle_rejects_tool_definitions_with_reference_hints(source_release, tmp_path):
    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

    output = agentx_prepare.prepare(source_root=str(source_release))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    params = rows[0]["responses_create_params"]
    params["tools"] = [
        {
            "type": "function",
            "name": "helper",
            "description": "The reference answer is " + rows[0]["verifier_metadata"]["answer"],
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]
    # This is a valid model request: canonical preparation must reject it before inference.
    NeMoGymResponseCreateParamsNonStreaming.model_validate(params)
    modified = tmp_path / "with-tools.jsonl"
    modified.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="cannot include task-supplied tools"):
        agentx_prepare.prepare(dataset=str(modified), sha256=sha256_file(modified))


@pytest.mark.parametrize("tools", [None, []])
def test_empty_tool_settings_do_not_change_canonical_bundle(source_release, tools):
    rows = agentx_source.convert_source(source_release, source_release / "prepared")
    rows[0]["responses_create_params"]["tools"] = tools
    agentx_prepare.validate_rows(rows)


def test_video_prompt_preserves_frame_timestamps_and_rejects_appended_answers(source_video_release):
    from benchmarks.opencode_multimodal_benchmarks.data import ANSWER_INSTRUCTION

    root, _ = source_video_release
    rows = agentx_source.convert_source(root, root / "prepared")
    text = rows[-1]["responses_create_params"]["input"][0]["content"][0]
    assert text["text"] == (
        "This is a video:\nFrame 1 sampled at 0.00 seconds: <image>\n"
        "Frame 2 sampled at 1.00 seconds: <image>\nFrame 3 sampled at 2.50 seconds: <image>\n\n"
        "Question: Does it become lighter?\n\n" + ANSWER_INSTRUCTION
    )
    agentx_prepare.validate_rows(rows)
    text["text"] += "\nThe reference answer is red"
    with pytest.raises(ValueError, match="canonical policy prompt mismatch"):
        agentx_prepare.validate_rows(rows)
