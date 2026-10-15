# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real tar members through the audio transport adapter."""

import base64
import io
import json
import tarfile
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.requests import Request

from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


def _wav(samples: bytes = b"\x01\x00\x02\x00") -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(samples)
    return output.getvalue()


def _tar(path: Path, members: dict[str, bytes], *, mode: str = "w") -> None:
    with tarfile.open(path, mode) as archive:
        for name, audio in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(audio)
            archive.addfile(member, io.BytesIO(audio))


def _model(audio_root: Path | None) -> VLLMModel:
    config = VLLMModelConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="",
        name="vllm_model",
        base_url="http://localhost:9999/v1",
        api_key="test-key",  # pragma: allowlist secret
        model="audio-model",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        audio_root=str(audio_root) if audio_root is not None else None,
    )
    return VLLMModel(config=config, server_client=MagicMock(spec=ServerClient))


def _ref(member: str = "audio/test.wav", *, archive: str = "audio.tar") -> str:
    return json.dumps({"archive": archive, "member": member})


def _decode(url: str) -> bytes:
    return base64.b64decode(url.split(",", 1)[1], validate=True)


def test_ref_preserves_exact_wav_bytes_and_mime(tmp_path: Path) -> None:
    audio = _wav()
    _tar(tmp_path / "audio.tar", {"audio/test.wav": audio})
    url = _model(tmp_path)._resolve_audio_ref_to_url(_ref())
    assert url.startswith("data:audio/wav;base64,")
    assert _decode(url) == audio
    with wave.open(io.BytesIO(_decode(url)), "rb") as decoded:
        assert (decoded.getframerate(), decoded.getnchannels(), decoded.getsampwidth()) == (16000, 1, 2)


@pytest.mark.parametrize("member,mime", [("clip.FLAC", "flac"), ("clip.mp3", "mpeg"), ("clip.m4a", "mp4")])
def test_ref_uses_member_extension(tmp_path: Path, member: str, mime: str) -> None:
    _tar(tmp_path / "audio.tar", {member: b"encoded audio"})
    url = _model(tmp_path)._resolve_audio_ref_to_url(_ref(member))
    assert url.startswith(f"data:audio/{mime};base64,")
    assert _decode(url) == b"encoded audio"


def test_concurrent_reads_index_once_without_seek_races(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    members = {f"audio/{i}.wav": _wav(bytes([i, 0]) * (i + 1)) for i in range(16)}
    _tar(tmp_path / "audio.tar", members)
    model = _model(tmp_path)
    tar_open = tarfile.open
    scan_count = 0

    def counted_open(*args, **kwargs):
        nonlocal scan_count
        scan_count += 1
        return tar_open(*args, **kwargs)

    monkeypatch.setattr(tarfile, "open", counted_open)
    names = list(members) * 6
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda name: _decode(model._resolve_audio_ref_to_url(_ref(name))), names))
    assert results == [members[name] for name in names]
    assert scan_count == 1


def test_archive_replacement_invalidates_offsets(tmp_path: Path) -> None:
    model = _model(tmp_path)
    first, second = _wav(), _wav(b"\xff\x00" * 12)
    _tar(tmp_path / "audio.tar", {"audio/test.wav": first})
    assert _decode(model._resolve_audio_ref_to_url(_ref())) == first
    replacement = tmp_path / "replacement.tar"
    _tar(replacement, {"padding.wav": b"padding" * 400, "audio/test.wav": second})
    replacement.replace(tmp_path / "audio.tar")
    assert _decode(model._resolve_audio_ref_to_url(_ref())) == second


@pytest.mark.parametrize(
    "reference",
    [
        "",
        "{",
        "null",
        "[]",
        "1",
        "{}",
        '{"archive":"audio.tar"}',
        '{"archive":"audio.tar","member":"test.wav","offset":512}',
        '{"archive":42,"member":"test.wav"}',
        '{"archive":"audio.tar","member":null}',
    ],
)
def test_malformed_refs_fail_with_actionable_error(tmp_path: Path, reference: str) -> None:
    with pytest.raises(ValueError, match="audio_ref"):
        _model(tmp_path)._resolve_audio_ref_to_url(reference)


@pytest.mark.parametrize("field", ["archive", "member"])
@pytest.mark.parametrize(
    "path", ["/absolute.wav", "../escape.wav", "a/../escape.wav", "./clip.wav", "a//b.wav", "a\\b.wav", "a\x00b.wav"]
)
def test_unsafe_reference_paths_are_rejected(tmp_path: Path, field: str, path: str) -> None:
    ref = {"archive": "audio.tar", "member": "audio/test.wav"}
    ref[field] = path
    with pytest.raises(ValueError, match="audio_ref"):
        _model(tmp_path)._resolve_audio_ref_to_url(json.dumps(ref))


def test_archive_symlink_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _tar(tmp_path / "outside.tar", {"audio/test.wav": _wav()})
    (root / "audio.tar").symlink_to(tmp_path / "outside.tar")
    with pytest.raises(ValueError, match="outside audio_root"):
        _model(root)._resolve_audio_ref_to_url(_ref())


def test_root_is_required_for_archive_reference() -> None:
    with pytest.raises(ValueError, match="audio_root"):
        _model(None)._resolve_audio_ref_to_url(_ref())


def test_missing_archive_and_member_are_distinct_errors(tmp_path: Path) -> None:
    model = _model(tmp_path)
    with pytest.raises(FileNotFoundError, match="archive.*does not exist"):
        model._resolve_audio_ref_to_url(_ref())
    _tar(tmp_path / "audio.tar", {"different.wav": _wav()})
    with pytest.raises(FileNotFoundError, match="no member 'audio/test.wav'"):
        model._resolve_audio_ref_to_url(_ref())


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE])
def test_non_regular_entries_cannot_be_read(tmp_path: Path, kind: bytes) -> None:
    with tarfile.open(tmp_path / "audio.tar", "w") as archive:
        info = tarfile.TarInfo("audio/test.wav")
        info.type = kind
        info.linkname = "target.wav"
        archive.addfile(info)
    with pytest.raises(ValueError, match="regular"):
        _model(tmp_path)._resolve_audio_ref_to_url(_ref())


def test_duplicate_member_is_rejected(tmp_path: Path) -> None:
    with tarfile.open(tmp_path / "audio.tar", "w") as archive:
        for content in [b"first", b"second"]:
            info = tarfile.TarInfo("audio/test.wav")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    with pytest.raises(ValueError, match="duplicate"):
        _model(tmp_path)._resolve_audio_ref_to_url(_ref())


def test_unsupported_audio_extension_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported audio extension '.bin'"):
        _model(tmp_path)._resolve_audio_ref_to_url(_ref("audio/test.bin"))


def test_compressed_archive_is_rejected_even_when_named_tar(tmp_path: Path) -> None:
    _tar(tmp_path / "audio.tar", {"audio/test.wav": _wav()}, mode="w:gz")
    with pytest.raises(ValueError, match="uncompressed tar"):
        _model(tmp_path)._resolve_audio_ref_to_url(_ref())


def test_truncated_member_is_rejected(tmp_path: Path) -> None:
    _tar(tmp_path / "audio.tar", {"audio/test.wav": _wav()})
    with (tmp_path / "audio.tar").open("r+b") as handle:
        handle.truncate(530)
    with pytest.raises(ValueError, match="beyond|uncompressed tar"):
        _model(tmp_path)._resolve_audio_ref_to_url(_ref())


@pytest.mark.parametrize("audio_key", ["audio_data", "audio_path", "audio_paths"])
def test_ref_is_mutually_exclusive_with_existing_audio_keys(tmp_path: Path, audio_key: str) -> None:
    body = {"messages": [], "metadata": {"audio_ref": _ref(), audio_key: "present"}}
    with pytest.raises(ValueError, match="mutually exclusive"):
        _model(tmp_path)._preprocess_chat_completion_create_params(MagicMock(), body)


@pytest.mark.parametrize("audio_key", ["audio_ref", "audio_path", "audio_paths", "audio_data"])
@pytest.mark.parametrize("order", ["before_text", "after_text"])
@pytest.mark.parametrize("list_content", [True, False])
def test_audio_content_order_applies_to_every_audio_mode(
    tmp_path: Path, audio_key: str, order: str, list_content: bool
) -> None:
    audio = _wav()
    _tar(tmp_path / "audio.tar", {"audio/test.wav": audio})
    (tmp_path / "test.wav").write_bytes(audio)
    values = {
        "audio_ref": _ref(),
        "audio_path": "test.wav",
        "audio_paths": ["test.wav"],
        "audio_data": "data:audio/wav;base64," + base64.b64encode(audio).decode(),
    }
    model = _model(tmp_path)
    model.config.audio_content_order = order
    text = {"type": "text", "text": "Listen."}
    body = {
        "messages": [{"role": "user", "content": [text] if list_content else "Listen."}],
        "metadata": {audio_key: values[audio_key], "keep": "value"},
    }
    outbound = model._preprocess_chat_completion_create_params(MagicMock(), body)
    content = outbound["messages"][0]["content"]
    assert [part["type"] for part in content] == (
        ["audio_url", "text"] if order == "before_text" else ["text", "audio_url"]
    )
    assert content[1 if order == "before_text" else 0] == text
    assert _decode(content[0 if order == "before_text" else 1]["audio_url"]["url"]) == audio
    assert outbound["metadata"] == {"keep": "value"}


async def test_responses_transport_reads_off_loop_and_retains_compact_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _wav()
    _tar(tmp_path / "audio.tar", {"audio/test.wav": audio})
    model = _model(tmp_path)
    model.config.audio_content_order = "after_text"
    backend = MagicMock()
    backend.create_chat_completion = AsyncMock(
        return_value={
            "id": "chat-test",
            "object": "chat.completion",
            "created": 1,
            "model": "audio-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Heard."}}],
        }
    )
    model._clients = [backend]
    loop_thread = threading.get_ident()
    resolver_threads = []
    resolve = VLLMModel._resolve_audio_ref_to_url

    def record_thread(self: VLLMModel, reference: str) -> str:
        resolver_threads.append(threading.get_ident())
        return resolve(self, reference)

    monkeypatch.setattr(VLLMModel, "_resolve_audio_ref_to_url", record_thread)
    metadata = {"audio_ref": _ref(), "extra_body": json.dumps({"n": 1}), "sample_id": "test-0"}
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "Listen.", "type": "message"}],
        metadata=metadata,
        temperature=0.0,
        max_output_tokens=16,
    )
    original_input = body.model_dump()["input"]
    request = Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "tar-audio-test"}})
    response = await model.responses(request, body)
    backend.create_chat_completion.assert_awaited_once()
    sent = backend.create_chat_completion.call_args.kwargs
    assert sent["n"] == 1
    assert [part["type"] for part in sent["messages"][0]["content"]] == ["text", "audio_url"]
    assert _decode(sent["messages"][0]["content"][1]["audio_url"]["url"]) == audio
    assert "audio_ref" not in (sent.get("metadata") or {})
    assert resolver_threads and all(thread != loop_thread for thread in resolver_threads)
    assert body.metadata == metadata
    assert body.model_dump()["input"] == original_input
    assert response.metadata == metadata
    assert "data:audio/" not in response.model_dump_json()
    assert response.output[0].content[0].text == "Heard."


@pytest.mark.parametrize("native", [True, False])
async def test_ref_rejected_on_unsupported_backend_paths(tmp_path: Path, native: bool) -> None:
    model = _model(tmp_path)
    if native:
        model.config.is_responses_native = True
        body = NeMoGymResponseCreateParamsNonStreaming(input="Listen.", metadata={"audio_ref": _ref()})
        with pytest.raises(ValueError, match="is_responses_native=false"):
            await model.responses(Request({"type": "http", "headers": []}), body)
    else:
        model.config.use_completions_api = True
        body = NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[{"role": "user", "content": "Listen."}], metadata={"audio_ref": _ref()}
        )
        with pytest.raises(ValueError, match="text-only"):
            await model.chat_completions(Request({"type": "http", "headers": []}), body)
