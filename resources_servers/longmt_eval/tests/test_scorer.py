# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoring dependency tests, also runnable directly in the Python 3.12 environment."""

import gzip
import importlib
import io
import os
import urllib.request
from pathlib import Path

import pytest


@pytest.fixture(scope="module", autouse=True)
def require_scoring_dependencies_in_ci() -> None:
    if os.environ.get("LONGMT_SCORER_TESTS_REQUIRED") == "1":
        for name in ("torch", "ersatz.split", "laser_encoders.download_models", "comet"):
            importlib.import_module(name)


class TestErsatzWeightsOnlyLoad:
    """The ersatz judge checkpoint pickles an ``argparse.Namespace`` ('args'), which
    torch>=2.6's ``weights_only=True`` default rejects. The pinned ersatz fork
    (>= d21f404) allowlists Namespace via ``torch.serialization.add_safe_globals`` so the
    judge loads WITHOUT a ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`` global override.

    These guard that contract with a tiny synthetic checkpoint — no multi-GB model
    download. If the pin regresses to an ersatz without the allowlist (or someone drops
    the ``add_safe_globals`` call), ``load_model`` raises ``UnpicklingError`` and these fail.
    """

    def _write_namespace_checkpoint(self, tmp_path: Path):
        import argparse

        import torch

        # Mirror the real checkpoint shape: a pickled argparse.Namespace under 'args'
        # (the part weights_only=True refuses), plus opaque tokenizer bytes and a
        # here-empty 'weights' state dict. Real weights are irrelevant to the load gate.
        ckpt = {
            "args": argparse.Namespace(vocab_size=8, transformer_nlayers=1, foo="bar"),
            "tokenizer": b"<sentencepiece-bytes>",
            "weights": {},
        }
        path = tmp_path / "ersatz_ckpt.pt"
        torch.save(ckpt, str(path))
        return path

    def _stub_model_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Exercise only load_model's torch.load path; stub the heavy tokenizer/model build.
        import ersatz.split as split

        class _FakeModel:
            def load_state_dict(self, *_a, **_k):
                pass

            def eval(self):
                return self

        monkeypatch.setattr(split, "SentencePiece", lambda **_k: object())
        monkeypatch.setattr(split, "ErsatzTransformer", lambda *_a, **_k: _FakeModel())

    def test_loads_namespace_checkpoint_without_env_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pytest.importorskip("ersatz.split")
        import ersatz.split as split
        import torch

        # Ensure we're testing the real torch default, not the global escape hatch.
        monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
        self._stub_model_construction(monkeypatch)
        ckpt = self._write_namespace_checkpoint(tmp_path)

        # add_safe_globals is process-global and sticky, so clear it first (restoring
        # after) to force load_model to perform the registration itself — otherwise a
        # regression that dropped the call could pass on residue from another test.
        ser = torch.serialization
        prior = list(ser.get_safe_globals()) if hasattr(ser, "get_safe_globals") else None
        if prior is not None:
            ser.clear_safe_globals()
        try:
            # Must NOT raise UnpicklingError under torch>=2.6 weights_only=True default.
            model = split.load_model(str(ckpt))
            assert model is not None
        finally:
            if prior is not None:
                ser.clear_safe_globals()
                if prior:
                    ser.add_safe_globals(prior)

    def test_does_not_disable_weights_only(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The fix must be add_safe_globals + the safe default, NOT weights_only=False."""
        pytest.importorskip("ersatz.split")
        import ersatz.split as split
        import torch

        monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
        self._stub_model_construction(monkeypatch)
        ckpt = self._write_namespace_checkpoint(tmp_path)

        captured: dict = {}
        real_load = torch.load

        def _spy_load(*args, **kwargs):
            captured["weights_only"] = kwargs.get("weights_only")
            return real_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", _spy_load)
        split.load_model(str(ckpt))
        # weights_only=False would defeat the protection wholesale; the allowlist keeps
        # the default (True on torch>=2.6). Absent kwarg (None) is also acceptable.
        assert captured.get("weights_only") is not False


class TestDownloadOnce:
    """Unit tests for _download_once() in segale_actor.py — the generic race-safe
    publish guard shared by the ersatz/laser/comet fetchers.

    Deliberately free of library or network deps: a fake ``produce`` callback plus
    real tmp dirs exercises every branch (fast-path, won-race file/dir, lost-race
    file/dir), so these run everywhere and are the cheapest coverage of the logic
    most likely to corrupt a shared cache.
    """

    def test_fast_path_skips_produce_when_present(self, tmp_path: Path) -> None:
        from scoring import scorer as segale_actor

        dest = tmp_path / "sub" / "f.bin"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"cached")
        called: list[int] = []

        segale_actor._download_once(str(dest), lambda tmp: called.append(1))

        assert called == []
        assert dest.read_bytes() == b"cached"

    def test_publishes_file_atomically(self, tmp_path: Path) -> None:
        from scoring import scorer as segale_actor

        dest = tmp_path / "sub" / "f.bin"  # parent does not exist yet
        segale_actor._download_once(str(dest), lambda tmp: Path(tmp).write_bytes(b"hello"))

        assert dest.read_bytes() == b"hello"
        assert not list(dest.parent.glob("*.tmp"))

    def test_publishes_directory_atomically(self, tmp_path: Path) -> None:
        from scoring import scorer as segale_actor

        dest = tmp_path / "model"

        def produce(tmp: str) -> None:
            ck = Path(tmp) / "checkpoints"
            ck.mkdir(parents=True)
            (ck / "model.ckpt").write_bytes(b"ckpt")

        segale_actor._download_once(str(dest), produce)

        assert (dest / "checkpoints" / "model.ckpt").read_bytes() == b"ckpt"
        assert not list(tmp_path.glob("*.tmp"))

    def test_lost_race_file_keeps_winner(self, tmp_path: Path) -> None:
        from scoring import scorer as segale_actor

        dest = tmp_path / "race" / "f.bin"

        def produce(tmp: str) -> None:
            Path(tmp).write_bytes(b"ours")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"theirs")  # a competitor publishes while we "download"

        segale_actor._download_once(str(dest), produce)

        assert dest.read_bytes() == b"theirs"  # winner kept, our copy discarded
        assert not list(dest.parent.glob("*.tmp"))

    def test_lost_race_directory_keeps_winner(self, tmp_path: Path) -> None:
        from scoring import scorer as segale_actor

        dest = tmp_path / "model"

        def produce(tmp: str) -> None:
            Path(tmp).mkdir()
            (Path(tmp) / "x").write_bytes(b"ours")
            dest.mkdir()
            (dest / "x").write_bytes(b"theirs")  # competitor publishes the dir first

        segale_actor._download_once(str(dest), produce)

        assert (dest / "x").read_bytes() == b"theirs"  # our tmp dir was rmtree'd
        assert not list(tmp_path.glob("*.tmp"))


class TestDownloadTo:
    """Unit tests for _download_to() — the single URL->file download body."""

    def test_plain_download(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from scoring import scorer as segale_actor

        monkeypatch.setattr(urllib.request, "urlopen", lambda url: io.BytesIO(b"raw-bytes"))
        out = tmp_path / "out.bin"
        segale_actor._download_to("http://x/f", str(out))

        assert out.read_bytes() == b"raw-bytes"

    def test_gunzip_download(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from scoring import scorer as segale_actor

        raw = b"decompressed-payload"
        monkeypatch.setattr(urllib.request, "urlopen", lambda url: io.BytesIO(gzip.compress(raw)))
        out = tmp_path / "out.bin"
        segale_actor._download_to("http://x/f.gz", str(out), gunzip=True)

        assert out.read_bytes() == raw


class TestDownloadErsatzModel:
    """Unit tests for _download_ersatz_model(): path wiring against the real ersatz
    registry (ERSATZ_DIR/MODELS) with the network mocked — no multi-GB fetch."""

    def _patch_registry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str = "http://x/model.gz"
    ) -> Path:
        eu = pytest.importorskip("ersatz.utils")
        monkeypatch.setattr(eu, "ERSATZ_DIR", str(tmp_path / "ersatz_home"))
        monkeypatch.setattr(
            eu,
            "MODELS",
            {"default-multilingual": {"destination": os.path.join("models", "dm"), "source": source}},
        )
        return Path(eu.ERSATZ_DIR) / "models" / "dm"

    def test_downloads_and_gunzips(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from scoring import scorer as segale_actor

        dest = self._patch_registry(monkeypatch, tmp_path)
        raw = b"ersatz-weights"
        monkeypatch.setattr(urllib.request, "urlopen", lambda url: io.BytesIO(gzip.compress(raw)))

        segale_actor._download_ersatz_model("default-multilingual")

        assert dest.read_bytes() == raw
        assert not list(dest.parent.glob("*.tmp"))

    def test_idempotent_second_call_skips_download(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from scoring import scorer as segale_actor

        dest = self._patch_registry(monkeypatch, tmp_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"cached")

        def _boom(url):
            raise AssertionError("must not download when the checkpoint already exists")

        monkeypatch.setattr(urllib.request, "urlopen", _boom)

        segale_actor._download_ersatz_model("default-multilingual")

        assert dest.read_bytes() == b"cached"


class TestDownloadLaserModel:
    """Unit tests for _download_laser_model(): fetches laser2.{pt,spm,cvocab} via the
    real LaserModelDownloader.base_url plumbing, with the network mocked."""

    def _patch_downloader(self, monkeypatch: pytest.MonkeyPatch, base_url: str = "http://x/laser") -> None:
        dl = pytest.importorskip("laser_encoders.download_models")

        class _FakeDownloader:
            def __init__(self, model_dir):
                self.base_url = base_url

        monkeypatch.setattr(dl, "LaserModelDownloader", _FakeDownloader)

    def test_downloads_all_three(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from scoring import scorer as segale_actor

        self._patch_downloader(monkeypatch)
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda url: io.BytesIO(b"data-" + url.rsplit("/", 1)[-1].encode())
        )
        model_dir = tmp_path / "laser_home"

        segale_actor._download_laser_model(str(model_dir))

        for fn in ("laser2.pt", "laser2.spm", "laser2.cvocab"):
            assert (model_dir / fn).read_bytes() == b"data-" + fn.encode()
        assert not list(model_dir.glob("*.tmp"))

    def test_skips_existing_files(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from scoring import scorer as segale_actor

        self._patch_downloader(monkeypatch)
        model_dir = tmp_path / "laser_home"
        model_dir.mkdir()
        (model_dir / "laser2.pt").write_bytes(b"cached-pt")

        fetched: list[str] = []

        def _urlopen(url):
            fetched.append(url.rsplit("/", 1)[-1])
            return io.BytesIO(b"new")

        monkeypatch.setattr(urllib.request, "urlopen", _urlopen)

        segale_actor._download_laser_model(str(model_dir))

        assert fetched == ["laser2.spm", "laser2.cvocab"]  # laser2.pt skipped via fast-path
        assert (model_dir / "laser2.pt").read_bytes() == b"cached-pt"


class TestDownloadCometModel:
    """Unit tests for _download_comet_model(): the produce-into-tmp then
    resolve-from-dest flow, with comet.download_model mocked (no HF/URL fetch)."""

    @staticmethod
    def _fake_download_model(calls: list):
        def fake(model, saving_directory=None, local_files_only=False):
            calls.append((saving_directory, local_files_only))
            ck = Path(saving_directory) / "checkpoints"
            ck.mkdir(parents=True, exist_ok=True)
            p = ck / "model.ckpt"
            if not p.exists():
                p.write_bytes(b"ckpt")
            return str(p)

        return fake

    def test_two_call_flow_and_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from scoring import scorer as segale_actor

        comet = pytest.importorskip("comet")
        calls: list = []
        monkeypatch.setattr(comet, "download_model", self._fake_download_model(calls))
        monkeypatch.setenv("LONGMT_COMET_CACHE", str(tmp_path / "comet_cache"))

        ckpt = segale_actor._download_comet_model("Unbabel/wmt22-cometkiwi-da")

        dest = tmp_path / "comet_cache" / "Unbabel__wmt22-cometkiwi-da"
        assert ckpt == str(dest / "checkpoints" / "model.ckpt")
        assert Path(ckpt).read_bytes() == b"ckpt"
        # First call produces into a private .tmp dir; second resolves locally from dest.
        assert len(calls) == 2
        assert calls[0][0].endswith(".tmp")
        assert calls[1] == (str(dest), True)
        assert not list(dest.parent.glob("*.tmp"))

    def test_default_path_uses_configured_gym_cache_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from scoring import scorer as segale_actor

        comet = pytest.importorskip("comet")
        calls: list = []
        monkeypatch.setattr(comet, "download_model", self._fake_download_model(calls))
        monkeypatch.setenv("LONGMT_CACHE_DIR", str(tmp_path / "gym_cache"))
        monkeypatch.delenv("LONGMT_COMET_CACHE", raising=False)

        ckpt = segale_actor._download_comet_model("model")

        dest = tmp_path / "gym_cache" / "longmt-comet" / "model"
        assert ckpt == str(dest / "checkpoints" / "model.ckpt")

    def test_idempotent_second_call_only_resolves(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from scoring import scorer as segale_actor

        comet = pytest.importorskip("comet")
        monkeypatch.setattr(comet, "download_model", self._fake_download_model([]))
        monkeypatch.setenv("LONGMT_COMET_CACHE", str(tmp_path / "comet_cache"))
        segale_actor._download_comet_model("m")  # populate the cache

        calls: list = []
        monkeypatch.setattr(comet, "download_model", self._fake_download_model(calls))

        ckpt = segale_actor._download_comet_model("m")

        dest = tmp_path / "comet_cache" / "m"
        # Fast-path skips produce entirely; only the local resolve runs.
        assert calls == [(str(dest), True)]
        assert ckpt == str(dest / "checkpoints" / "model.ckpt")
