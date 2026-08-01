# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SEGALE Ray actor pool for the longmt_eval resource server.

Mirrors the wmt_translation CometActor pattern. Each _SegaleActor holds
LASER2 and COMETKiwi resident in GPU memory across calls.

Call _build_segale_actor_class() once after Ray.init() to get the remote
class, then instantiate one actor per GPU on the gym node.
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import ray


LOG = logging.getLogger(__name__)


def _mirror_python(cache_env_var: str, default_cache: str) -> Path:
    """Mirror the venv's uv Python install to a shared-FS path for Ray workers.

    Identical strategy to wmt_translation/app.py: uv ships python-build-
    standalone binaries whose absolute paths change across containers. Copying
    the whole python root to a stable shared-FS location lets remote Ray
    workers find a py_executable that resolves at runtime.
    """
    venv_python = Path(sys.executable).resolve()
    if not venv_python.exists():
        raise RuntimeError(f"sys.executable not found: {venv_python}")
    uv_python_root = venv_python.parent.parent

    cache_root = Path(os.environ.get(cache_env_var, default_cache))
    mirrored_root = cache_root / uv_python_root.name
    mirrored_bin = mirrored_root / "bin" / venv_python.name

    if not mirrored_bin.exists():
        LOG.info("Mirroring uv Python %s -> %s for cross-node Ray actors", uv_python_root, mirrored_root)
        mirrored_root.parent.mkdir(parents=True, exist_ok=True)
        tmp = mirrored_root.with_suffix(".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(uv_python_root, tmp, symlinks=True)
        tmp.rename(mirrored_root)

    return mirrored_bin


def _download_once(dest: str, produce) -> None:
    """Produce `dest` (a file or dir) exactly once, race-safe across Ray actors.

    `produce(tmp)` fills a private pid-scoped tmp beside `dest`; we then atomically
    os.replace() it in (atomic for files and dirs on one filesystem). A competing
    actor that published first simply wins.
    """
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = f"{dest}.{os.getpid()}.tmp"
    produce(tmp)
    if os.path.exists(dest):
        # Lost the race: another actor published first. Discard our just-made copy.
        shutil.rmtree(tmp) if os.path.isdir(tmp) else os.remove(tmp)
    else:
        os.replace(tmp, dest)


def _download_to(url: str, tmp: str, gunzip: bool = False) -> None:
    with urllib.request.urlopen(url) as resp:
        stream = gzip.GzipFile(fileobj=resp) if gunzip else resp
        with open(tmp, "wb") as out:
            shutil.copyfileobj(stream, out)


def _download_ersatz_model(model_name: str = "default-multilingual") -> None:
    """Fetch ersatz weights once; its download_model() writes in-place with no lock."""
    from ersatz.utils import ERSATZ_DIR, MODELS

    m = MODELS[model_name]
    _download_once(os.path.join(ERSATZ_DIR, m["destination"]), lambda tmp: _download_to(m["source"], tmp, gunzip=True))


def _download_laser_model(model_dir: str) -> None:
    """Fetch LASER2 weights once; its downloader's cross-fs shutil.move() isn't atomic."""
    from laser_encoders.download_models import LaserModelDownloader

    base_url = LaserModelDownloader(model_dir).base_url
    for filename in ("laser2.pt", "laser2.spm", "laser2.cvocab"):
        url = f"{base_url}/{filename}"
        _download_once(os.path.join(model_dir, filename), lambda tmp, url=url: _download_to(url, tmp))


def _download_comet_model(comet_model: str) -> str:
    """Fetch a COMET checkpoint once and return its path; its non-HF URL fallback isn't race-safe."""
    from comet import download_model

    cache_root = os.environ.get("LONGMT_COMET_CACHE", "/opt/Gym/.cache/longmt-comet")
    dest = os.path.join(cache_root, comet_model.replace("/", "__"))
    _download_once(dest, lambda tmp: download_model(comet_model, saving_directory=tmp))
    # Files are present now; resolve the checkpoint path locally, no network.
    return download_model(comet_model, saving_directory=dest, local_files_only=True)


def _build_segale_actor_class(actors_per_gpu: int = 1, use_extra_gpu: bool = False):
    """Build the _SegaleActor @ray.remote class. Must be called after Ray.init().

    Built lazily so importing this module does not require Ray to be
    initialised (mirrors _build_comet_actor_class in wmt_translation/app.py).

    actors_per_gpu controls how many actors share one physical GPU.

    use_extra_gpu selects the Ray resource mode:
      False (default): actors claim fractional num_gpus so Ray manages
        CUDA_VISIBLE_DEVICES. Use this when the gym runs its own Ray cluster
        with dedicated GPU nodes (HTTP-separated from vLLM).
      True: actors claim the custom extra_gpu resource (num_gpus=0) and manage
        GPU visibility themselves via RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES.
        Use this when the gym joins the vLLM Ray cluster and a separate node has
        been registered with `ray start --resources='{"extra_gpu": N}'`.
    """
    mirrored_bin = _mirror_python(
        "LONGMT_EVAL_PY_CACHE",
        "/opt/Gym/.cache/longmt-python",
    )

    venv_dir = Path(sys.executable).parent.parent
    site_packages = venv_dir / "lib" / "python3.12" / "site-packages"

    env_vars: Dict[str, str] = {
        "PYTHONPATH": f"{site_packages}:{os.environ.get('PYTHONPATH', '')}",
    }
    if use_extra_gpu:
        # Ray thinks this node has 0 GPUs; preserve physical CUDA_VISIBLE_DEVICES
        # so the actor can still access its assigned GPU directly.
        env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    # HF_HOME: root HuggingFace cache dir; hub cache derives from it.
    # HF_HUB_OFFLINE: when 1, blocks all HF Hub network calls.
    # HF_HUB_CACHE: overrides just the hub model-cache location.
    # TRANSFORMERS_CACHE: legacy transformers cache dir (deprecated; forwarded for compat).
    # LASER_HOME: dir for LASER2 weights
    # ERSATZ: dir for ersatz segmenter weights
    for key in ("HF_HOME", "HF_HUB_OFFLINE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        if os.environ.get(key):
            env_vars[key] = os.environ[key]
    env_vars["LASER_HOME"] = os.environ.get("LASER_HOME", "/opt/Gym/.cache/longmt-laser")
    env_vars["ERSATZ"] = os.environ.get("ERSATZ", "/opt/Gym/.cache/longmt-ersatz")

    gpu_fraction = 1 / actors_per_gpu
    if use_extra_gpu:
        ray_kwargs = dict(num_gpus=0, resources={"extra_gpu": gpu_fraction})
    else:
        ray_kwargs = dict(num_gpus=gpu_fraction)

    class _SegaleActorImpl:  # pragma: no cover – needs live Ray cluster + CUDA
        def __init__(
            self,
            gpu_idx: int,
            comet_model: str,
            comet_batch_size: int,
            embed_batch_size: int,
        ):
            import torch
            from comet import load_from_checkpoint
            from ersatz.split import EvalModel as ErsatzModel
            from ersatz.utils import get_model_path
            from laser_encoders import LaserEncoderPipeline

            assert torch.cuda.is_available(), "SegaleActor requires CUDA."
            n = torch.cuda.device_count()
            self._gpu_idx = gpu_idx
            self._device = f"cuda:{gpu_idx % n}"
            self._lightning_devices = [gpu_idx % n]
            self._embed_batch_size = embed_batch_size
            self._comet_batch_size = comet_batch_size
            LOG.info(
                "SegaleActor[%d] placement: device_count=%d CUDA_VISIBLE_DEVICES=%r device=%s",
                gpu_idx,
                n,
                os.environ.get("CUDA_VISIBLE_DEVICES"),
                self._device,
            )

            laser_home = os.environ.get("LASER_HOME")
            LOG.info("SegaleActor[%d]: loading LASER2 from %s", gpu_idx, laser_home)
            _download_laser_model(laser_home)
            self._laser = LaserEncoderPipeline(laser="laser2", model_dir=laser_home)

            LOG.info("SegaleActor[%d]: loading COMETKiwi %s on %s", gpu_idx, comet_model, self._device)
            ckpt = _download_comet_model(comet_model)
            self._comet = load_from_checkpoint(ckpt)
            self._comet.to(self._device).eval()

            LOG.info("SegaleActor[%d]: loading ersatz segmenter", gpu_idx)
            from ersatz.candidates import MultilingualPunctuation

            _download_ersatz_model("default-multilingual")
            self._ersatz = ErsatzModel(get_model_path("default-multilingual"))
            self._ersatz.device = torch.device("cpu")
            self._ersatz_candidates = MultilingualPunctuation()
            LOG.info("SegaleActor[%d]: ready", gpu_idx)

            # langdetect: seed once for determinism; map our locale codes to ISO 639-1.
            # ar_AE is intentionally absent — our datasets use ar_EG / ar_SA.
            try:
                from langdetect import DetectorFactory

                DetectorFactory.seed = 0
            except ImportError:
                pass
            self._lang_map: Dict[str, str] = {
                "ar_EG": "ar",
                "ar_SA": "ar",
                "bg_BG": "bg",
                "bn_IN": "bn",
                "ca_ES": "ca",
                "cs_CZ": "cs",
                "da_DK": "da",
                "de_DE": "de",
                "el_GR": "el",
                "es_MX": "es",
                "et_EE": "et",
                "fa_IR": "fa",
                "fi_FI": "fi",
                "fil_PH": "tl",
                "fr_CA": "fr",
                "fr_FR": "fr",
                "gu_IN": "gu",
                "he_IL": "he",
                "hi_IN": "hi",
                "hr_HR": "hr",
                "hu_HU": "hu",
                "id_ID": "id",
                "it_IT": "it",
                "ja_JP": "ja",
                "kn_IN": "kn",
                "ko_KR": "ko",
                "lt_LT": "lt",
                "lv_LV": "lv",
                "ml_IN": "ml",
                "mr_IN": "mr",
                "nl_NL": "nl",
                "no_NO": "no",
                "pa_IN": "pa",
                "pl_PL": "pl",
                "pt_BR": "pt",
                "pt_PT": "pt",
                "ro_RO": "ro",
                "ru_RU": "ru",
                "sk_SK": "sk",
                "sl_SI": "sl",
                "sv_SE": "sv",
                "sw_KE": "sw",
                "sw_TZ": "sw",
                "ta_IN": "ta",
                "te_IN": "te",
                "th_TH": "th",
                "tr_TR": "tr",
                "uk_UA": "uk",
                "ur_PK": "ur",
                "vi_VN": "vi",
                "zh_CN": "zh-cn",
                "zh_TW": "zh-tw",
            }

        def ping(self) -> bool:
            return True

        def score(self, source_text: str, mt_text: str, target_language: str) -> Dict:
            """Run the full 3-phase SEGALE pipeline for one document pair.

            Returns a dict with comet_qe, lang_fidelity, total_seg,
            misaligned_seg, and error (None on success).
            """
            import tempfile

            import numpy as np
            import segale_align as sa

            # Module globals must be set before any segale_align function call.
            sa.VERBOSE = 0
            sa.SPACY = "ersatz"
            sa.STOP_JUMP = 0.15
            sa.COST_MIN = 0.30
            sa.COST_MAX = 0.30

            lang_fidelity = self._lang_fidelity(mt_text, target_language)
            empty = {
                "comet_qe": 0.0,
                "lang_fidelity": lang_fidelity,
                "total_seg": 0,
                "misaligned_seg": 0,
                "error": None,
            }

            import time as _time

            _t0 = _time.perf_counter()

            # Phase 1: segment
            src_sents = self._segment_ersatz(source_text)
            mt_sents = self._segment_ersatz(mt_text)
            if not src_sents or not mt_sents:
                return empty
            _t1 = _time.perf_counter()
            LOG.info(
                "SEGALE timing [%d]: ersatz segment %.1fs  src=%d mt=%d sents",
                self._gpu_idx,
                _t1 - _t0,
                len(src_sents),
                len(mt_sents),
            )

            # Phase 1: overlaps via stub encoder (CPU-only pass)
            class _Stub:
                def encode_sentences(self, sentences):
                    return np.empty(1, dtype=np.float32)

            stub = _Stub()
            src_overlaps, _ = sa.generate_overlap_and_embedding(src_sents, stub, None, max_size=8)
            mt_overlaps, _ = sa.generate_overlap_and_embedding(mt_sents, stub, None, max_size=8)
            _t2 = _time.perf_counter()
            LOG.info(
                "SEGALE timing [%d]: overlap gen %.1fs  src=%d mt=%d overlaps",
                self._gpu_idx,
                _t2 - _t1,
                len(src_overlaps),
                len(mt_overlaps),
            )

            # Phase 1: LASER2 encode (length-sorted for GPU efficiency)
            all_overlaps = src_overlaps + mt_overlaps
            sort_idx = np.argsort([len(s) for s in all_overlaps])
            restore_idx = np.empty(len(sort_idx), dtype=np.int64)
            restore_idx[sort_idx] = np.arange(len(sort_idx))

            sorted_embeds: List = []
            for i in range(0, len(sort_idx), self._embed_batch_size):
                batch = [all_overlaps[j] for j in sort_idx[i : i + self._embed_batch_size]]
                sorted_embeds.extend(self._laser.encode_sentences(batch))

            all_embeds = np.array(sorted_embeds)[restore_idx]
            src_embed = all_embeds[: len(src_overlaps)]
            mt_embed = all_embeds[len(src_overlaps) :]
            _t3 = _time.perf_counter()
            LOG.info(
                "SEGALE timing [%d]: LASER2 encode %.1fs  %d overlaps  device=%s",
                self._gpu_idx,
                _t3 - _t2,
                len(all_overlaps),
                getattr(getattr(self._laser, "encoder", None), "use_cuda", "?"),
            )

            # Phase 2: vecalign alignment
            with tempfile.TemporaryDirectory() as tmpdir:
                alignments = sa.run_vecalign_explore(
                    "\n".join(src_sents),
                    "\n".join(mt_sents),
                    "\n".join(src_overlaps),
                    "\n".join(mt_overlaps),
                    src_embed,
                    mt_embed,
                    "doc",
                    tmpdir,
                    max_size=8,
                )
            _t4 = _time.perf_counter()
            LOG.info(
                "SEGALE timing [%d]: vecalign %.1fs  %d alignments",
                self._gpu_idx,
                _t4 - _t3,
                len(alignments) if alignments else 0,
            )

            if not alignments:
                return empty

            spans = [
                {
                    "src": " ".join(src_sents[i] for i in si) if si else "",
                    "tgt": " ".join(mt_sents[i] for i in ti) if ti else "",
                }
                for si, ti in alignments
            ]

            # Phase 3: COMETKiwi scoring
            comet_spans = self._comet_score(spans)
            _t5 = _time.perf_counter()
            LOG.info(
                "SEGALE timing [%d]: COMETKiwi %.1fs  %d spans  total=%.1fs",
                self._gpu_idx,
                _t5 - _t4,
                len(spans),
                _t5 - _t0,
            )
            all_comet = [s["comet_qe"] for s in comet_spans]

            return {
                "comet_qe": sum(all_comet) / len(all_comet) if all_comet else 0.0,
                "lang_fidelity": lang_fidelity,
                "total_seg": len(comet_spans),
                "misaligned_seg": sum(1 for s in comet_spans if s["deleted"] or s["hallucinated"]),
                "spans": comet_spans,
                "error": None,
            }

        def _comet_score(self, spans: List[Dict]) -> List[Dict]:
            result: List[Dict] = []
            comet_data: List[Dict] = []
            comet_indices: List[int] = []
            for s in spans:
                has_src = bool(s["src"])
                has_tgt = bool(s["tgt"])
                if not has_src and not has_tgt:
                    continue  # drop both-empty spans entirely
                elif has_src and has_tgt:
                    comet_indices.append(len(result))
                    comet_data.append({"src": s["src"], "mt": s["tgt"]})
                    result.append(
                        {"src": s["src"], "tgt": s["tgt"], "comet_qe": 0.0, "hallucinated": False, "deleted": False}
                    )
                elif not has_tgt:
                    result.append(
                        {"src": s["src"], "tgt": s["tgt"], "comet_qe": 0.0, "hallucinated": False, "deleted": True}
                    )
                else:
                    result.append(
                        {"src": s["src"], "tgt": s["tgt"], "comet_qe": 0.0, "hallucinated": True, "deleted": False}
                    )
            if comet_data:
                out = self._comet.predict(
                    comet_data,
                    batch_size=self._comet_batch_size,
                    devices=self._lightning_devices,
                )
                for idx, score in zip(comet_indices, out.scores):
                    result[idx]["comet_qe"] = float(score)
            return result

        def _segment_ersatz(self, text: str) -> List[str]:
            sentences = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                for output in self._ersatz.parallel_evaluation(
                    line, batch_size=16, candidates=self._ersatz_candidates
                ):
                    if output is not None:
                        sentences.extend(s.strip() for s in output.splitlines() if s.strip())
            return sentences

        def _lang_fidelity(self, text: str, target_language: str) -> Optional[float]:
            if not text or len(text) < 50:
                return 1.0
            expected = self._lang_map.get(target_language)
            if expected is None:
                return None
            try:
                from langdetect import detect
            except ImportError:
                return 1.0
            chunks = [text[i : i + 500] for i in range(0, len(text), 500)]
            if len(chunks) > 1 and len(chunks[-1]) < 100:
                chunks = chunks[:-1]
            correct = detected = 0
            for chunk in chunks:
                try:
                    correct += detect(chunk) == expected
                    detected += 1
                except Exception:
                    pass
            return correct / detected if detected else 1.0

    _SegaleActor = ray.remote(
        **ray_kwargs,
        runtime_env={"py_executable": str(mirrored_bin), "env_vars": env_vars},
    )(_SegaleActorImpl)
    return _SegaleActor
