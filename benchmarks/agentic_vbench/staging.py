# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-task adjustments applied to the pinned tasks before they are written into the Harbor entry.

Three stages, each copying a task only when it changes it, so an untouched task stays the upstream folder
byte for byte:

- ``stage_assembly_judge_guard``: the pinned assembly judges score a non-object ``solution.json`` 0 instead of
  crashing (verified by digest, so only the exact pinned judges are patched);
- ``stage_judge``: a repurpose task's LLM judge runs through the NVIDIA Inference Hub with one key that reaches
  only the verifier, and a judge-service outage exits 75 instead of scoring 0;
- ``stage_materials``: a repair task's golden-asset download retries longer within a bound, and a download that
  still fails exits 75 instead of leaving no reward.

Exit 75 (``EX_TEMPFAIL``) without a reward is what the ``harbor`` environment server masks as
``harbor:verifier_unavailable``, so infrastructure failures at verification time never count as 0. Both stages
also declare ``/workspace/output`` as the step's artifact, so with the server's ``separate_verifier_for_artifacts``
the judged and repair verifiers grade in a fresh sandbox of the task image, where the agent can neither read the
judge key nor fake the exit code.
"""

import hashlib
import json
import shutil
import tomllib
from pathlib import Path


JUDGE_PROTOCOLS = {
    # The upstream judge pair (Claude Opus 4.7, Gemini 3.1 Pro) on the Hub routes that serve them.
    "nvinference-upstream": {
        "AVB_JUDGE_BASE_URL": "https://inference-api.nvidia.com",
        "AVB_JUDGE_CLAUDE_MODEL": "azure/anthropic/claude-opus-4-7",
        "AVB_JUDGE_GEMINI_MODEL": "gcp/google/gemini-3.1-pro-preview",
        "AVB_JUDGE_GEMINI_THINKING": "low",
    },
    # A cheaper pair (Claude Opus 4.8, Gemini 3.5 Flash with bounded thinking). Route entitlements differ per
    # Hub key: hub_judge.py checks each route with one request before judging, and a route the key cannot
    # use makes the verification unavailable (masked) instead of scoring 0.
    "nvinference-hybrid": {
        "AVB_JUDGE_BASE_URL": "https://inference-api.nvidia.com",
        "AVB_JUDGE_CLAUDE_MODEL": "us/azure/anthropic/eccn-claude-opus-4-8",
        "AVB_JUDGE_GEMINI_MODEL": "us/gcp/google/eccn-gemini-3.5-flash",
        "AVB_JUDGE_GEMINI_THINKING": "low",
    },
}
DEFAULT_JUDGE_PROTOCOL = "nvinference-upstream"
# The upstream tasks' own judge credentials: the templates stage_judge removes from the agent's environment.
UPSTREAM_JUDGE_ENV = {"ANTHROPIC_API_KEY", "GEMINI_API_KEY"}
# What the judged and repair verifiers read of the agent's work; declared as the step's artifact so the server can
# verify them in a fresh sandbox (`separate_verifier_for_artifacts`). The repair verifiers also read the task's
# materials, which the setup step copies from the image's /baked into the agent's sandbox, so a fresh verifier
# sandbox needs them copied too; the judged verifiers read their source clip from /baked directly.
OUTPUT_ARTIFACT = "/workspace/output"
MATERIALS_ARTIFACT = "/workspace/materials"
# Informational scan in the repurpose verifier; inaccessible /proc entries in rootless containers must not
# abort the verifier under pipefail.
TYPING_DIAGNOSTIC = "find / -name 'typing_extensions*' 2>/dev/null | head -20"
HUB_JUDGE = Path(__file__).with_name("hub_judge.py")

_ASSEMBLY_FAMILY = "agentic_vbench_assembly"
_ASSEMBLY_MANIFEST_PARSER = """    segments = sol.get("segments")
    if not isinstance(segments, list):
        return _zero("solution.json: segments not a list")

    pred = [_normalize(seg.get("source", "")) for seg in segments]
"""
_ASSEMBLY_HARDENED_PARSER = """    if not isinstance(sol, dict):
        return _zero("solution.json: expected top-level object")

    segments = sol.get("segments")
    if not isinstance(segments, list):
        return _zero("solution.json: segments not a list")

    for i, segment in enumerate(segments):
        if not isinstance(segment, dict):
            return _zero(f"solution.json: segment {i} is not an object")

    pred = [_normalize(seg.get("source", "")) for seg in segments]
"""
# (source judge digest, hardened judge digest) of every pinned assembly task.
_ASSEMBLY_JUDGE_SHA256 = {
    "agentic-vbench-assembly-task1": (
        "61a3ec3270a5ecf9ea394c6f7216d08a9af1d9ef794bf6ac297752f30fee168e",  # pragma: allowlist secret
        "5bc591c8439c11c262dc4073d1db3750f15603f6d6dbc7d6551ca045b3d21fad",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task2": (
        "7c70dc9c2dc357993f91e80b51a74d153c9bf8dc5274ca826ad9eaf2ae0f9ca3",  # pragma: allowlist secret
        "161d8e04440b8add9280fa2649e4d1470f49f4fcd68697b6cd32eae7aa80868b",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task3": (
        "0e76e6774046ea9448c3d5a906d7c04f50cf05706739f2afb82c77d887639548",  # pragma: allowlist secret
        "f7a2030f194ebc0befb1fd349efb6d39d78f260abc88672923d91f161a0ec453",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task4": (
        "73da97f7929d94831141a23954b910ae03f5d0de93ac12cf27df3a9a1c713ba5",  # pragma: allowlist secret
        "02f5894208585abf30fde48c68f5ea9ccaf53182c934eadc1520f8ab97339646",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task5": (
        "129b6d6ceb8ed68e2f1c89246b99ed18db5908c80bffa7425c03dc2dc08ac094",  # pragma: allowlist secret
        "61e63f8d5401fd496f13757221dec0f550d8f1d9bf93aa30d4a410e4e1f7e2f9",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task6": (
        "609f102e08adf0eccfbd0bb4973a20ff6b88b402f7a61a906a45eb3a55dc3dc4",  # pragma: allowlist secret
        "6d449633c37da9fd10a8f4cf046623c21f7f15c9da372c01b127d372d31a6156",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task7": (
        "7dd6d792007f581f2b9b843d5ca1958560884b253edd7a8c34e73db1376191d6",  # pragma: allowlist secret
        "08978a086806d75c1880979160227cc753965eeb6ced6dfc1028d5ff45c8eb5e",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task8": (
        "52b01834e4c342888e0238e399507572ed82e65b6082eb27ddaedded7c985e75",  # pragma: allowlist secret
        "2dd1339341fe5a4ca6052aeb4edb58c740019a4a2610356eeadef56acb5373a4",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task9": (
        "49b585fd2413798d6b484b1123c5ff476e791715dc99fa42f21e642637cf492f",  # pragma: allowlist secret
        "30e2816a2fd73e4b0a165943655cd32dea316f830c92a16cdb05e9c3feeb1d15",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task10": (
        "9ce0f3da6ae606f775505e33e8380539e3f27de185bc137d6d401d4747ee40c4",  # pragma: allowlist secret
        "d7c7124dc304f8a6e8e394b463838b13b1d309615518b6d2ec3d89388e818045",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task11": (
        "2de466f807ab097e1ca754908ebc36ffe9c1ba716a540455287568322de68c06",  # pragma: allowlist secret
        "955ebfc11fae35bff96f6a9cf2d281bc03f43c2630aa2d3919314bb8ae8164ba",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task12": (
        "c2648737e1de760e743fa29115a0c241ef18c3f0a6132eea70161ab96497ddcb",  # pragma: allowlist secret
        "fe363ddc8b720c512833064d032ab5e8a3df691efb4d726a98fa4e5475417a38",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task13": (
        "ef488cd4f71e0947cf19ecab470ff6693e15de26a5cdd07e8e203cb75e84866a",  # pragma: allowlist secret
        "821ee8de3bd3be80c7ece58b0c1ecd9d1d3845572bfb021014b2b3955074d379",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task14": (
        "20660911630e38d25bad3c8d2d9c9148ab9781a3fa4d3d6c5a89c72ec64143ea",  # pragma: allowlist secret
        "0d8538b9696f1bc1089d8415c0a8a35e5efef8fa34fc289e46a8c92c7305e305",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task15": (
        "26451dde4b9328495aad4610f0f48282b51d4660695219645118cc37b0468b6e",  # pragma: allowlist secret
        "bab7f16a3c74117ab03c7e5fd8dcae473454768021cbb9ceb31a1f7f37085a1f",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task16": (
        "a7293e8e95cc59c27cdbafe9802983c21d961f868fb426c68f30a99d7e628689",  # pragma: allowlist secret
        "670f636b37cf077fd5e58054021cad1cd98d34fabd62ef844687199fdb4dc9c0",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task17": (
        "7ad56e110ec984d69a9d2afbaf2d9ed93eda71d1a621ae7c350e1e845858922d",  # pragma: allowlist secret
        "c5d6dd370825d997bce0138f08760ba14e543e60d70f4503d59391ada669fc99",  # pragma: allowlist secret
    ),
    "agentic-vbench-assembly-task18": (
        "f2dea83ca090e44108be5ce52817b6bbfd4266a0ecf909b30b7f7bcbc2954682",  # pragma: allowlist secret
        "49724c26534af8519849e42e5411ebc4793257ce6ea68bbdbd1d430812d807cc",  # pragma: allowlist secret
    ),
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    return _digest(path.read_bytes())


def _harden_assembly_judge(source: bytes, task_id: str) -> bytes:
    expected = _ASSEMBLY_JUDGE_SHA256.get(task_id)
    if expected is None:
        raise ValueError(f"Unexpected pinned assembly task: {task_id}")
    source_sha256, staged_sha256 = expected
    if _digest(source) != source_sha256:
        raise ValueError(f"Pinned assembly judge has an unexpected source digest: {task_id}")
    text = source.decode()
    if text.count(_ASSEMBLY_MANIFEST_PARSER) != 1:
        raise ValueError(f"Pinned assembly judge has an unexpected manifest parser: {task_id}")
    hardened = text.replace(_ASSEMBLY_MANIFEST_PARSER, _ASSEMBLY_HARDENED_PARSER).encode()
    if _digest(hardened) != staged_sha256:
        raise ValueError(f"Assembly judge hardening produced an unexpected digest: {task_id}")
    return hardened


def stage_assembly_judge_guard(task_path: Path, *, staging: Path, output: Path) -> Path:
    """Make the pinned assembly judge reject a non-object manifest with reward zero.

    The upstream judge calls ``sol.get`` right after ``json.loads``, so valid JSON of the wrong top-level shape
    (a bare list) raises ``AttributeError`` instead of scoring 0 like every other malformed manifest. Only the
    exact pinned assembly judges are patched; valid-object scoring is unchanged and the source stays clean.
    """
    if task_path.parent.name != _ASSEMBLY_FAMILY:
        return task_path
    task_id = task_path.name
    source_judge = task_path / "steps/solve/tests/judge.py"
    hardened = _harden_assembly_judge(source_judge.read_bytes(), task_id)
    staged = staging / task_path.name
    expected_staged_sha256 = _ASSEMBLY_JUDGE_SHA256[task_id][1]
    if staged.exists():
        staged_judge = staged / "steps/solve/tests/judge.py"
        if not staged_judge.is_file() or _file_digest(staged_judge) != expected_staged_sha256:
            raise ValueError(f"Existing staged assembly judge has an unexpected digest: {task_id}")
        return staged
    shutil.copytree(task_path, staged)
    judge = staged / "steps/solve/tests/judge.py"
    judge.write_bytes(hardened)
    _write_record(
        output / "assembly_judge_guard.json",
        {
            "source_task": str(task_path),
            "reason": "non-object solution manifests score zero instead of crashing the verifier",
            "original_judge_sha256": _file_digest(source_judge),
            "staged_judge_sha256": _file_digest(judge),
        },
    )
    return staged


def stage_judge(task_path: Path, *, staging: Path, output: Path, protocol: str) -> Path:
    """Copy an LLM-judged task with its judge routed through the NVIDIA Inference Hub.

    Tasks without the upstream judge credentials are returned unchanged. The Hub key is a verifier-only template
    (``${NVINFERENCE_API_KEY}`` in ``[steps.verifier.env]``), never part of the agent's environment, and the
    step declares ``/workspace/output`` as its artifact so the verifier can run separately from the agent's sandbox.
    """
    if protocol not in JUDGE_PROTOCOLS:
        raise ValueError(f"Unknown judge protocol {protocol!r}; one of {sorted(JUDGE_PROTOCOLS)}")
    text = (task_path / "task.toml").read_text()
    task = tomllib.loads(text)
    if set(task.get("environment", {}).get("env", {})) != UPSTREAM_JUDGE_ENV:
        return task_path
    if len(task["steps"]) != 1 or task["steps"][0].get("verifier", {}).get("env"):
        raise ValueError(f"Unexpected judged task layout: {task_path}")
    tests_relative = Path("steps") / task["steps"][0]["name"] / "tests"
    if not (task_path / tests_relative / "judge.py").is_file():
        raise ValueError(f"Judged task without tests/judge.py: {task_path}")
    staged = staging / task_path.name
    shutil.copytree(task_path, staged)
    start, end = text.index("[environment.env]"), text.index("[[steps]]")
    settings = JUDGE_PROTOCOLS[protocol]
    text = text[:start] + text[end:] + '\n[steps.verifier.env]\nNVINFERENCE_API_KEY = "${NVINFERENCE_API_KEY}"\n'
    text += "".join(f'{key} = "{value}"\n' for key, value in settings.items())
    (staged / "task.toml").write_text(_declare_output_artifact(text))
    tests = staged / tests_relative
    verifier = tests / "test.sh"
    verifier.write_text(verifier.read_text().replace(TYPING_DIAGNOSTIC, TYPING_DIAGNOSTIC + " || true"))
    (tests / "judge.py").rename(tests / "avb-original-judge.py")
    shutil.copyfile(HUB_JUDGE, tests / "judge.py")
    files = {
        "task.toml": Path("task.toml"),
        "test.sh": tests_relative / "test.sh",
        "judge.py": tests_relative / "judge.py",
        "avb-original-judge.py": tests_relative / "avb-original-judge.py",
    }
    _write_record(
        output / "judge_protocol.json",
        {
            "protocol": protocol,
            "settings": settings,
            "source_task": str(task_path),
            "original_sha256": {
                name: _file_digest(task_path / relative)
                for name, relative in files.items()
                if (task_path / relative).is_file()
            },
            "staged_sha256": {
                name: _file_digest(staged / relative)
                for name, relative in files.items()
                if (staged / relative).is_file()
            },
        },
    )
    return staged


# The repair verifiers fetch their golden assets from Hugging Face at verification time with
# `curl --retry 5 --retry-delay 3`. Under tens of concurrent episodes from one egress address Hugging Face answers
# bursts with 429/500 and the six attempts finish inside ~20 s, leaving no reward. The patched fetch retries longer,
# bounded below the 1800 s verifier budget so a stalled download ends as "could not grade", not as the verifier
# timeout.
MATERIALS_RETRY = "--retry 5 --retry-delay 3"
MATERIALS_RETRY_PATCHED = "--retry 12 --retry-delay 20 --retry-all-errors --max-time 300 --retry-max-time 1200"
MATERIALS_FETCH = '"$MATERIALS_URL"'
# When the download still fails, the verifier says why and exits 75 instead of exiting without a reward.
MATERIALS_FETCH_GUARD = (
    ' || { rc=$?; echo "verifier unavailable: golden assets download from MATERIALS_URL failed (curl exit $rc)" >&2;'
    " exit 75; }"
)


def stage_materials(task_path: Path, *, staging: Path, output: Path) -> Path:
    """Copy a task whose verifier downloads MATERIALS_URL: longer, bounded curl retries, a failed download exits 75,
    and ``/workspace/output`` and ``/workspace/materials`` become the step's artifacts. Tasks without such a download
    are returned unchanged."""
    text = (task_path / "task.toml").read_text()
    if "MATERIALS_URL" not in text:
        return task_path
    verifiers = [v for v in task_path.glob("steps/*/tests/test.sh") if MATERIALS_RETRY in v.read_text()]
    if not verifiers:
        return task_path
    staged = staging / task_path.name
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(task_path, staged)
    digests = {}
    for verifier in verifiers:
        target = staged / verifier.relative_to(task_path)
        target.write_text(_guard_materials_fetch(target.read_text().replace(MATERIALS_RETRY, MATERIALS_RETRY_PATCHED)))
        digests[str(verifier.relative_to(task_path))] = {
            "original_sha256": _file_digest(verifier),
            "staged_sha256": _file_digest(target),
        }
    (staged / "task.toml").write_text(_declare_output_artifact(text, MATERIALS_ARTIFACT))
    _write_record(
        output / "materials.json",
        {
            "source_task": str(task_path),
            "reason": "bounded longer golden-asset retries; a failed download exits 75 (verifier unavailable)",
            "retry": {"original": MATERIALS_RETRY, "staged": MATERIALS_RETRY_PATCHED},
            "artifacts": [OUTPUT_ARTIFACT, MATERIALS_ARTIFACT],
            "verifiers": digests,
            "task_toml_sha256": {
                "original": _file_digest(task_path / "task.toml"),
                "staged": _file_digest(staged / "task.toml"),
            },
        },
    )
    return staged


def _declare_output_artifact(text: str, *extra: str) -> str:
    """Add ``artifacts = ["/workspace/output", ...]`` to the task's single ``[[steps]]`` table, after its ``name``."""
    if "artifacts" in text:
        raise ValueError("Pinned task already declares artifacts")
    marker = "[[steps]]\n"
    if text.count(marker) != 1:
        raise ValueError("Expected exactly one [[steps]] table")
    lines = text.split("\n")
    name = next(i for i, line in enumerate(lines) if line.startswith("name = ") and i > lines.index("[[steps]]"))
    paths = ", ".join(f'"{path}"' for path in (OUTPUT_ARTIFACT, *extra))
    lines.insert(name + 1, f"artifacts = [{paths}]")
    return "\n".join(lines)


def _guard_materials_fetch(script: str) -> str:
    """Append the exit-75 guard to the command line that fetches ``$MATERIALS_URL``.

    The upstream fetch is one ``curl`` command that may continue over several lines; the guard goes after its last
    line, which is the first line naming ``"$MATERIALS_URL"`` that does not end in a continuation backslash.
    """
    lines = script.split("\n")
    fetch = next((i for i, line in enumerate(lines) if MATERIALS_FETCH in line), None)
    if fetch is None:
        raise ValueError('MATERIALS_URL verifier without a curl fetch of "$MATERIALS_URL"')
    end = fetch
    while lines[end].rstrip().endswith("\\"):
        end += 1
    if MATERIALS_FETCH_GUARD not in lines[end]:
        lines[end] = lines[end].rstrip() + MATERIALS_FETCH_GUARD
    return "\n".join(lines)


def _write_record(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n")
