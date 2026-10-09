# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install an isolated runtime and warm the pretrained molecular-property oracles."""

import fcntl
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
# PyTDC 0.4.1's immutable Harvard Dataverse artifact IDs and validated SHA-256s.
ORACLE_ARTIFACTS = {
    "drd2": (6413411, "ef1f00e47d5e4670a45b0a4178db3c41b2e1aa9dad7113ac9d0f58e3f9d67532"),  # pragma: allowlist secret
    "gsk3b": (6413412, "d3a20701b80e5179c88c3ad4dc3483dd7ab35c50dc055c6773a7f5b63e89b6d5"),  # pragma: allowlist secret
    "jnk3": (6413420, "cde8576fb4fa3f60b9f258ff9cf1b9ff346eb50d196d5cbbe25965efc1864889"),  # pragma: allowlist secret
}
ORACLE_CHECK = """
import math
from tdc import Oracle
for name in ('drd2', 'gsk3b', 'jnk3'):
    oracle = Oracle(name=name)
    value = float(oracle.evaluator_func('CCO'))
    if not math.isfinite(value):
        raise RuntimeError(f'Invalid {name} oracle result: {value}')
"""


def ensure_oracle_artifacts(cache: Path) -> None:
    """Validate before unpickling; never publish HTTP errors as model checkpoints."""
    oracle_dir = cache / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    for name, (file_id, expected) in ORACLE_ARTIFACTS.items():
        path = oracle_dir / f"{name}_current.pkl"
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(
                    f"Corrupt ChemCoTBench oracle checkpoint: {path}; preserve it and replace with the pinned artifact"
                )
            continue
        temporary = None
        try:
            # Dataverse rejects urllib's default User-Agent with HTTP 403.
            request = Request(
                f"https://dataverse.harvard.edu/api/access/datafile/{file_id}",
                headers={"User-Agent": "NeMo-Gym/1.0 (ChemCoTBench oracle setup)"},
            )
            with urlopen(request, timeout=120) as response:
                with NamedTemporaryFile(dir=oracle_dir, delete=False) as stream:
                    temporary = Path(stream.name)
                    shutil.copyfileobj(response, stream)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError(f"ChemCoTBench {name} oracle download checksum mismatch")
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def ensure_molopt_runtime(python_path: str | None = None, cache_dir: str | None = None) -> tuple[Path, Path]:
    cache = Path(cache_dir).expanduser().resolve() if cache_dir else ROOT / ".molopt"
    cache.mkdir(parents=True, exist_ok=True)
    # Multiple server instances may initialize against the same cache.
    with (cache / ".setup.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _ensure_runtime(python_path, cache)


def _ensure_runtime(python_path: str | None, cache: Path) -> tuple[Path, Path]:
    requirements = ROOT / "requirements-molopt.txt"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    # absolute(), not resolve(): following a venv's interpreter symlink loses the venv.
    python = Path(python_path).expanduser().absolute() if python_path else cache / "venv" / "bin" / "python"
    marker = cache / "ready.json"
    signature = {"requirements": digest, "python": str(python)}
    artifacts = [cache / "oracle" / f"{name}_current.pkl" for name in ("drd2", "gsk3b", "jnk3")]
    try:
        ready = json.loads(marker.read_text()) == signature
    except (OSError, ValueError):
        ready = False
    ensure_oracle_artifacts(cache)
    if python.exists() and ready and all(path.is_file() and path.stat().st_size for path in artifacts):
        return python, cache
    if python_path is None:
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required to install the ChemCoTBench MolOpt runtime")
        if not python.exists():
            # Packaged or moved environments can retain a broken interpreter symlink.
            # Recreate only our managed venv when its interpreter is missing.
            subprocess.run([uv, "venv", "--clear", "--python", "3.11", str(cache / "venv")], check=True, timeout=600)
        subprocess.run(
            [uv, "pip", "install", "--python", str(python), "-r", str(requirements)], check=True, timeout=600
        )
    subprocess.run([str(python), "-c", ORACLE_CHECK], cwd=cache, check=True, timeout=600)
    marker.write_text(json.dumps(signature))
    return python, cache


if __name__ == "__main__":
    python, cache = ensure_molopt_runtime()
    print(f"MolOpt Python: {python}\nOracle directory: {cache}")
