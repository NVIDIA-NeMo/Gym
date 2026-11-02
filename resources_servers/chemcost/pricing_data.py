# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The pinned ChemCost release on Hugging Face (YoungerWu/ChemCost, CC BY-SA 4.0): records and pricing database.

The labels in ``data/chemcost.jsonl`` are computed from ``pricing/pricing_db.sqlite`` of the same revision, so both
files come from one pinned revision and are checked by SHA-256 before use.
"""

import hashlib
import os
import shutil
from pathlib import Path
from typing import Optional


HF_REPO_ID = "YoungerWu/ChemCost"
HF_REVISION = "8af73ec996cea894dfd2f46ea1461ca71e0e6fef"  # pragma: allowlist secret
RECORDS_FILE = "data/chemcost.jsonl"
PRICING_DB_FILE = "pricing/pricing_db.sqlite"
SHA256 = {
    RECORDS_FILE: "cddd1ca311924402af7ba4471d90bd331bbcbe99f4e3cf78d7141dba9a281ecc",  # pragma: allowlist secret
    PRICING_DB_FILE: "72ee21f27f91f113902353f1451166d3e96b92779ffeb3c16abb7013ee8291f3",  # pragma: allowlist secret
}
DEFAULT_DATA_DIR = Path(__file__).absolute().parent / ".data"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_release_file(filename: str, target: Optional[Path] = None) -> Path:
    """Return a verified local copy of one release file, downloading the pinned revision when it is missing."""
    target = target or DEFAULT_DATA_DIR / Path(filename).name
    if not target.exists():
        from huggingface_hub import hf_hub_download

        cached = Path(hf_hub_download(HF_REPO_ID, filename, repo_type="dataset", revision=HF_REVISION))
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.{os.getpid()}.partial")  # concurrent workers
        shutil.copyfile(cached, partial)
        partial.replace(target)
    actual = _sha256(target)
    if actual != SHA256[filename]:
        raise ValueError(
            f"{target} has SHA-256 {actual}, expected {SHA256[filename]} ({HF_REPO_ID}@{HF_REVISION}:{filename}); "
            "delete it to download the pinned file again"
        )
    return target


def ensure_pricing_db(target: Optional[Path] = None) -> Path:
    return ensure_release_file(PRICING_DB_FILE, target)
