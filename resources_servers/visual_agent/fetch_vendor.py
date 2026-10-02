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
"""Download the pinned JavaScript libraries that tasks may use offline.

Rollouts run without CDN access at grading time, so 3D and chart tasks get these copied into
`/workspace/output/vendor/`. Both libraries are MIT-licensed; they are downloaded from the npm
registry (integrity-checked) instead of being committed.

    python resources_servers/visual_agent/fetch_vendor.py
"""

import base64
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path


VENDOR_DIR = Path(__file__).parent / "data" / "vendor"

# package -> (version, {path inside the npm tarball: path under data/vendor})
PACKAGES = {
    "three": (
        "0.186.0",
        {
            "package/build/three.module.js": "three/three.module.js",
            "package/build/three.core.js": "three/three.core.js",
            "package/examples/jsm/controls/OrbitControls.js": "three/addons/controls/OrbitControls.js",
            "package/examples/jsm/geometries/RoundedBoxGeometry.js": "three/addons/geometries/RoundedBoxGeometry.js",
            "package/examples/jsm/math/SimplexNoise.js": "three/addons/math/SimplexNoise.js",
            "package/examples/jsm/math/ImprovedNoise.js": "three/addons/math/ImprovedNoise.js",
            "package/examples/jsm/environments/RoomEnvironment.js": "three/addons/environments/RoomEnvironment.js",
            "package/LICENSE": "three/LICENSE",
        },
    ),
    "chart.js": (
        "4.5.1",
        {
            "package/dist/chart.umd.min.js": "chartjs/chart.umd.min.js",
            "package/LICENSE.md": "chartjs/LICENSE.md",
        },
    ),
}


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def fetch_vendor(vendor_dir: Path = VENDOR_DIR) -> Path:
    for package, (version, files) in PACKAGES.items():
        if all((vendor_dir / dest).exists() for dest in files.values()):
            continue
        meta = json.loads(_fetch(f"https://registry.npmjs.org/{package}/{version}"))
        tarball = _fetch(meta["dist"]["tarball"])
        algorithm, _, expected = meta["dist"]["integrity"].partition("-")
        actual = base64.b64encode(hashlib.new(algorithm, tarball).digest()).decode()
        if actual != expected:
            raise RuntimeError(f"Integrity check failed for {package}@{version}")
        with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as archive:
            for member_name, dest in files.items():
                member = archive.extractfile(member_name)
                if member is None:
                    raise FileNotFoundError(f"{member_name} missing from {package}@{version}")
                out = vendor_dir / dest
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(member.read())
    return vendor_dir


if __name__ == "__main__":
    print(fetch_vendor())
