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
# Usage: python -I -B runner.py REPO_ROOT OUT_JSON [--block-legacy] < {"module": [[fn, args], ...]}
# Runs each module's cases and writes {"module": {"cases": [...]} | {"import_error": str}} to OUT_JSON.
import dataclasses
import importlib
import importlib.abc
import json
import os
import sys

API_FUNCS = {
    "money": ["exchange"],
    "store": ["lookup", "save"],
    "clock": ["elapsed_ms", "offset"],
    "events": ["publish"],
    "config": ["setting"],
}
CALLS = [0]


class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "acme.legacy" or name.startswith("acme.legacy."):
            raise ModuleNotFoundError("acme.legacy has been removed", name=name)
        return None


def canon(x):
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        out = {"__type__": type(x).__name__}
        out.update({f.name: canon(getattr(x, f.name)) for f in dataclasses.fields(x)})
        return out
    if isinstance(x, (list, tuple)):
        return [canon(v) for v in x]
    if isinstance(x, dict):
        return {str(k): canon(v) for k, v in x.items()}
    if isinstance(x, float) and x != x:
        return "NaN"
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return {"__repr__": type(x).__name__}


def counted(fn):
    def wrapper(*args, **kwargs):
        CALLS[0] += 1
        return fn(*args, **kwargs)
    return wrapper


def install_counters():
    # Patch before student modules import, so `from acme.api.x import f` also binds the counted version.
    for mod_name, names in API_FUNCS.items():
        mod = importlib.import_module("acme.api." + mod_name)
        for name in names:
            setattr(mod, name, counted(getattr(mod, name)))


def run_module(name, cases):
    from acme.api import store
    store.reset()
    try:
        mod = importlib.import_module("acme.mods." + name)
    except BaseException as e:
        return {"import_error": f"{type(e).__name__}: {e}"[:300]}
    rows = []
    for fn, args in cases:
        before = CALLS[0]
        try:
            rows.append({"ok": canon(getattr(mod, fn)(*args))})
        except BaseException as e:
            rows.append({"raises": type(e).__name__, "msg": str(e)[:200]})
        rows[-1]["calls"] = CALLS[0] - before
    return {"cases": rows}


def main():
    root, out_path = sys.argv[1], sys.argv[2]
    jobs = json.load(sys.stdin)
    sys.path.insert(0, os.path.abspath(root))
    if "--block-legacy" in sys.argv:
        sys.meta_path.insert(0, BlockLegacy())
    sys.setrecursionlimit(3000)
    install_counters()
    # Student code may print; keep it out of the result channel.
    devnull = open(os.devnull, "w")
    sys.stdout = sys.stderr = devnull
    results = {name: run_module(name, cases) for name, cases in jobs.items()}
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main()
