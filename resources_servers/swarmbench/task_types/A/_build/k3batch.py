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
"""Generic build-time K3 batch runner (self-hosted pool only). Input JSONL: {"id", "messages", ["max_tokens"]};
output JSONL (appended, resumable): {"id", "content", "usage", "s"}.
usage: python3 -I k3batch.py IN.jsonl OUT.jsonl [--urls URL1,URL2 | --endpoints-dir DIR] [--workers 10]
With --endpoints-dir (the serve jobs write kimi-<job>.txt = "JOB IP TS" there), the live replicas are
re-probed every 2 min, so replicas that end or renew are followed."""
import argparse
import glob
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("--urls", default="")
ap.add_argument("--endpoints-dir", default="")
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--temperature", type=float, default=0.0)
a = ap.parse_args()
lock = threading.Lock()
_live = {"t": 0, "urls": [], "i": 0}


def probe():
    if a.urls:
        return a.urls.split(",")
    ok = []
    for f in glob.glob(a.endpoints_dir + "/kimi-*.txt"):
        parts = open(f).read().split()
        if len(parts) < 2:
            continue
        u = f"http://{parts[1]}:8000/v1"
        try:
            with urllib.request.urlopen(u + "/models", timeout=5) as r:
                if "Kimi-K3" in r.read().decode():
                    ok.append(u)
        except Exception:
            pass
    return sorted(set(ok))


def next_url():
    with lock:
        if time.time() - _live["t"] > 120 or not _live["urls"]:
            _live["urls"], _live["t"] = probe(), time.time()
        if not _live["urls"]:
            return None
        _live["i"] += 1
        return _live["urls"][_live["i"] % len(_live["urls"])]
try:
    done = {json.loads(l)["id"] for l in open(a.out)}
except OSError:
    done = set()


def call(body):
    for i in range(8):
        url = next_url()
        try:
            if url is None:
                raise RuntimeError("no live K3 replica")
            req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.load(r)
        except Exception as e:
            print("retry", i, url, e, flush=True)
            _live["t"] = 0
            time.sleep(30 * (i + 1))
    return None


def work(rec):
    t0 = time.time()
    r = call({"model": "policy_model", "messages": rec["messages"],
                          "max_tokens": rec.get("max_tokens", 16000), "temperature": a.temperature})
    if r is None:
        print("FAILED", rec["id"], flush=True)
        return
    out = {"id": rec["id"], "content": r["choices"][0]["message"].get("content") or "", "usage": r.get("usage"),
           "s": round(time.time() - t0, 1)}
    with lock:
        with open(a.out, "a") as f:
            f.write(json.dumps(out) + "\n")
    print(rec["id"], out["s"], flush=True)


recs = [json.loads(l) for l in open(a.inp)]
with ThreadPoolExecutor(a.workers) as ex:
    list(ex.map(work, [r for r in recs if r["id"] not in done]))
print("done", flush=True)
