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
"""Build-time only: ask self-hosted Kimi K3 (vLLM, served as policy_model; no hosted inference) to write a type's task-facing text, freeze it as text/k3.json.

usage (anywhere that reaches a self-hosted K3 server): python3 k3gen.py <type_dir> --url http://HOST:8000/v1
Reads <type_dir>/text/request.json: {"user": str, "keys": [top-level keys], "placeholders": {key: ["{n}", ...]},
"lists": {key: min_len}}. K3 must answer with one JSON object. Writes text/k3.json (the object) and text/k3_raw.json
(model, prompt, raw reply, usage, time). Placeholders are checked so code can fill numbers into K3's sentences.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

MODEL = "policy_model"  # self-hosted moonshotai/Kimi-K3 (revision 9f62e4e9) on vLLM
SYSTEM = ("You write task text for an offline evaluation of software agents. Follow the facts you are given exactly; "
          "do not add rules, numbers or names that are not in the brief. Reply with exactly one JSON object and "
          "nothing else.")


def call(user, url):
    body = {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "max_tokens": 32000, "temperature": 0.6}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def flat_strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, list):
        for x in v:
            yield from flat_strings(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from flat_strings(x)


def problems(obj, req):
    out = [f"missing key {k}" for k in req.get("keys", []) if k not in obj]
    for k, phs in req.get("placeholders", {}).items():
        if k in obj:
            for s in flat_strings(obj[k]) if isinstance(obj[k], list) else [json.dumps(obj[k])]:
                miss = [p for p in phs if p not in s]
                if miss and not isinstance(obj[k], list):
                    out.append(f"{k} lacks {miss}")
            if isinstance(obj[k], list):
                bad = [s for s in obj[k] if isinstance(s, str) and any(p not in s for p in phs)]
                if bad:
                    out.append(f"{k}: {len(bad)} items lack {phs}, e.g. {bad[0][:80]!r}")
    for k, m in req.get("lists", {}).items():
        if k in obj and len(obj[k]) < m:
            out.append(f"{k} has {len(obj[k])} < {m} items")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("type_dir")
    ap.add_argument("--url", required=True, help="self-hosted K3 base url, e.g. http://HOST:8000/v1")
    ap.add_argument("--request", default="request.json")
    ap.add_argument("--out", default="k3.json")
    a = ap.parse_args()
    td = Path(a.type_dir) / "text"
    req = json.loads((td / a.request).read_text())
    user, attempts = req["user"], []
    for attempt in range(4):
        t0 = time.time()
        resp = call(user, a.url)
        msg = resp["choices"][0]["message"]["content"] or ""
        attempts.append({"seconds": round(time.time() - t0, 1), "usage": resp.get("usage"), "model": resp.get("model"),
                         "reply": msg})
        s = msg.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            obj = json.loads(s)
            errs = problems(obj, req)
        except ValueError as e:
            obj, errs = None, [f"not JSON: {e}"]
        attempts[-1]["problems"] = errs
        print(f"attempt {attempt}: {attempts[-1]['seconds']} s, problems={errs[:5]}", file=sys.stderr)
        if not errs:
            break
        user = req["user"] + "\n\nYour previous reply had these problems, fix them: " + "; ".join(errs[:20])
    raw = {"api": a.url, "model_requested": MODEL, "system": SYSTEM, "request": req, "attempts": attempts,
           "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    (td / a.out.replace(".json", "_raw.json")).write_text(json.dumps(raw, indent=1))
    if errs:
        sys.exit(f"K3 output still invalid: {errs[:5]}")
    (td / a.out).write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    print(f"wrote {td / a.out}")


if __name__ == "__main__":
    main()
