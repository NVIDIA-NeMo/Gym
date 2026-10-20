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
import heapq
import math
import random

DAY = 86400
SPAN = 7 * DAY
LOOKBACK = 2 * 3600  # incidents start at most this long before their ticket opens
HEALTH = 900  # a failure line propagates only if its target logs an ERROR within this many seconds

DOMAINS = ["checkout", "payments", "orders", "cart", "catalog", "search", "inventory", "shipping", "pricing", "users",
           "auth", "notifications", "reviews", "recs", "ads", "billing", "fraud", "ledger", "media", "messaging",
           "geo", "loyalty", "returns", "support", "analytics", "identity", "tax", "promotions", "warehouse",
           "fulfillment", "subscriptions", "wallet"]
COMPONENTS = [["web", "gateway", "bff", "mobile-api", "edge"],
              ["api", "graphql", "service", "frontend-api"],
              ["worker", "engine", "processor", "scheduler", "resolver", "aggregator", "validator", "sync", "indexer",
               "router", "writer", "reader", "planner", "matcher", "exporter", "consumer"],
              ["db-proxy", "cache-proxy", "store", "repo", "dao", "cache"],
              ["postgres", "redis", "kafka", "elasticsearch", "mysql", "cassandra", "memcached", "s3-gateway"]]
LANGS = ["go", "java", "python", "kotlin", "rust", "node"]
PEOPLE = ["a.kowalski", "b.okafor", "c.nguyen", "d.haddad", "e.lindqvist", "f.moreau", "g.tanaka", "h.patel",
          "i.romero", "j.schmidt", "k.osei", "l.ferrari", "m.ivanova", "n.cohen", "o.silva", "p.dubois", "q.zhang",
          "r.murphy", "s.yilmaz", "t.berg", "u.kim", "v.rossi", "w.adeyemi", "x.novak"]
BAD_DEPLOY = ["NullPointerException in {C}Serializer.write", "panic: index out of range in Batch{C}Writer",
              "IllegalStateException: {C}Cache not initialised", "TypeError: cannot read 'id' of undefined in {C}Handler",
              "deadlock detected in {C}Repository.save", "schema mismatch: unknown column '{c}_ref'"]
BAD_CONFIG = [("db.max_pool_size", "64", "4", "connection pool exhausted: {w} requests waiting (db.max_pool_size=4)"),
              ("http.client_timeout_ms", "2000", "40", "deadline exceeded before upstream reply (http.client_timeout_ms=40)"),
              ("tls.min_version", "1.2", "1.4", "TLS handshake failed: unsupported protocol (tls.min_version=1.4)"),
              ("cache.ttl_s", "300", "0", "cache stampede: backend saturated, {w} misses/s (cache.ttl_s=0)"),
              ("queue.max_inflight", "512", "1", "queue backpressure: {w} messages blocked (queue.max_inflight=1)"),
              ("auth.jwks_url", "https://jwks.internal/v2", "https://jwks.internal/v3",
               "token validation failed: jwks fetch 404 (auth.jwks_url=https://jwks.internal/v3)")]
SAFE_CONFIG = [("log.level", "info", "debug"), ("feature.new_layout", "false", "true"), ("rate_limit.rps", "500", "650"),
               ("metrics.sample_rate", "0.1", "0.2"), ("feature.dark_mode", "false", "true"),
               ("retry.backoff_ms", "100", "120"), ("gc.target_pct", "70", "75"), ("tracing.enabled", "false", "true")]
INFRA = [("node pool rotation: primary pod evicted", "primary unavailable: node drained, failover in progress"),
         ("certificate rotation on {s}", "TLS handshake failed: certificate signed by unknown authority"),
         ("storage migration to gp3 volumes", "write failed: volume is read-only"),
         ("minor version upgrade by platform", "replication broken: replica cannot connect to primary")]
SAFE_INFRA = ["routine backup snapshot", "kernel patch rollout", "autoscaler min replicas 3 -> 4",
              "monitoring agent upgrade", "log retention policy update"]
NOISE_ERR = ["unhandled KeyError in handler /v1/{r}: 'customer_id'", "failed to flush metrics batch: buffer full",
             "unhandled ValueError in handler /v1/{r}: invalid literal", "job {r}-reconcile crashed: assertion failed",
             "failed to parse webhook payload for /v1/{r}"]
BLIP_ERR = "health check failed: worker pool timeout"
CAUSES = ["deploy", "config", "infra", "leak"]


def _cap(s):
    return s.split("-")[0].capitalize()


def build_services(n, rng):
    L = 4 + round(math.log2(n) / 2)
    edge, infra = max(4, round(0.07 * n)), max(4, round(0.1 * n))
    mid = n - edge - infra
    layers = [edge] + [mid // (L - 2) + (1 if i < mid % (L - 2) else 0) for i in range(L - 2)] + [infra]
    used, svcs = {}, []
    for layer, size in enumerate(layers):
        tier = 0 if layer == 0 else 1 if layer == 1 else 4 if layer == L - 1 else 3 if layer == L - 2 else 2
        for _ in range(size):
            dom = rng.choice(DOMAINS)
            base = f"{dom}-{rng.choice(COMPONENTS[tier])}"
            used[base] = used.get(base, 0) + 1
            name = base if used[base] == 1 else f"{base}-{used[base]}"
            kind = {0: "web", 1: "api", 2: "service", 3: "service", 4: "datastore"}[tier]
            # Failing callers log every `period` s: edge services are loud, datastores quiet.
            period = {0: (10, 25), 1: (20, 45), 2: (45, 100), 3: (60, 120), 4: (120, 170)}[tier]
            svcs.append({"name": name, "layer": layer, "tier": tier, "kind": kind, "team": f"{dom}-team",
                         "lang": "-" if tier == 4 else rng.choice(LANGS), "owners": rng.sample(PEOPLE, 2),
                         "rps": round(rng.uniform(200, 2000) / (1 + layer), 1), "period": rng.randint(*period),
                         "mem_base": rng.choice([256, 384, 512, 768, 1024]), "mem_limit": rng.choice([1536, 2048, 3072]),
                         "ver": [rng.randint(1, 6), rng.randint(0, 30), rng.randint(0, 9)], "calls": [], "deps": [],
                         "ep": []})
    by_layer = [[i for i, s in enumerate(svcs) if s["layer"] == l] for l in range(L)]
    indeg = [0] * n
    for i, s in enumerate(svcs):
        if s["layer"] == L - 1:
            continue
        want = rng.randint(3, 6) if s["layer"] == 0 else rng.randint(2, 5)
        dom = s["team"]
        while len(s["calls"]) < want:
            l = min(L - 1, s["layer"] + (1 if rng.random() < 0.7 else rng.randint(2, 3)))
            pool = by_layer[l]
            same = [j for j in pool if svcs[j]["team"] == dom]
            cand = same if same and rng.random() < 0.5 else pool
            # Preferential attachment: shared infra ends up with many callers.
            j = rng.choices(cand, weights=[1 + indeg[c] for c in cand])[0]
            if svcs[j]["name"] not in s["calls"]:
                s["calls"].append(svcs[j]["name"])
                indeg[j] += 1
        # The catalog is stale: about 8% of real call edges are undeclared.
        s["deps"] = [d for d in s["calls"] if rng.random() > 0.08] or s["calls"][:1]
    return svcs, L


def _callers(svcs):
    idx = {s["name"]: i for i, s in enumerate(svcs)}
    callers = [[] for _ in svcs]
    for i, s in enumerate(svcs):
        for d in s["calls"]:
            callers[idx[d]].append(i)
    return idx, callers


def propagate(svcs, callers, root, t_on, cap, rng):
    # Earliest-arrival spread over a random subgraph of caller edges; parent = first failing dependency to reach it.
    onset, parent, hop = {root: t_on}, {root: None}, {root: 0}
    edges = {}  # (dep, caller) -> arrival time, for every propagating edge
    heap, done = [(t_on, root)], []
    while heap and len(done) < cap:
        t, i = heapq.heappop(heap)
        if i in done or onset.get(i) != t:
            continue
        done.append(i)
        for c in callers[i]:
            if c in done or rng.random() > 0.85 * 0.88 ** hop[i]:
                continue
            arr = t + rng.randint(45, 360)
            while any(a == arr for (d, cc), a in edges.items() if cc == c):
                arr += 1  # unique arrival times keep the parent well defined
            edges[(i, c)] = arr
            if arr < onset.get(c, 1 << 62):
                onset[c], parent[c], hop[c] = arr, i, hop[i] + 1
                heapq.heappush(heap, (arr, c))
    members = set(done)
    edges = {k: v for k, v in edges.items() if k[1] in members}
    return {i: onset[i] for i in done}, {i: parent[i] for i in done}, edges


class Gen:
    def __init__(self, n, seed):
        self.rng = random.Random(seed)
        self.n = n
        self.svcs, self.L = build_services(n, self.rng)
        self.idx, self.callers = _callers(self.svcs)
        self.changes = []
        self.quiet = {}  # service -> list of (a, b) where no background change may land
        self.windows = []

    def add_change(self, i, t, typ, **kw):
        c = {"id": None, "service": self.svcs[i]["name"], "t": t, "type": typ, **kw}
        self.changes.append(c)
        return c

    def incidents(self):
        rng, n, svcs = self.rng, self.n, self.svcs
        k = min(8, max(2, round(math.log2(n)) - 3))
        cap = max(8, n // 12)
        slot = SPAN // k
        causes = (CAUSES * 2)[:k]
        rng.shuffle(causes)
        incs = []
        for j, cause in enumerate(causes):
            for _ in range(500):
                if cause == "infra":
                    root = rng.choice([i for i, s in enumerate(svcs) if s["tier"] == 4 and self.callers[i]])
                else:
                    root = rng.choice([i for i, s in enumerate(svcs) if s["tier"] in (2, 3) and self.callers[i]])
                if any(root == x["root"] for x in incs):
                    continue
                t_on = j * slot + rng.randint(4 * 3600, slot - 5 * 3600)
                onset, parent, edges = propagate(svcs, self.callers, root, t_on, cap, rng)
                if len(onset) < 4:
                    continue
                top = [i for i in onset if svcs[i]["tier"] <= 1 and parent[i] is not None and parent[i] != root]
                if top:
                    break
            else:
                raise RuntimeError("could not place incident")
            low = min(svcs[i]["tier"] for i in top)
            ticket = max((i for i in top if svcs[i]["tier"] == low), key=lambda i: -svcs[i]["period"])
            alert = onset[ticket] + rng.randint(120, 300)
            opened = alert + rng.randint(60, 180)
            t_fix = max(max(onset.values()) + rng.randint(600, 1500), alert + rng.randint(1200, 3600))
            resolved = t_fix + rng.randint(300, 900)
            inc = {"cause": cause, "root": root, "t_on": t_on, "onset": onset, "parent": parent, "edges": edges,
                   "ticket_svc": ticket, "alert": alert, "opened": opened, "t_fix": t_fix, "resolved": resolved,
                   "window": (opened - LOOKBACK, resolved)}
            self.plan_root(inc)
            incs.append(inc)
            self.windows.append(inc["window"])
        return incs

    def plan_root(self, inc):
        rng, i, t_on = self.rng, inc["root"], inc["t_on"]
        s = self.svcs[i]
        cause, word = inc["cause"], _cap(s["name"])
        delay = rng.randint(60, 240)
        if cause == "leak":
            t_c = t_on - rng.randint(3 * 3600, 10 * 3600)
            inc["change"] = self.add_change(i, t_c, "deploy", bump=True)
            k, a, b = rng.choice(SAFE_CONFIG)
            self.add_change(i, rng.randint(t_c + 1800, t_on - 1800) // 60 * 60, "config", key=k, old=a, new=b)
            inc["msg"] = f"OOMKilled: container exceeded memory limit {s['mem_limit']}Mi, restarting"
            inc["oom_period"] = rng.randint(300, 600)
            self.quiet[i] = self.quiet.get(i, []) + [(t_c - 3 * 3600, inc["resolved"] + 3600)]
        else:
            t_c = (t_on - delay) // 60 * 60
            inc["t_on"] = t_c + delay
            if cause == "deploy":
                inc["change"] = self.add_change(i, t_c, "deploy", bump=True)
                inc["msg"] = rng.choice(BAD_DEPLOY).format(C=word, c=word.lower())  # version tag added later
                k, a, b = rng.choice(SAFE_CONFIG)
                self.add_change(i, t_c - rng.randint(10, 50) * 60, "config", key=k, old=a, new=b)
            elif cause == "config":
                k, a, b, msg = rng.choice(BAD_CONFIG)
                inc["change"] = self.add_change(i, t_c, "config", key=k, old=a, new=b)
                inc["msg"] = msg.format(w=rng.randint(20, 400))
                self.add_change(i, t_c - rng.randint(20, 90) * 60, "deploy", bump=True)
            else:
                what, msg = rng.choice(INFRA)
                inc["change"] = self.add_change(i, t_c, "infra", summary=what.format(s=s["name"]))
                inc["msg"] = msg
                self.add_change(i, t_c - rng.randint(60, 180) * 60, "infra", summary=rng.choice(SAFE_INFRA))
            self.quiet[i] = self.quiet.get(i, []) + [(t_c - 3 * 3600, inc["resolved"] + 3600)]
        # Shift the whole spread so it starts at the (rounded) true onset.
        shift = inc["t_on"] - t_on
        if shift:
            inc["onset"] = {k: v + shift for k, v in inc["onset"].items()}
            inc["edges"] = {k: v + shift for k, v in inc["edges"].items()}
            for key in ("alert", "opened"):
                inc[key] += shift
            inc["window"] = (inc["opened"] - LOOKBACK, inc["resolved"])
        # The mitigation lands at t_fix on the root.
        if cause == "config":
            c = inc["change"]
            self.add_change(i, inc["t_fix"] // 60 * 60, "config", key=c["key"], old=c["new"], new=c["old"],
                            note="revert")
        elif cause == "infra":
            self.add_change(i, inc["t_fix"] // 60 * 60, "infra", summary="failover completed, primary restored")
        else:
            self.add_change(i, inc["t_fix"] // 60 * 60, "deploy", rollback=True)
        inc["t_fix"] = inc["t_fix"] // 60 * 60
        # Coincident deploy on a failing caller between the culprit and the alert: bait for "latest deploy".
        later = [m for m in inc["onset"] if m != i and inc["onset"][m] < inc["alert"]]
        m = inc["ticket_svc"] if rng.random() < 0.5 else rng.choice(later)
        lo = max(inc["change"]["t"] + 60, inc["opened"] - 1800)
        t_d = rng.randint(lo, inc["opened"] - 30) // 60 * 60
        if t_d <= inc["change"]["t"]:
            t_d = inc["change"]["t"] + 60
        inc["decoy"] = self.add_change(m, t_d, "deploy", bump=True)
        self.quiet[m] = self.quiet.get(m, []) + [(t_d - 600, inc["opened"] + 600)]

    def background_changes(self):
        rng = self.rng
        for i, s in enumerate(self.svcs):
            quiet = self.quiet.get(i, [])
            rate = {4: 0.25}.get(s["tier"], 0.8)  # changes per day
            t = rng.expovariate(rate / DAY)
            while t < SPAN:
                tt = int(t) // 60 * 60
                if not any(a <= tt <= b for a, b in quiet):
                    if s["tier"] == 4:
                        if rng.random() < 0.5:
                            self.add_change(i, tt, "infra", summary=rng.choice(SAFE_INFRA))
                        else:
                            k, a, b = rng.choice(SAFE_CONFIG)
                            self.add_change(i, tt, "config", key=k, old=a, new=b)
                    elif rng.random() < 0.65:
                        self.add_change(i, tt, "deploy", bump=True)
                    else:
                        k, a, b = rng.choice(SAFE_CONFIG)
                        self.add_change(i, tt, "config", key=k, old=a, new=b)
                t += rng.expovariate(rate / DAY)

    def finalize_changes(self):
        rng = self.rng
        self.changes.sort(key=lambda c: (c["t"], c["service"]))
        ids = set()
        hist = {}
        for c in self.changes:
            while True:
                cid = "chg-%06x" % rng.getrandbits(24)
                if cid not in ids:
                    break
            ids.add(cid)
            c["id"] = cid
            s = self.svcs[self.idx[c["service"]]]
            c["author"] = "platform-bot" if c["type"] == "infra" else rng.choice(s["owners"] + ["deploy-bot"])
            if c["type"] == "deploy":
                stack = hist.setdefault(c["service"], ["v%d.%d.%d" % tuple(s["ver"])])
                if c.pop("rollback", False) and len(stack) > 1:
                    old, new = stack[-1], stack[-2]
                    stack.append(new)
                    c["summary"] = f"rollback {old} -> {new}"
                else:
                    c.pop("rollback", None)
                    old = stack[-1]
                    a, b, p = (int(x) for x in max(stack, key=lambda v: tuple(map(int, v[1:].split("."))))[1:].split("."))
                    new = f"v{a}.{b + 1}.0" if rng.random() < 0.4 else f"v{a}.{b}.{p + 1}"
                    stack.append(new)
                    c["summary"] = f"deploy {old} -> {new}"
                c["old"], c["new"] = old, new
                c.pop("bump", None)
            elif c["type"] == "config":
                note = c.pop("note", "")
                c["summary"] = f"{'revert ' if note else ''}set {c['key']}: {c['old']} -> {c['new']}"

    def healthy(self, i, a, b):
        return not any(e["start"] <= b and e["end"] >= a for e in self.svcs[i]["ep"] if e["level"] == "ERROR")

    def in_window(self, t, margin=0):
        return any(a - margin <= t <= b + margin for a, b in self.windows)

    def emit_incidents(self, incs):
        svcs = self.svcs
        for inc in incs:
            r = svcs[inc["root"]]
            msg = inc["msg"]
            if inc["cause"] == "deploy":
                msg = f"[{inc['change']['new']}] {msg}"
            period = inc["oom_period"] if inc["cause"] == "leak" else self.rng.randint(90, 170)
            r["ep"].append({"kind": "self", "level": "ERROR", "start": inc["t_on"], "end": inc["t_fix"],
                            "period": period, "msg": msg})
            if inc["cause"] == "leak":
                r["leak"] = [inc["change"]["t"], inc["t_on"], inc["t_fix"], inc["oom_period"]]
            for (d, c), arr in inc["edges"].items():
                style = "timeout" if self.rng.random() < 0.4 else "503"
                svcs[c]["ep"].append({"kind": "fail", "level": "ERROR", "start": arr, "target": svcs[d]["name"],
                                      "end": inc["t_fix"] + self.rng.randint(30, 240), "period": svcs[c]["period"],
                                      "style": style})

    def emit_noise(self, incs):
        rng, svcs, n = self.rng, self.svcs, self.n
        member_ws = [(set(inc["onset"]), inc["window"]) for inc in incs]
        roots = {inc["root"] for inc in incs}
        change_times = {}
        for c in self.changes:
            change_times.setdefault(c["service"], []).append(c["t"])

        def busy(i, a, b):
            # True if i is a member of an incident whose window (plus margin) overlaps [a, b].
            return any(i in m and a <= w[1] + 1800 and b >= w[0] - 1800 for m, w in member_ws)

        # Loud unrelated error bursts: some inside every incident window, the rest anywhere.
        bursts = []
        for inc in incs:
            for _ in range(3):
                bursts.append((inc["alert"] - rng.randint(0, 1800), rng.randint(3, 10), rng.randint(1500, 3600)))
        for _ in range(max(3, n // 12)):
            bursts.append((rng.randint(0, SPAN - 3600), rng.randint(5, 60), rng.randint(180, 1800)))
        for t, period, dur in bursts:
            for _ in range(100):
                i = rng.randrange(n)
                # No change on i shortly before, so an unrelated burst never has a plausible culprit.
                near = any(t - 900 <= ct <= t + dur for ct in change_times.get(svcs[i]["name"], []))
                if not busy(i, t, t + dur) and i not in roots and not near:
                    r = svcs[i]["team"].split("-")[0]
                    svcs[i]["ep"].append({"kind": "self", "level": "ERROR", "start": t, "end": t + dur,
                                          "period": period, "msg": rng.choice(NOISE_ERR).format(r=r)})
                    break
        # Short self-healing blips that do propagate one hop, always well away from incident windows.
        for _ in range(max(2, n // 40)):
            for _ in range(100):
                q = rng.randrange(n)
                t = rng.randint(0, SPAN - 3600)
                if self.in_window(t, 3 * 3600) or not self.callers[q]:
                    continue
                dur = rng.randint(120, 420)
                svcs[q]["ep"].append({"kind": "self", "level": "ERROR", "start": t, "end": t + dur, "period": 40,
                                      "msg": BLIP_ERR})
                for c in rng.sample(self.callers[q], min(3, len(self.callers[q]))):
                    svcs[c]["ep"].append({"kind": "fail", "level": "ERROR", "start": t + rng.randint(30, 90),
                                          "target": svcs[q]["name"], "end": t + dur + 60,
                                          "period": svcs[c]["period"], "style": "503"})
                break
        # Isolated failures against healthy dependencies ("flaky network"): noise that looks like propagation.
        # They are dense inside incident windows, so every service needs its failure lines checked.
        into, out_of = {}, {}  # service -> times of isolated lines pointing at it / logged by it
        gap = HEALTH + 120

        def try_isolated(c, t):
            deps = [self.idx[d] for d in svcs[c]["calls"]]
            if not deps or c in roots:
                return False
            x = rng.choice(deps)
            if busy(x, t, t) or not self.healthy(x, t - gap, t + gap):
                return False
            # c's new ERROR line must not turn an earlier isolated line against c into apparent propagation.
            if any(abs(tt - t) <= gap for tt in into.get(c, [])):
                return False
            into.setdefault(x, []).append(t)
            out_of.setdefault(c, []).append(t)
            svcs[c]["ep"].append({"kind": "fail", "level": "ERROR", "start": t, "end": t, "period": 60,
                                  "target": svcs[x]["name"], "style": rng.choice(["503", "timeout"])})
            return True

        for inc in incs:
            a, b = inc["window"]
            for m in inc["onset"]:
                for _ in range(rng.choice([0, 0, 1, 2]) if m != inc["root"] else 0):
                    try_isolated(m, rng.randint(a, b))
            want = round(1.2 * n)
            for _ in range(want * 4):
                if want == 0:
                    break
                c = rng.randrange(n)
                if c in inc["onset"] or busy(c, a, b):
                    continue
                want -= try_isolated(c, rng.randint(a, b))
        for _ in range(n // 4):
            c = rng.randrange(n)
            t = rng.randint(0, SPAN)
            if not busy(c, t - 3600, t + 3600):
                try_isolated(c, t)
