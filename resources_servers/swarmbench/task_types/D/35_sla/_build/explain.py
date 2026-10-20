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
"""Audit helper: python3 35_sla/_build/explain.py N SEED CUS0001 [...] (run from types/D). Prints every incident that
touches the customer's instances with the gold decision, the contract parameters, tickets and claims."""
import sys

sys.path.insert(0, ".")
import loader  # noqa: E402

m = loader.load_type("35_sla")
n, seed = int(sys.argv[1]), int(sys.argv[2])
gincs, gnot = m.global_events(seed)
used = set()
cs = [m._customer(i, seed, used, gincs, gnot) for i in range(n)]
allinc = gincs + [x for c in cs for x in c.incs]
alln = {x["id"]: x for x in gnot + [x for c in cs for x in c.notices]}
for cid in sys.argv[3:]:
    c = next(c for c in cs if c.id == cid)
    print("=====", cid, "target", c.target, "tiers", c.tiers, "w", c.weight, "minlen", c.minlen, "notice_h", c.notice_h,
          "win", m.window_text(c), "upstream_counts", c.upstream_counts, "tz", c.tz, "cap", c.cap,
          "p", c.p1, c.p2, c.p3, "sc", c.support_credit, "claimdays", c.claim_days)
    print("inst", c.inst)
    for x in sorted(allinc, key=lambda x: x["start"]):
        if not m.affects(c, x):
            continue
        one = m.downtime(c, [x], list(alln.values()))
        extra = ""
        if x["notice"] != "-":
            nt = alln[x["notice"]]
            extra = f" notice sent {nt['sent']} lead_h {(nt['ps'] - nt['sent']).total_seconds() / 3600:.2f} planned {nt['ps']}-{nt['pe']} inwin {m.in_window(c, nt['ps'], nt['pe'])} wd {nt['ps'].strftime('%a')}"
        print(f"  {m.inc_row(x)} | full {m.mins(x['start'], x['end'])} | counted {one}{extra}")
    print("  total downtime", m.downtime(c, allinc, list(alln.values())), "avail", c.avail, c.breaches, c.credit)
    for t in c.tickets:
        print("  ticket", t["prio"], t["open"], t["resp"], "cal", m.mins(t["open"], t["resp"]), "bh",
              m.business_minutes(t["open"], t["resp"], c.tz))
    print("  claims", c.claims)
