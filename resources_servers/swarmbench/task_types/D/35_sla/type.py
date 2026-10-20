"""Type 35 sla: SLA breach and service-credit audit for one reporting month over N customer contracts.

Unit = one customer contract for March 2026. Gold comes from a minute-exact simulation of the incident log, the
maintenance notices, the support tickets and the claims against each contract's parameters. Task-facing prose
(prompt, rules, every contract clause, tool descriptions) is K3's (text/k3.json, see README); ids, names, times,
parameters and tables are code."""
import datetime as dt
import json
import random
import re
import threading
from pathlib import Path

from .. import common

TYPE_ID = "35_sla"
TIERS = {"easy": (6, 1800), "medium": (16, 1200), "hard": (60, 1200)}  # K3 single-unit loop ~145 s/unit
MAX_CUSTOMERS = 2000
HERE = Path(__file__).parent
T = common.load_text(HERE)
M0, M1 = dt.datetime(2026, 3, 1), dt.datetime(2026, 4, 1)
MONTH = "2026-03"
TOTAL = int((M1 - M0).total_seconds() // 60)  # 44640 minutes
SERVICES = ["compute", "object-storage", "managed-db", "message-queue", "cdn", "identity", "search", "analytics"]
REGIONS = ["us-east", "us-west", "eu-central", "eu-west", "ap-south", "ap-northeast"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TZS = [-480, -300, 0, 60, 330, 540]
TARGETS = {99.95: (99.5, 99.0), 99.9: (99.0, 95.0), 99.5: (99.0, 95.0), 99.0: (97.0, 95.0)}
NAME_A = ["Aldermoor", "Brackenridge", "Calloway", "Dunmore", "Elmstead", "Fenwick", "Garrison", "Hollins", "Ingram",
          "Jessop", "Kirkland", "Langford", "Merriweather", "Northcott", "Ormsby", "Prescott", "Quinlan", "Radley",
          "Stanhope", "Thackeray", "Underhill", "Vickery", "Whitlock", "Yelverton"]
NAME_B = ["Biotech", "Insurance", "Logistics", "Media", "Retail", "Energy", "Telecom", "Travel", "Gaming", "Health",
          "Banking", "Education", "Robotics", "Foods", "Mobility", "Analytics"]
AVAIL_KINDS = ["degraded", "overlap", "maint_ok", "maint_late", "maint_outside", "short", "straddle", "own_action",
               "upstream", "force_majeure", "decoy"]
TICKET_KINDS = ["t_bh_ok", "t_bh_breach", "t_p1_weekend", "t_p1_ok", "t_prev_month", "t_tz"]
CLAIM_KINDS = ["ok", "ok", "ok", "late", "none", "wrong_month"]


def hid(prefix, slot, mul, add, mod=1000003, width=7):
    return f"{prefix}-{(slot * mul + add) % mod:0{width}d}"


def inst_id(i, k):
    return hid("INS", i * 10 + k, 7919, 12345)


def ts(t):
    return t.strftime("%Y-%m-%d %a %H:%M")


def tz_text(m):
    return f"UTC{'+' if m >= 0 else '-'}{abs(m) // 60:02d}:{abs(m) % 60:02d}"


def mins(a, b):
    return int((b - a).total_seconds() // 60)


def business_minutes(a, b, tzm):
    """Minutes in [a, b) that fall Mon-Fri 09:00-17:00 in the customer's fixed-offset local time."""
    off = dt.timedelta(minutes=tzm)
    la, lb, n = a + off, b + off, 0
    day = dt.datetime(la.year, la.month, la.day)
    while day < lb:
        if day.weekday() < 5:
            s, e = max(la, day + dt.timedelta(hours=9)), min(lb, day + dt.timedelta(hours=17))
            n += max(0, mins(s, e))
        day += dt.timedelta(days=1)
    return n


# ---------- world ----------
def _rand_time(rng, lo=M0, hi=M1):
    return lo + dt.timedelta(minutes=rng.randrange(mins(lo, hi)))


def _inc(iid, start, dur, service, region, scope, impact="outage", cause="platform", by="-", notice="-"):
    return {"id": iid, "start": start, "end": start + dt.timedelta(minutes=dur), "service": service, "region": region,
            "scope": scope, "impact": impact, "cause": cause, "by": by, "notice": notice}


def global_events(seed):
    """Region-wide incidents and maintenances: depend on the seed only, so unit i depends only on (seed, i)."""
    rng = random.Random(f"sla:{seed}:global")
    incs, notices = [], []
    for j in range(14):
        s, r = rng.choice(SERVICES), rng.choice(REGIONS)
        cause = rng.choices(["platform", "network", "upstream_provider", "force_majeure", "customer_action"],
                            [45, 20, 15, 8, 12])[0]
        by = hid("TEN", j, 104729, 777, 100003, 5) if cause == "customer_action" else "-"
        incs.append(_inc(hid("INC", j, 7927, 4242), _rand_time(rng), rng.randint(3, 30), s, r, "region-wide",
                         rng.choice(["outage", "outage", "degraded"]), cause, by))
    for j in range(6):
        s, r = rng.choice(SERVICES), rng.choice(REGIONS)
        day = M0 + dt.timedelta(days=rng.randrange(31))
        ps = day + dt.timedelta(hours=rng.randint(0, 3), minutes=rng.choice([0, 15, 30]))
        pdur = rng.choice([30, 45, 60, 90])
        nid = hid("NTC", j, 6007, 99, 100003, 5)
        notices.append({"id": nid, "sent": ps - dt.timedelta(hours=rng.choice([6, 20, 30, 50, 80, 100]),
                                                             minutes=rng.randint(0, 59)),
                        "service": s, "region": r, "scope": "region-wide", "ps": ps,
                        "pe": ps + dt.timedelta(minutes=pdur)})
        over = rng.choice([0, 0, rng.randint(5, 25)])
        incs.append(_inc(hid("INC", 100 + j, 7927, 4242), ps, pdur + over, s, r, "region-wide", "outage",
                         "maintenance", "-", nid))
    return incs, notices


class _Cust:
    pass


def _params(i, seed, used):
    rng = random.Random(f"sla:{seed}:{i}:params")
    c = _Cust()
    c.i, c.id = i, f"CUS{i + 1:04d}"
    c.contract = hid("SLA", i, 4999, 321, 100003, 5)
    while True:
        name = f"{rng.choice(NAME_A)} {rng.choice(NAME_B)} {rng.choice(['Ltd', 'Inc', 'GmbH', 'Group', 'plc'])}"
        if name not in used or len(used) > 1500:
            break
    used.add(name)
    c.name, c.tz = name, rng.choice(TZS)
    c.target = rng.choice(list(TARGETS))
    t2, t3 = TARGETS[c.target]
    c1 = rng.choice([5, 10])
    c2 = rng.choice([15, 25])
    c3 = rng.choice([30, 50])
    c.tiers = [(c.target, c1), (t2, c2), (t3, c3)]
    c.weight = rng.choice([1.0, 0.5, 0.0])
    c.minlen = rng.choice([0, 5, 10, 15])
    c.notice_h = rng.choice([24, 48, 72])
    c.win = (rng.choice([5, 6, 1, 2]), rng.randint(0, 2), rng.randint(4, 6))  # weekday, start hour, length h
    c.upstream_counts = rng.random() < 0.5
    c.p1, c.p2, c.p3 = rng.choice([30, 60, 120]), rng.choice([4, 8]), rng.choice([16, 24])
    c.support_credit = rng.choice([5, 10])
    c.cap = rng.choice([20, 25, 30, 40])
    c.claim_days = rng.choice([15, 30, 45])
    c.capped = rng.random() < 0.2  # a major outage pushes the tier credit past a low cap
    if c.capped:
        c.cap, c.tiers[1] = 20, (t2, 25)
    pairs = rng.sample([(s, r) for s in SERVICES for r in REGIONS], rng.choice([2, 3]))
    c.inst = [(inst_id(i, k), s, r) for k, (s, r) in enumerate(pairs)]
    excl = ["maint_ok", "own_action", "force_majeure"] + (["short"] if c.minlen else []) + \
        ([] if c.upstream_counts else ["upstream"])
    first = rng.choice(excl)  # every unit has an exclusion that changes its availability
    c.avail_kinds = [first, rng.choice([k for k in AVAIL_KINDS if k != first])]
    c.base = rng.random() < 0.5
    c.ticket_kinds = [rng.choice(["t_bh_ok", "t_prev_month", "t_p1_ok", "t_tz"])] + \
        ([rng.choice(["t_bh_breach", "t_p1_weekend"])] if rng.random() < 0.4 else [])
    c.claim_kind = rng.choice(CLAIM_KINDS)
    c.order = rng.sample(range(12), 12)
    return c


def fmt(x):
    return f"{x:.2f}"  # 99.50, not 99.5: K3 misread "99.767 < 99.5" in 3 of 20 units with the short form


def window_text(c):
    d, h, ln = c.win
    return f"{DAYS[d]} {h:02d}:00-{h + ln:02d}:00 UTC"


def in_window(c, ps, pe):
    d, h, ln = c.win
    day = dt.datetime(ps.year, ps.month, ps.day)
    return ps.weekday() == d and ps >= day + dt.timedelta(hours=h) and pe <= day + dt.timedelta(hours=h + ln)


def _window_start(c, rng, inside=True):
    d, h, ln = c.win
    days = [M0 + dt.timedelta(days=k) for k in range(1, 27) if (M0 + dt.timedelta(days=k)).weekday() == d]
    day = rng.choice(days)
    if inside:
        return day + dt.timedelta(hours=h, minutes=rng.choice([0, 15, 30]))
    if rng.random() < 0.5:  # wrong weekday, same hours
        return day + dt.timedelta(days=rng.choice([1, 2, 3]), hours=h, minutes=rng.choice([0, 30]))
    return day + dt.timedelta(hours=h + ln - 1, minutes=rng.choice([0, 15, 30]))  # runs past the window end


def _planted(c, seed, attempt):
    """Instance-level incidents and notices of customer c (only its own or decoy instance ids)."""
    rng = random.Random(f"sla:{seed}:{c.i}:inc:{attempt}")
    allow = (100 - c.target) / 100 * TOTAL
    incs, notices, slot = [], [], [0]

    def nid():
        slot[0] += 1
        return 1000 + c.i * 40 + slot[0]

    def own():
        return rng.choice(c.inst)

    def dur(lo, hi):
        return max(3, int(rng.uniform(lo, hi) * allow))

    def add(start, d, impact="outage", cause="platform", by="-", notice="-", inst=None):
        x = inst or own()
        incs.append(_inc(hid("INC", nid(), 7927, 4242), start, d, x[1], x[2], x[0], impact, cause, by, notice))
        return x

    def maint(inside, lead_ok):
        x = own()
        ps = _window_start(c, rng, inside)
        pdur = rng.choice([60, 90, 120, 150]) if inside else rng.choice([90, 120])
        if inside:
            d, h, ln = c.win
            pdur = min(pdur, (h + ln) * 60 - (ps.hour * 60 + ps.minute))
        lead = c.notice_h * 60 + rng.randint(120, 2880) if lead_ok else c.notice_h * 60 - rng.randint(60, 1200)
        n_id = hid("NTC", nid(), 6007, 99, 100003, 5)
        notices.append({"id": n_id, "sent": ps - dt.timedelta(minutes=lead), "service": x[1], "region": x[2],
                        "scope": x[0], "ps": ps, "pe": ps + dt.timedelta(minutes=pdur)})
        over = rng.choice([0, rng.randint(5, 30)])
        add(ps, pdur + over, "outage", "maintenance", "-", n_id, x)

    if c.base or not set(c.avail_kinds) & {"degraded", "overlap", "maint_late", "maint_outside", "straddle"}:
        add(_rand_time(rng), dur(0.1, 0.5))
    if c.capped:
        t2 = c.tiers[1][0]
        add(_rand_time(rng, M0 + dt.timedelta(days=1), M1 - dt.timedelta(days=3)),
            int((100 - t2) / 100 * TOTAL * rng.uniform(1.1, 1.5)))
    for k in c.avail_kinds:
        if k == "degraded":
            add(_rand_time(rng), dur(0.3, 1.0), "degraded")
        elif k == "overlap":
            s = _rand_time(rng, M0 + dt.timedelta(days=1), M1 - dt.timedelta(days=1))
            d1 = dur(0.2, 0.6)
            add(s, d1, rng.choice(["outage", "degraded"]))
            add(s + dt.timedelta(minutes=rng.randint(1, max(1, d1 - 1))), dur(0.2, 0.6), rng.choice(["outage", "degraded"]))
        elif k == "maint_ok":
            maint(True, True)
        elif k == "maint_late":
            maint(True, False)
        elif k == "maint_outside":
            maint(False, True)
        elif k == "short":
            m = c.minlen
            if m:
                for _ in range(rng.choice([1, 2])):
                    add(_rand_time(rng), rng.randint(max(1, m - 4), m - 1))
                if rng.random() < 0.5:
                    add(_rand_time(rng), m)
            else:
                for _ in range(2):
                    add(_rand_time(rng), rng.randint(2, 4))
        elif k == "straddle":
            if rng.random() < 0.5:
                s = M0 - dt.timedelta(minutes=rng.randint(10, 120))
                add(s, mins(s, M0) + rng.randint(10, 90))
            else:
                s = M1 - dt.timedelta(minutes=rng.randint(10, 90))
                add(s, mins(s, M1) + rng.randint(10, 120))
        elif k == "own_action":
            add(_rand_time(rng), dur(0.5, 2.0), "outage", "customer_action", c.id)
        elif k == "upstream":
            add(_rand_time(rng), dur(0.3, 0.9), "outage", "upstream_provider")
        elif k == "force_majeure":
            add(_rand_time(rng), dur(0.5, 2.0), "outage", "force_majeure")
        elif k == "decoy":
            x = own()
            add(_rand_time(rng), dur(0.5, 2.0), "outage", "platform", "-", "-", (inst_id(c.i, 5 + rng.randrange(5)),
                                                                                x[1], x[2]))
    return incs, notices


def affects(c, inc):
    mine = {x[0] for x in c.inst}
    if inc["scope"] == "region-wide":
        return any(s == inc["service"] and r == inc["region"] for _, s, r in c.inst)
    return bool(mine & set(inc["scope"].split(", ")))


def downtime(c, incs, notices):
    """Weighted downtime minutes: union of counted pieces, heaviest weight per minute, clipped to the month."""
    nt = {n["id"]: n for n in notices}
    pieces = []
    for inc in incs:
        if not affects(c, inc):
            continue
        if inc["cause"] == "force_majeure" or (inc["cause"] == "customer_action" and inc["by"] == c.id):
            continue
        if inc["cause"] == "upstream_provider" and not c.upstream_counts:
            continue
        if mins(inc["start"], inc["end"]) < c.minlen:
            continue
        w = 1.0 if inc["impact"] == "outage" else c.weight
        if w == 0:
            continue
        parts = [(inc["start"], inc["end"])]
        n = nt.get(inc["notice"])
        if inc["cause"] == "maintenance" and n and mins(n["sent"], n["ps"]) >= c.notice_h * 60 and \
                in_window(c, n["ps"], n["pe"]):
            parts = [(inc["start"], min(inc["end"], n["ps"])), (max(inc["start"], n["pe"]), inc["end"])]
        for a, b in parts:
            a, b = max(a, M0), min(b, M1)
            if a < b:
                pieces.append((mins(M0, a), mins(M0, b), w))
    pts = sorted({p for a, b, _ in pieces for p in (a, b)})
    tot = 0.0
    for a, b in zip(pts, pts[1:]):
        ws = [w for s, e, w in pieces if s <= a and e >= b]
        if ws:
            tot += (b - a) * max(ws)
    return tot


def _tickets(c, seed):
    rng = random.Random(f"sla:{seed}:{c.i}:tickets")
    out = []
    tgt = {"P1": c.p1, "P2": c.p2 * 60, "P3": c.p3 * 60}

    def mk(kind):
        for _ in range(20000):
            prio = "P1" if kind.startswith("t_p1") else rng.choice(["P2", "P3"])
            if kind == "t_prev_month":
                prio = rng.choice(["P1", "P2", "P3"])
                op = dt.datetime(2026, 2, 20) + dt.timedelta(minutes=rng.randrange(8 * 1440))
                if op >= M0:
                    continue
            else:
                op = _rand_time(rng, M0, M1 - dt.timedelta(days=6))
            delta = int(rng.choice([rng.uniform(5, 300), rng.uniform(60, 4000), rng.uniform(600, 7000)]))
            if kind == "t_tz":
                delta = int(rng.uniform(tgt[prio] * 0.8, tgt[prio] + 3 * 1440))
            rs = op + dt.timedelta(minutes=delta)
            cal, bh = mins(op, rs), business_minutes(op, rs, c.tz)
            bh_utc = business_minutes(op, rs, 0) if kind == "t_tz" else 0
            t = tgt[prio]
            lo = op + dt.timedelta(minutes=c.tz)
            ok = {"t_bh_ok": bh <= t - 15 and cal >= t + 15,
                  "t_bh_breach": bh >= t + 15,
                  "t_p1_weekend": lo.weekday() >= 5 and cal >= t + 15,
                  "t_p1_ok": cal <= t - 10,
                  "t_prev_month": rs >= M0 and cal >= t + 60 and bh >= t + 60,
                  "t_tz": prio != "P1" and abs(bh - t) >= 10 and abs(bh_utc - t) >= 10 and (bh > t) != (bh_utc > t),
                  "plain": bh <= t - 15 and cal <= t - 15}[kind]
            if ok:
                return {"prio": prio, "open": op, "resp": rs}
        return None
    if c.tz == 0 and "t_tz" in c.ticket_kinds:
        c.ticket_kinds = [k if k != "t_tz" else "t_bh_ok" for k in c.ticket_kinds]
    for k in c.ticket_kinds + ["plain"] * rng.choice([0, 1, 2]):
        out.append(mk(k) or mk("t_bh_ok"))
    out.sort(key=lambda x: x["open"])
    for j, t in enumerate(out):
        t["id"] = hid("TKT", c.i * 10 + j, 3001, 17, 1000003, 6)
    return out


def support_breached(c, tickets):
    for t in tickets:
        if not (M0 <= t["open"] < M1):
            continue
        if t["prio"] == "P1":
            if mins(t["open"], t["resp"]) > c.p1:
                return True
        elif business_minutes(t["open"], t["resp"], c.tz) > (c.p2 if t["prio"] == "P2" else c.p3) * 60:
            return True
    return False


def _claims(c, seed):
    rng = random.Random(f"sla:{seed}:{c.i}:claims")
    last = M1.date() - dt.timedelta(days=1)
    if c.claim_kind == "none":
        return []
    if c.claim_kind == "ok":
        return [("2026-03", last + dt.timedelta(days=rng.randint(1, c.claim_days)))]
    if c.claim_kind == "late":
        return [("2026-03", last + dt.timedelta(days=c.claim_days + rng.randint(1, 20)))]
    return [("2026-02", dt.date(2026, 3, rng.randint(2, 25)))]


def claim_valid(c, claims):
    last = M1.date() - dt.timedelta(days=1)
    return any(m == MONTH and d <= last + dt.timedelta(days=c.claim_days) for m, d in claims)


def verdict(c, avail, sup, valid):
    breaches = (["availability"] if avail < c.target else []) + (["support"] if sup else [])
    tier = 0
    for th, cr in c.tiers:
        if avail < th:
            tier = cr
    credit = min(c.cap, tier + (c.support_credit if sup else 0))
    return breaches, (credit if valid else 0)


def _customer(i, seed, used, gincs, gnotices):
    c = _params(i, seed, used)
    for attempt in range(200):
        incs, notices = _planted(c, seed, attempt)
        avail = 100 * (TOTAL - downtime(c, gincs + incs, gnotices + notices)) / TOTAL
        if all(abs(avail - th) >= 0.003 for th, _ in c.tiers):
            break
    c.incs, c.notices, c.avail = incs, notices, avail
    c.tickets = _tickets(c, seed)
    c.claims = _claims(c, seed)
    c.sup = support_breached(c, c.tickets)
    c.breaches, c.credit = verdict(c, avail, c.sup, claim_valid(c, c.claims))
    return c


def render_contract(c):
    t1, t2, t3 = c.tiers
    pick = lambda key, i: T[key][(c.i + i) % len(T[key])]  # noqa: E731
    clauses = [common.fill(pick("c_target", 0), target=fmt(c.target)),
               pick({1.0: "c_deg_full", 0.5: "c_deg_half", 0.0: "c_deg_none"}[c.weight], 1),
               common.fill(pick("c_minlen", 2), minutes=c.minlen) if c.minlen else pick("c_minlen_none", 2),
               common.fill(pick("c_maint", 3), hours=c.notice_h, window=window_text(c)),
               pick("c_upstream_counts" if c.upstream_counts else "c_upstream_excluded", 4),
               pick("c_force_majeure", 5), pick("c_customer_action", 6),
               common.fill(pick("c_support", 7), p1=c.p1, p2=c.p2, p3=c.p3),
               common.fill(pick("c_support_credit", 8), support_credit=c.support_credit),
               common.fill(pick("c_tiers", 9), t1=fmt(t1[0]), c1=t1[1], t2=fmt(t2[0]), c2=t2[1], t3=fmt(t3[0]), c3=t3[1]),
               common.fill(pick("c_cap", 10), cap=c.cap),
               common.fill(pick("c_claim", 11), days=c.claim_days)]
    clauses = [clauses[k] for k in c.order]
    body = "\n\n".join(f"{j + 1}. {x}" for j, x in enumerate(clauses))
    return (f"CONTRACT {c.contract}\nCustomer: {c.name} ({c.id})\nCustomer timezone: {tz_text(c.tz)}\n"
            f"Reporting month: {MONTH} ({(M1 - M0).days} days, {TOTAL} minutes)\n\n{body}\n")


def inc_row(x):
    return " | ".join([x["id"], ts(x["start"]), ts(x["end"]), str(mins(x["start"], x["end"])), x["service"], x["region"], x["scope"], x["impact"],
                       x["cause"], x["by"], x["notice"]])


def generate(n, seed, out_dir):
    assert n <= MAX_CUSTOMERS
    gincs, gnotices = global_events(seed)
    used = set()
    cs = [_customer(i, seed, used, gincs, gnotices) for i in range(n)]
    incs = sorted(gincs + [x for c in cs for x in c.incs], key=lambda x: (x["start"], x["id"]))
    notices = sorted(gnotices + [x for c in cs for x in c.notices], key=lambda x: x["id"])
    gold = {"customers": {c.id: {"breaches": c.breaches, "availability": round(c.avail, 6), "credit": c.credit,
                                 "kinds": c.avail_kinds + c.ticket_kinds + ["claim_" + c.claim_kind]
                                 + (["base"] if c.base else []) + (["cap"] if c.capped else [])} for c in cs}}
    world = {
        "customers.json": [{"id": c.id, "row": f"{c.id} | {c.name} | {c.contract}", "contract": render_contract(c),
                            "subs": [f"{a} | {s} | {r}" for a, s, r in c.inst],
                            "tickets": [f"{t['id']} | {c.id} | {t['prio']} | {ts(t['open'])} | {ts(t['resp'])}"
                                        for t in c.tickets],
                            "claims": [f"{hid('CLM', c.i * 3 + j, 2003, 5, 100003, 5)} | {c.id} | {m} | {d}"
                                       for j, (m, d) in enumerate(c.claims)]} for c in cs],
        "incidents.json": [inc_row(x) for x in incs],
        "notices.json": {x["id"]: " | ".join([x["id"], ts(x["sent"]), x["service"], x["region"], x["scope"],
                                              ts(x["ps"]), ts(x["pe"])]) for x in notices}}
    prompt = "\n\n".join([common.fill(T["intro"], company=T["company"], n=n, first=cs[0].id, last=cs[-1].id,
                                      month=MONTH), rules_text(example(cs[0].id))])
    return common.finish(TYPE_ID, n, seed, out_dir, prompt, gold, TIERS, files=world)


def example(cid):
    return json.dumps({cid: {"breaches": ["availability"], "availability": 99.123, "credit": 10}})


def rules_text(ex):
    return "\n\n".join([T[k] for k in ("rule_scope", "rule_measure", "rule_support", "rule_credit")]
                       + [common.fill(T["answer_line"], example=ex)])


# ---------- tools ----------
_cache, _lock = {}, threading.Lock()
INC_HEAD = "incident | start (UTC) | end (UTC) | logged minutes | service | region | scope | impact | cause | caused by | notice"


def _world(world_dir):
    with _lock:
        if world_dir not in _cache:
            p = Path(world_dir)
            cs = json.loads((p / "customers.json").read_text())
            _cache[world_dir] = ({c["id"]: c for c in cs}, [c["id"] for c in cs],
                                 json.loads((p / "incidents.json").read_text()),
                                 json.loads((p / "notices.json").read_text()))
        return _cache[world_dir]


def _page(head, rows, off, lim, what):
    out, size = [], len(head)
    for r in rows[off:off + lim]:
        if out and size + len(r) + 1 > 7500:
            break
        out.append(r)
        size += len(r) + 1
    nxt = off + len(out)
    more = f"\n({what} {off}-{nxt - 1} of {len(rows)}; next offset {nxt})" if nxt < len(rows) else \
        f"\n({len(rows)} {what})"
    return head + "\n" + "\n".join(out) + more


def _cust(world_dir, a):
    by = _world(world_dir)[0]
    cid = str(a["customer_id"]).strip().upper()
    if cid not in by:
        raise KeyError(f"unknown customer {cid}")
    return by[cid]


def _parse_ts(s):
    d, _, hm = s.split(" ")
    return dt.datetime.fromisoformat(f"{d} {hm}")


def _day(s, name):
    if not s:
        return None
    return dt.datetime.fromisoformat(str(s).strip()[:10])


def tools(world_dir):
    def list_customers(a):
        by, order, _, _ = _world(world_dir)
        off, lim = common.page_args(a, 100, 100)
        return _page("customer | name | contract", [by[x]["row"] for x in order], off, lim, "customers")

    def read_contract(a):
        return _cust(world_dir, a)["contract"]

    def get_subscriptions(a):
        return "instance | service | region\n" + "\n".join(_cust(world_dir, a)["subs"])

    def list_tickets(a):
        c = _cust(world_dir, a)
        return "ticket | customer | priority | opened (UTC) | first response (UTC)\n" + "\n".join(c["tickets"]) \
            + f"\n({len(c['tickets'])} tickets)"

    def list_claims(a):
        c = _cust(world_dir, a)
        return "claim | customer | month claimed | filed on\n" + "\n".join(c["claims"]) + \
            f"\n({len(c['claims'])} claims)"

    def search_incidents(a):
        _, _, rows, _ = _world(world_dir)
        sv, rg = str(a.get("service") or "").strip().lower(), str(a.get("region") or "").strip().lower()
        ins = str(a.get("instance_id") or "").strip().upper()
        d0, d1 = _day(a.get("start_date"), "start_date"), _day(a.get("end_date"), "end_date")
        hits = []
        for r in rows:
            f = r.split(" | ")
            if sv and f[4] != sv or rg and f[5] != rg or ins and ins not in f[6].split(", "):
                continue
            s, e = _parse_ts(f[1]), _parse_ts(f[2])
            if d0 and e <= d0 or d1 and s >= d1 + dt.timedelta(days=1):
                continue
            hits.append(r)
        off, lim = common.page_args(a, 40, 40)
        return _page(INC_HEAD, hits, off, lim, "matching incidents")

    def read_notice(a):
        nt = _world(world_dir)[3]
        k = str(a["notice_id"]).strip().upper()
        if k not in nt:
            raise KeyError(f"unknown notice {k}")
        return "notice | sent (UTC) | service | region | scope | planned start (UTC) | planned end (UTC)\n" + nt[k]

    D = T["tools"]
    S, I = {"type": "string"}, {"type": "integer"}
    return [common.tool("list_customers", D["list_customers"], {"offset": I, "limit": I}, [], list_customers),
            common.tool("read_contract", D["read_contract"], {"customer_id": S}, ["customer_id"], read_contract),
            common.tool("get_subscriptions", D["get_subscriptions"], {"customer_id": S}, ["customer_id"],
                        get_subscriptions),
            common.tool("search_incidents", D["search_incidents"],
                        {"service": S, "region": S, "instance_id": S, "start_date": S, "end_date": S, "offset": I,
                         "limit": I}, [], search_incidents),
            common.tool("read_notice", D["read_notice"], {"notice_id": S}, ["notice_id"], read_notice),
            common.tool("list_tickets", D["list_tickets"], {"customer_id": S}, ["customer_id"], list_tickets),
            common.tool("list_claims", D["list_claims"], {"customer_id": S}, ["customer_id"], list_claims),
            common.calculator_tool(D["calculator"])]


# ---------- scoring ----------
def parse_answer(answer):
    obj = common.last_json(answer, lambda o: isinstance(o, dict) and any(re.fullmatch(r"CUS\d{4}", str(k).strip().upper())
                                                                         for k in o))
    out = {}
    for k, v in (obj or {}).items():
        k = str(k).strip().upper()
        if not re.fullmatch(r"CUS\d{4}", k) or not isinstance(v, dict):
            continue
        br = v.get("breaches")
        br = [br] if isinstance(br, str) else br if isinstance(br, list) else []
        br = sorted({str(x).strip().lower() for x in br if str(x).strip().lower() not in ("", "none")})
        out[k] = {"breaches": br, "availability": common.num(v.get("availability")), "credit": common.num(v.get("credit"))}
    return out


def unit_fields(pa, g):
    return (pa["breaches"] == sorted(g["breaches"]),
            pa["availability"] is not None and abs(pa["availability"] - g["availability"]) <= 0.0015,
            pa["credit"] is not None and abs(pa["credit"] - g["credit"]) < 0.01)


def score(task, answer):
    g = common.gold_of(task)["customers"]
    p = parse_answer(answer)
    ok, part = 0, 0.0
    for cid, gc in g.items():
        if cid not in p:
            continue
        f = unit_fields(p[cid], gc)
        ok += all(f)
        part += sum(f) / 3
    n = len(g)
    return {"score": ok / n, "score_exact": float(ok == n), "score_partial": part / n, "n_correct": ok, "n_units": n}
