"""Type 60 solutions: tool-only oracle (parses specs with regexes built from the K3 templates), plausible-wrong
baselines, single units (check 5)."""
import json
import random
import re

from . import type as track

T = track.T
NUM = r"[\d,]+"
UNIT = r"req/(?:s|min)"
SID = r"S\d{4}"
ANY = r".+?"


def tre(tmpl, **groups):
    """Regex for a filled K3 template: placeholders become (named) groups, everything else is literal."""
    rx = re.escape(tmpl)
    for name, pat in groups.items():
        named = name[0] != "_"
        rx = rx.replace(re.escape("{" + name.lstrip("_") + "}"), f"(?P<{name}>{pat})" if named else f"(?:{pat})", 1)
    return re.compile(rx)


def _i(s):
    return int(s.replace(",", ""))


ACTS = [(k, tre(T["act_" + k], c=NUM, _unit=UNIT)) for k in ("add", "sub", "mul")]
COND_OWN = tre(T["cond_own"], ref=SID, _ref_unit=UNIT, thr=NUM, then=ANY, **{"else": ANY})
COND_CONV = tre(T["cond_conv"], ref=SID, _u=UNIT, thr=NUM, _u2=UNIT, then=ANY, **{"else": ANY})
EDGE = tre(T["edge_capacity"], shown=NUM, mu=UNIT, _u=UNIT)
RERUN = tre(T["rerun"], new=NUM, mu=UNIT)
STEP4 = tre(T["step4"], fac=r"\d+")
CAP = tre(T["step5_cap"], cap=NUM, _u=UNIT, _cap2=NUM)
A_FAC = tre(T["amend_factor"], _old=r"\d+", new=r"\d+")
A_W = tre(T["amend_weight"], id=SID, _old=r"\d+", new=r"\d+")
A_THR = tre(T["amend_threshold"], _old=NUM, new=NUM)
A_REP = tre(T["amend_replace"], old_id=SID, new_id=SID, unit=UNIT, w=r"\d+|n/a", _old_id2=SID)
BOOT = tre(T["bootstrap"], _id=SID, default=NUM, _u=UNIT)
SNAP = tre(T["snapshot"], snapshot=NUM, _u=UNIT)


def _action(t):
    for kind, rx in ACTS:
        m = rx.fullmatch(t.strip())
        if m:
            return (kind, _i(m.group("c")))
    raise ValueError(f"unparsed action {t!r}")


def parse_spec(text, careless=None):
    # careless="log" skips the change log; careless="units" skips unit conversions (to measure each trap).
    unit = re.search(r"Published unit: (req/s|req/min)", text).group(1)
    node = {"unit": unit, "inputs": [], "op": None, "cond": None, "factor": 100, "cap": None, "measured": None,
            "default": _i(BOOT.search(text).group("default")), "snapshot": _i(SNAP.search(text).group("snapshot"))}
    log = re.findall(r"^- \S+ \(adopted\): (.*)$", text, re.M) if careless != "log" else []
    if T["edge_inputs"] in text:
        m = EDGE.search(text)
        v, u = _i(m.group("shown")), m.group("mu")
        for e in log:
            if r := RERUN.fullmatch(e):
                v, u = _i(r.group("new")), r.group("mu")
        node["measured"] = int(track._conv(v, u, unit))
        return node
    for sid, u, w in re.findall(r"^\| \d+ \| (S\d{4}) \| (req/s|req/min) \| (\d+|n/a) \|$", text, re.M):
        node["inputs"].append({"id": sid, "unit": u if careless != "units" else unit, "w": int(w) if w != "n/a" else 1})
    node["op"] = next(op for op in ("sum", "max", "min") if T["step2_" + op] in text)
    for on, rx in (("own", COND_OWN), ("converted", COND_CONV)):
        m = next((rx.fullmatch(line) for line in text.splitlines() if rx.fullmatch(line)), None)
        if m:
            node["cond"] = {"ref": m.group("ref"), "on": on, "thr": _i(m.group("thr")),
                            "then": _action(m.group("then")), "else": _action(m.group("else"))}
    node["factor"] = int(STEP4.search(text).group("fac"))
    m = CAP.search(text)
    node["cap"] = _i(m.group("cap")) if m else None
    for e in log:
        if m := A_FAC.fullmatch(e):
            node["factor"] = int(m.group("new"))
        elif m := A_W.fullmatch(e):
            next(i for i in node["inputs"] if i["id"] == m.group("id"))["w"] = int(m.group("new"))
        elif m := A_THR.fullmatch(e):
            node["cond"]["thr"] = _i(m.group("new"))
        elif m := A_REP.fullmatch(e):
            i = next(i for i in node["inputs"] if i["id"] == m.group("old_id"))
            i.update(id=m.group("new_id"), unit=m.group("unit") if careless != "units" else unit,
                     w=int(m.group("w")) if m.group("w") != "n/a" else 1)
    return node


def _fns(task):
    return {t["name"]: t["fn"] for t in track.tools(task["world_dir"])}


def targets(task):
    # the task line is the second-to-last prompt paragraph and ends with the target ids
    return re.findall(r"S\d{4}", task["prompt"].rsplit("\n\n", 2)[-2].rsplit(":", 1)[-1])


def all_ids(fns):
    ids, off = [], 0
    while True:
        page = fns["list_services"]({"offset": off, "limit": 100})
        ids += re.findall(r"^(S\d{4}) \|", page, re.M)
        if "[end]" in page:
            return ids
        off += 100


def full_spec(fns, sid):
    text, off = "", 0
    while True:
        chunk = fns["read_spec"]({"service_id": sid, "offset": off})
        m = re.search(r"\n\[chars \d+-(\d+) of \d+; continue with offset=\d+\]$", chunk)
        if not m:
            return text + chunk
        text += chunk[:m.start()]
        off = int(m.group(1))


def solve_all(task, careless=None):
    fns = _fns(task)
    specs = {s: parse_spec(full_spec(fns, s), careless) for s in all_ids(fns)}
    vals = {}

    def solve(s):
        stack = [s]
        while stack:
            x = stack[-1]
            todo = [i["id"] for i in specs[x]["inputs"] if i["id"] not in vals]
            if todo:
                stack += todo
                continue
            stack.pop()
            vals.setdefault(x, track.evaluate(specs[x], vals))
        return vals[s]
    return {t: solve(t) for t in targets(task)}, specs, vals


def oracle(task):
    return json.dumps(solve_all(task)[0])


def wrong(task):
    fns = _fns(task)
    ts = targets(task)
    specs = {t: parse_spec(full_spec(fns, t)) for t in ts}
    snapshot = {t: specs[t]["snapshot"] for t in ts}
    defaults = {}
    for t in ts:
        par = {i["id"]: parse_spec(full_spec(fns, i["id"]))["default"] for i in specs[t]["inputs"]}
        defaults[t] = track.evaluate(specs[t], par)
    return {"q2_snapshot": json.dumps(snapshot), "ignore_dependencies": json.dumps(defaults),
            "skip_changelog": json.dumps(solve_all(task, "log")[0]),
            "skip_units": json.dumps(solve_all(task, "units")[0])}


def units(task, k, rng):
    gold = json.loads(open(task["gold_path"]).read())
    vals = gold["values"]
    nodes = [x for x in gold["services"] if x["inputs"]]
    # prefer the trap-heavy units: amendments, conditions, mixed units
    nodes.sort(key=lambda x: -(bool(x["amend"]) + bool(x["cond"]) + len({i["unit"] for i in x["inputs"]} | {x["unit"]})))
    pick = nodes[:max(k * 2, k)]
    rng.shuffle(pick)
    out = []
    for x in pick[:k]:
        sid = x["id"]
        fns = _fns(task)
        shown = re.findall(r"^\| \d+ \| (S\d{4}) \|", full_spec(fns, sid), re.M)
        ids = sorted(set(shown) | {i["id"] for i in x["inputs"]})
        inputs = ", ".join(f"{i} = {vals[i]:,}" for i in ids)
        prompt = T["unit_prompt"].replace("{sid2}", sid).replace("{sid}", sid).replace("{inputs}", inputs)
        g = vals[sid]

        def check(text, g=g, sid=sid):
            m = re.findall(rf"UNIT\W*\|\W*{sid}\W*\|\W*(-?[\d,]+(?:\.\d+)?)", text or "")
            if not m:
                return None
            v = float(m[-1].replace(",", ""))
            return abs(v - g) <= max(1, track.REL_TOL * g)
        kind = "amend" if x["amend"] else "cond" if x["cond"] else "plain"
        out.append({"id": f"{task['id']}:{sid}", "kind": kind, "prompt": prompt, "gold": g, "check": check})
    return out
