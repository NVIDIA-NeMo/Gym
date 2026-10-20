"""Type 51 solutions: tool-only oracle, plausible-wrong baselines, single units (check 5)."""
import json
import re
from pathlib import Path

from . import type as track

LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ERROR .* call to (\S+) failed")


class Client:
    # Tool-only access, counting characters returned so the README read-cost estimate is measured, not guessed.
    def __init__(self, world_dir):
        self.fns = {t["name"]: t["fn"] for t in track.tools(world_dir)}
        self.calls = self.chars = 0

    def __call__(self, name, **args):
        out = self.fns[name](args)
        self.calls += 1
        self.chars += len(out)
        assert not out.startswith("Error"), (name, args, out)
        return out


def all_services(c):
    names, off = [], 0
    while True:
        out = c("list_services", offset=off)
        names += [x.split(" | ")[0] for x in out.splitlines()[1:] if " | " in x]
        if "[end]" in out:
            return names
        off = int(re.search(r"offset=(\d+)", out).group(1))


def read_tickets(c, prompt):
    out = {}
    for tid in re.search(r"Tickets: (.+)", prompt).group(1).split(", "):
        text = c("read_ticket", ticket_id=tid)
        opened, resolved = re.search(r"Opened: (\S+ \S+).*Resolved: (\S+ \S+)", text).groups()
        alerting = re.search(r"Alerting service: (\S+)", text).group(1)
        out[tid] = (track.parse_time(opened, 0) - 2 * 3600, track.parse_time(resolved, 0), alerting)
    return out


def failing_near(c, cache, svc, t):
    key = (svc, t)
    if key not in cache:
        out = c("query_logs", service=svc, start=track.fmt(t - 900), end=track.fmt(t + 900), pattern=" ERROR ",
                limit=1)
        cache[key] = not out.startswith("0 lines")
    return cache[key]


def first_real_upstream(c, cache, svc, a, b):
    out = c("query_logs", service=svc, start=track.fmt(a), end=track.fmt(b), pattern="call to .* failed", limit=20)
    seen = set()
    for line in out.splitlines()[1:]:
        m = LINE.match(line)
        if not m or m.group(2) in seen:
            continue
        seen.add(m.group(2))
        if failing_near(c, cache, m.group(2), track.parse_time(m.group(1), 0)):
            return m.group(2)
    return None


def culprit(c, root, a, b):
    first = c("query_logs", service=root, start=track.fmt(a), end=track.fmt(b), pattern=" ERROR ", limit=1)
    line = first.splitlines()[1]
    t_err = track.parse_time(line[:19], 0)
    changes = []
    for row in c("list_changes", service=root).splitlines()[1:]:
        if row.startswith("chg-"):
            cid, when, _, typ = row.split(" | ")[:4]
            changes.append((track.parse_time(when, 0), cid, typ))
    if "OOMKilled" in line:
        lo = max(0, t_err - 12 * 3600)
        rows = c("query_metrics", service=root, metric="memory_mb", start=track.fmt(lo), end=track.fmt(t_err))
        pts = [(track.parse_time(r[:16], 0), float(r.split()[-1])) for r in rows.splitlines()[1:]]
        base = min(v for _, v in pts[:10])
        climb = next(t for t, v in pts if v > base * 1.04)
        step = pts[1][0] - pts[0][0]
        return max((x for x in changes if x[2] == "deploy" and x[0] <= climb + step), key=lambda x: x[0])[1]
    return max((x for x in changes if t_err - 300 <= x[0] <= t_err), key=lambda x: x[0])[1]


def oracle_answer(task, client=None):
    c = client or Client(task["world_dir"])
    tickets = read_tickets(c, task["prompt"])
    names = all_services(c)
    lines, cache = [], {}
    for tid, (a, b, alerting) in tickets.items():
        parent = {s: first_real_upstream(c, cache, s, a, b) for s in names}
        root = alerting
        while parent[root]:
            root = parent[root]
        lines.append(f"{tid} | root | {root}")
        lines.append(f"{tid} | change | {culprit(c, root, a, b)}")
        lines += [f"{tid} | edge | {p} -> {s}" for s, p in parent.items() if p]
    return "\n".join(lines)


def scan(c, names, a, b):
    # Per service in the window: ERROR count and the target of its earliest failure line (no health check).
    errs, first = {}, {}
    for s in names:
        out = c("query_logs", service=s, start=track.fmt(a), end=track.fmt(b), pattern=" ERROR ", limit=200)
        errs[s] = int(out.split()[0])
        m = next((LINE.match(x) for x in out.splitlines()[1:] if LINE.match(x)), None)
        if m:
            first[s] = m.group(2)
    return errs, first


def changes_before(c, svc, t, deploy_only=False):
    rows = []
    for row in c("list_changes", service=svc, end=track.fmt(t)).splitlines()[1:]:
        if row.startswith("chg-"):
            cid, when, _, typ = row.split(" | ")[:4]
            if not deploy_only or typ == "deploy":
                rows.append((track.parse_time(when, 0), cid))
    return rows


def answer(task, kind):
    c = Client(task["world_dir"])
    names = all_services(c)
    lines = []
    for tid, (a, b, _) in read_tickets(c, task["prompt"]).items():
        opened = a + 2 * 3600
        errs, first = scan(c, names, a, b)
        if kind == "loudest":
            root = max(errs, key=errs.get)
            cands = changes_before(c, root, opened)
            change = max(cands)[1] if cands else "none"
        else:
            cands = [(t, cid, s) for s in names if errs[s] for t, cid in changes_before(c, s, opened, True)]
            _, change, root = max(cands)
        lines += [f"{tid} | root | {root}", f"{tid} | change | {change}"]
        lines += [f"{tid} | edge | {up} -> {s}" for s, up in first.items()]
    return "\n".join(lines)



def path_walk(task):
    # Shortcut probe: right root and change from the alert-to-root walk, chain = only the walked path.
    c = Client(task["world_dir"])
    lines, cache = [], {}
    for tid, (a, b, alerting) in read_tickets(c, task["prompt"]).items():
        node = alerting
        while True:
            up = first_real_upstream(c, cache, node, a, b)
            if not up:
                break
            lines.append(f"{tid} | edge | {up} -> {node}")
            node = up
        lines += [f"{tid} | root | {node}", f"{tid} | change | {culprit(c, node, a, b)}"]
    return "\n".join(lines)


def oracle(task):
    return oracle_answer(task)


def wrong(task):
    # loudest / recent_deploy use the naive tree (no health check) as their chain; the *_nochain variants are the
    # same heuristics answering root and change only. path_walk is the known v1 shortcut (reported, not <= 0.1).
    loud, recent = answer(task, "loudest"), answer(task, "recent_deploy")
    strip = lambda ans: "\n".join(x for x in ans.splitlines() if "| edge |" not in x)
    return {"loudest": loud, "recent_deploy": recent, "loudest_nochain": strip(loud),
            "recent_deploy_nochain": strip(recent), "path_walk": path_walk(task)}


def pick_units(task, k_member, k_no, rng):
    # Every root, members (preferring ones whose earliest failure line is noise), hard NOs and unrelated bursts.
    with open(task["gold_path"]) as f:
        gold = json.load(f)["incidents"]
    w = track.load(task["world_dir"])
    units = []
    for g in gold:
        a, b = g["window"]
        units.append((g["root"], g["ticket"], "root"))
        members = [d for _, d in g["edges"]]
        tricky = []
        for m in members:
            lines = w.episode_lines(w.svc(m), a, b, errors_only=True)
            first = min(lines)[2] if lines else ""
            up = next(u for u, d in g["edges"] if d == m)
            if not first.startswith(f"call to {up} "):
                tricky.append(m)
        rest = [m for m in members if m not in tricky]
        pick = rng.sample(tricky, min(len(tricky), k_member // 2))
        pick += rng.sample(rest, min(len(rest), k_member - len(pick)))
        units += [(m, g["ticket"], "member_tricky" if m in tricky else "member") for m in pick]
        member_set = set(members) | {g["root"]}
        noisy, burst = [], []
        for s in w.svcs:
            if s["name"] in member_set:
                continue
            msgs = [m for _, _, m, _ in w.episode_lines(s, a, b, errors_only=True)]
            if any(not m.startswith("call to") for m in msgs):
                burst.append(s["name"])
            elif msgs:
                noisy.append(s["name"])
        units += [(s, g["ticket"], "no_noise") for s in rng.sample(noisy, min(len(noisy), k_no))]
        units += [(s, g["ticket"], "burst") for s in rng.sample(burst, min(len(burst), 1))]
    return units



def units(task, k, rng):
    tmpl = load_text_unit()
    out = []
    picked = pick_units(task, 2, 1, rng)
    rng.shuffle(picked)
    for svc, tid, kind in picked[:k]:
        gold = track.unit_label(task, tid, svc)
        prompt = (tmpl.replace("{n}", str(task["n_units"])).replace("{start}", track.fmt(0, False))
                  .replace("{end}", track.fmt(7 * 86400, False)).replace("{service}", svc).replace("{ticket}", tid)
                  .replace("{rules}", "\n".join(f"- {r.lstrip('- ')}" for r in track.T["rules"])))

        def check(text, gold=gold):
            got = track.parse_unit(text)
            return None if got is None else (got[0] == gold[0] and got[1].lower() == gold[1].lower())
        out.append({"id": f"{task['id']}:{tid}:{svc}", "kind": kind, "prompt": prompt, "gold": list(gold),
                    "check": check})
    return out


def load_text_unit():
    return json.loads((Path(__file__).parent / "text" / "k3_unit.json").read_text())["unit_prompt"]
