"""Type 53 solutions: tool-only oracle and plausible-wrong audits."""
import json
import re

from . import type as track


def _fns(task):
    return {t["name"]: t["fn"] for t in track.tools(task["world_dir"])}


def read_all(task):
    fns, hosts, off = _fns(task), [], 0
    while True:
        page = fns["list_hosts"]({"offset": off})
        hosts += [tuple(x.split(" | ")) for x in page.splitlines()[1:-1]]
        if "[end]" in page:
            break
        off += 200
    files, excs, base = {}, [], {}
    for h, role in hosts:
        files[h] = {p: fns["read_file"]({"host": h, "path": p}) for p in fns["list_files"]({"host": h}).splitlines()}
        for row in fns["get_exceptions"]({"host": h}).splitlines()[1:]:
            eid, target, kv, exp, _ = row.split(" | ", 4)
            k, v = kv.split(" = ", 1)
            excs.append({"id": eid, "target": target.removeprefix("target "), "key": k, "value": v,
                         "expires": exp.removeprefix("expires ")})
        if role not in base:
            base[role] = dict(line.split(" = ", 1) for line in fns["get_baseline"]({"role": role}).splitlines())
    uniq = list({e["id"]: e for e in excs}.values())
    return hosts, files, base, uniq


def lines(found):
    return "\n".join(" | ".join(x) for x in sorted(found))


def oracle(task):
    return lines(track.audit(*read_all(task)))


def wrong(task):
    hosts, files, base, excs = read_all(task)
    gold = track.audit(hosts, files, base, excs)
    # base.conf only (ignores conf.d overrides)
    base_only = track.audit(hosts, {h: {p: t for p, t in f.items() if p.endswith("base.conf")} for h, f in files.items()},
                            base, excs)
    # every deviation reported as UNAPPROVED, exceptions ignored
    no_exc = track.audit(hosts, files, base, [])
    # disabled files and commented lines read as active
    loose = {h: {p.replace(".conf.disabled", ".conf"): t.replace("\n# ", "\n") for p, t in f.items()}
             for h, f in files.items()}
    careless = track.audit(hosts, loose, base, excs)
    # right keys, reason always UNAPPROVED
    reason_blind = [(h, k, "UNAPPROVED") for h, k, _ in gold]
    names = [h for h, _ in hosts]
    shifted = [(names[(names.index(h) + 1) % len(names)], k, r) for h, k, r in gold]  # off-by-one host attribution
    any_value = []  # every baseline key that has a non-baseline value on any line of any file, as UNAPPROVED
    for h, role in hosts:
        text = "\n".join(files[h].values())
        for k, v in base[role].items():
            if re.search(rf"{re.escape(k)} = (?!{re.escape(v)}$)", text, re.M):
                any_value.append((h, k, "UNAPPROVED"))
    return {"hosts_shifted": lines(shifted), "grep_any_other_value": lines(any_value), "base_conf_only": lines(base_only), "ignore_exceptions": lines(no_exc), "read_comments_and_disabled":
            lines(careless), "reason_always_unapproved": lines(reason_blind)}


def units(task, k, rng):
    gold = {}
    for h, key, r in json.loads(open(task["gold_path"]).read())["findings"]:
        gold.setdefault(h, set()).add((h, key, r))
    hosts = [h for h, _ in json.loads(open(task["world_dir"] + "/hosts.json").read())]
    tricky = [h for h in hosts if any(r in ("EXPIRED", "MISMATCH") for _, _, r in gold.get(h, ()))]
    pick = rng.sample(tricky, min(len(tricky), k // 2))
    pick += rng.sample([h for h in hosts if h not in pick], k - len(pick))
    rules = "\n".join(f"- {x.lstrip('- ')}" for x in track.T["rules"])
    out = []
    for h in pick:
        g = gold.get(h, set())
        prompt = (track.T["unit_prompt"].replace("{host2}", h).replace("{host}", h).replace("{rules}", rules)
                  .replace("{date}", track.AUDIT_DATE))

        def check(text, g=g, h=h):
            if not re.search(rf"UNIT DONE\W*{re.escape(h)}", text or ""):
                return None
            return {x for x in track.parse_answer(text) if x[0] == h} == g
        out.append({"id": f"{task['id']}:{h}", "kind": "exception_case" if h in tricky else "plain", "prompt": prompt,
                    "gold": sorted(g), "check": check})
    return out
