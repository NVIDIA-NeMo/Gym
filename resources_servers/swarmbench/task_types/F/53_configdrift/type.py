"""Type 53 configdrift: find which hosts deviate from their role baseline, which keys, and why (unapproved, expired
exception, value differs from the approved exception, or key missing). Procedural (exact gold needs simulation).
Unit = one host. Score = F1 over (host, key, reason)."""
import json
import random
import re
import threading
from pathlib import Path

from .. import common

TYPE_ID = "53_configdrift"
TIERS = {"easy": (64, 1800), "medium": (256, 1200), "hard": (1024, 1200)}
T = common.load_text(Path(__file__).parent)
AUDIT_DATE = "2026-06-30"
REASONS = ["UNAPPROVED", "EXPIRED", "MISMATCH", "MISSING"]
KEYS = {  # key -> (choices for baselines, drift values)
    "ssh.permit_root_login": (["no", "prohibit-password"], ["yes"]),
    "ssh.password_auth": (["no"], ["yes"]),
    "ssh.max_auth_tries": (["3", "4"], ["6", "10"]),
    "ssh.client_alive_interval": (["300", "600"], ["0", "3600"]),
    "kernel.ip_forward": (["0"], ["1"]),
    "kernel.swappiness": (["10", "30", "60"], ["1", "100"]),
    "kernel.somaxconn": (["1024", "4096"], ["128", "65535"]),
    "kernel.file_max": (["1048576", "2097152"], ["65536"]),
    "app.log_level": (["info", "warn"], ["debug", "trace"]),
    "app.max_connections": (["200", "500", "1000"], ["50", "5000"]),
    "app.worker_threads": (["8", "16", "32"], ["2", "64"]),
    "app.tls_min_version": (["1.2", "1.3"], ["1.0", "1.1"]),
    "app.metrics_enabled": (["true"], ["false"]),
    "app.cache_ttl_s": (["60", "300"], ["0", "86400"]),
    "app.request_timeout_ms": (["2000", "5000"], ["500", "60000"]),
    "ntp.server": (["ntp1.corp.internal", "ntp2.corp.internal"], ["pool.ntp.org", "time.example.net"]),
    "audit.enabled": (["true"], ["false"]),
    "backup.schedule": (["daily", "hourly"], ["weekly", "never"]),
}
ROLES = ["web", "api", "db", "cache", "worker", "gateway", "batch", "search"]


def _date(rng, lo, hi):
    return f"2026-{rng.randint(lo, hi):02d}-{rng.randint(1, 28):02d}"


def baselines(rng):
    out = {}
    for role in ROLES:
        keys = sorted(rng.sample(sorted(KEYS), 12))
        out[role] = {k: rng.choice(KEYS[k][0]) for k in keys}
    return out


def render(lines_kv, header, rng, junk_keys=()):
    """A config file: a K3 header comment, key = value lines, comments and blank lines."""
    out = [f"# {header.lstrip('# ')}", ""]
    for kind, k, v in lines_kv:
        if kind == "comment":
            out.append(f"# {k} = {v}")
        else:
            out.append(f"{k} = {v}")
        if rng.random() < 0.15:
            out.append("")
    return "\n".join(out) + "\n"


def build_host(name, role, base, rng, exc_pool):
    """Returns files {path: text}, effective config, exceptions for this host, gold findings."""
    eff = dict(base)
    plan = {}  # key -> final effective value (or None for missing)
    for k in base:
        r = rng.random()
        if r < 0.10:
            plan[k] = rng.choice(KEYS[k][1])
        elif r < 0.13:
            plan[k] = None
    # base file holds the baseline, possibly with a drifted value that a later file restores (no drift)
    base_lines, d_lines, host_lines, disabled = [], [], [], []
    for k, v in base.items():
        if plan.get(k, v) is None:
            if rng.random() < 0.5:
                base_lines.append(("comment", k, v))  # commented out = missing
            continue
        if k in plan and rng.random() < 0.5:
            base_lines.append(("kv", k, v))
            (d_lines if rng.random() < 0.5 else host_lines).append(("kv", k, plan[k]))
        elif k in plan:
            base_lines.append(("kv", k, plan[k]))
        else:
            if rng.random() < 0.25:  # decoy: drifted in base, restored later
                base_lines.append(("kv", k, rng.choice(KEYS[k][1])))
                host_lines.append(("kv", k, v))
            else:
                base_lines.append(("kv", k, v))
            if rng.random() < 0.2:  # decoy: drift that only appears in a commented line or a disabled file
                if rng.random() < 0.5:
                    d_lines.append(("comment", k, rng.choice(KEYS[k][1])))
                else:
                    disabled.append(("kv", k, rng.choice(KEYS[k][1])))
    rng.shuffle(d_lines)
    for kind, k, v in d_lines:
        if kind == "kv":
            eff[k] = v
    for kind, k, v in host_lines:
        if kind == "kv":
            eff[k] = v
    files = {"/etc/fleet/base.conf": render(base_lines, rng.choice(T["file_headers"]), rng),
             f"/etc/fleet/conf.d/50-{role}.conf": render([x for x in d_lines if x[0] == "kv"] +
                                                         [x for x in d_lines if x[0] == "comment"],
                                                         rng.choice(T["file_headers"]), rng),
             f"/etc/fleet/conf.d/90-host-{name}.conf": render(host_lines, rng.choice(T["file_headers"]), rng)}
    if disabled:
        files["/etc/fleet/conf.d/70-experiment.conf.disabled"] = render(disabled, rng.choice(T["file_headers"]), rng)
    # recompute the effective config exactly as the rules say (lexical order of *.conf, last wins)
    eff = {}
    for path in sorted(p for p in files if p.endswith(".conf")):
        for line in files[path].splitlines():
            m = re.match(r"^([\w.]+) = (.*)$", line)
            if m:
                eff[m.group(1)] = m.group(2)
    # exceptions: some drifted keys get valid ones (no finding), expired ones, or ones for another value
    excs, gold = [], []
    for k, v in base.items():
        cur = eff.get(k)
        if cur == v:
            if rng.random() < 0.04:  # decoy: an exception that is not needed
                excs.append({"target": name, "key": k, "value": rng.choice(KEYS[k][1]), "expires": _date(rng, 7, 12)})
            continue
        if cur is None:
            gold.append((name, k, "MISSING"))
            continue
        r = rng.random()
        target = name if rng.random() < 0.75 else f"role:{role}"
        if r < 0.35:
            excs.append({"target": target, "key": k, "value": cur, "expires": _date(rng, 7, 12)})  # covered
        elif r < 0.6:
            excs.append({"target": target, "key": k, "value": cur, "expires": _date(rng, 1, 6)})
            gold.append((name, k, "EXPIRED"))
        elif r < 0.8:
            other = rng.choice([x for x in KEYS[k][0] + KEYS[k][1] if x != cur])
            excs.append({"target": target, "key": k, "value": other, "expires": _date(rng, 7, 12)})
            gold.append((name, k, "MISMATCH"))
        else:
            if rng.random() < 0.3:  # decoy: the same exception exists for a different host
                excs.append({"target": rng.choice(exc_pool), "key": k, "value": cur, "expires": _date(rng, 7, 12)})
            gold.append((name, k, "UNAPPROVED"))
    for e in excs:
        e["reason"] = rng.choice(T["justifications"])
    return files, excs, gold


def generate(n, seed, out_dir):
    rng = random.Random(f"53:{seed}:{n}")
    base = baselines(rng)
    roles = [rng.choice(ROLES) for _ in range(n)]
    names = [f"{r}-{rng.choice(['use1', 'usw2', 'euw1', 'aps1'])}-{i:04d}" for i, r in enumerate(roles)]
    hosts, files, excs, gold = [], {}, [], []
    for name, role in zip(names, roles):
        f, e, g = build_host(name, role, base[role], rng, names)
        hosts.append([name, role])
        files[name] = f
        excs += e
        gold += g
    # role-wide exceptions that match a host's drift are only valid for hosts of that role: already ensured above
    rng.shuffle(excs)
    for i, e in enumerate(excs):
        e["id"] = f"EXC-{1000 + i}"
    # Gold is the rule-based audit itself: role-wide exceptions planted for one host also reach its role peers, so the
    # generator's per-host bookkeeping is only a plan.
    gold = audit(hosts, files, base, excs)
    prompt = "\n\n".join([T["intro"].replace("{n}", str(n)).replace("{date}", AUDIT_DATE),
                          "\n".join(f"- {x.lstrip('- ')}" for x in T["rules"]),
                          "Tools:\n" + "\n".join(f"- {k}: {v}" for k, v in T["tools"].items()),
                          T["answer"], T["score"]])
    return common.finish(TYPE_ID, n, seed, out_dir, prompt, {"findings": sorted(gold)}, TIERS,
                         files={"hosts.json": json.dumps(hosts), "files.json": json.dumps(files),
                                "baselines.json": json.dumps(base), "exceptions.json": json.dumps(excs)})


def effective(host_files):
    eff = {}
    for path in sorted(p for p in host_files if p.endswith(".conf")):
        for line in host_files[path].splitlines():
            m = re.match(r"^\s*([\w.]+)\s*=\s*(.*?)\s*$", line)
            if m and not line.lstrip().startswith("#"):
                eff[m.group(1)] = m.group(2)
    return eff


def audit(hosts, files, base, excs):
    out = []
    for name, role in hosts:
        eff = effective(files[name])
        mine = [e for e in excs if e["target"] in (name, f"role:{role}")]
        for k, v in base[role].items():
            cur = eff.get(k)
            if cur == v:
                continue
            if cur is None:
                out.append((name, k, "MISSING"))
                continue
            es = [e for e in mine if e["key"] == k]
            valid = [e for e in es if e["expires"] >= AUDIT_DATE]
            if any(e["value"] == cur for e in valid):
                continue
            if any(e["value"] == cur for e in es):
                out.append((name, k, "EXPIRED"))
            elif valid:
                out.append((name, k, "MISMATCH"))
            else:
                out.append((name, k, "UNAPPROVED"))
    return out


_worlds, _wlock = {}, threading.Lock()


def _load(world_dir):
    with _wlock:
        if world_dir not in _worlds:
            d = Path(world_dir)
            _worlds[world_dir] = {k: json.loads((d / f"{k}.json").read_text())
                                  for k in ("hosts", "files", "baselines", "exceptions")}
        return _worlds[world_dir]


def tools(world_dir):
    def host_of(a):
        w = _load(world_dir)
        h = str(a.get("host", "")).strip()
        if h not in w["files"]:
            raise KeyError(f"unknown host {a.get('host')!r}; use list_hosts")
        return w, h

    def list_hosts(a):
        w = _load(world_dir)
        off, lim = common.page_args(a, 200, 200)
        rows = [f"{h} | {r}" for h, r in w["hosts"][off:off + lim]]
        tail = f"[next: offset={off + lim}]" if off + lim < len(w["hosts"]) else "[end]"
        return f"{len(w['hosts'])} hosts; showing {off}-{off + len(rows) - 1}\n" + "\n".join(rows) + "\n" + tail

    def list_files(a):
        w, h = host_of(a)
        return "\n".join(sorted(w["files"][h]))

    def read_file(a):
        w, h = host_of(a)
        path = str(a.get("path", "")).strip()
        if path not in w["files"][h]:
            raise KeyError(f"no file {path!r} on {h}; use list_files")
        return w["files"][h][path]

    def get_baseline(a):
        w = _load(world_dir)
        role = str(a.get("role", "")).strip()
        if role not in w["baselines"]:
            raise KeyError(f"unknown role {role!r}; one of {sorted(w['baselines'])}")
        return "\n".join(f"{k} = {v}" for k, v in w["baselines"][role].items())

    def get_exceptions(a):
        w, h = host_of(a)
        role = dict(w["hosts"])[h]
        es = [e for e in w["exceptions"] if e["target"] in (h, f"role:{role}")]
        return (f"{len(es)} exceptions apply to {h} (host or role:{role})\n" +
                "\n".join(f"{e['id']} | target {e['target']} | {e['key']} = {e['value']} | expires {e['expires']} | "
                          f"{e['reason']}" for e in es))

    S, I = {"type": "string"}, {"type": "integer"}
    return [common.tool("list_hosts", T["tools"]["list_hosts"], {"offset": I, "limit": I}, [], list_hosts),
            common.tool("list_files", T["tools"]["list_files"], {"host": S}, ["host"], list_files),
            common.tool("read_file", T["tools"]["read_file"], {"host": S, "path": S}, ["host", "path"], read_file),
            common.tool("get_baseline", T["tools"]["get_baseline"], {"role": S}, ["role"], get_baseline),
            common.tool("get_exceptions", T["tools"]["get_exceptions"], {"host": S}, ["host"], get_exceptions)]


LINE = re.compile(r"^([\w.-]+)\s*\|\s*([\w.]+)\s*\|\s*([A-Z]+)$")


def parse_answer(answer):
    out = set()
    for line in (answer or "").splitlines():
        m = LINE.match(common.clean_line(line).strip("|").strip())
        if m and m.group(3) in REASONS:
            out.add((m.group(1), m.group(2), m.group(3)))
    return out


def score(task, answer):
    gold = {tuple(x) for x in common.gold_of(task)["findings"]}
    pred = parse_answer(answer)
    f, p, r = common.f1(pred, gold)
    keyonly = common.f1({x[:2] for x in pred}, {x[:2] for x in gold})[0]
    return {"score": f, "score_partial": f, "score_exact": float(pred == gold), "precision": p, "recall": r,
            "drift_key_f1": keyonly, "n_gold": len(gold), "n_pred": len(pred)}
