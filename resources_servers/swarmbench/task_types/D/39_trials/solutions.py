"""Type 39 solutions: tool-only oracle (reads the world through the tools, parses K3 note templates back with
common.tre, applies the trial encoding shipped in data/trials.json), plausible-wrong baselines, partial readers that skip
one trap kind each, and single units (check 5)."""
import datetime as dt
import json
import re

from .. import common
from . import type as track

T = track.T
DATE = r"\d{4}-\d{2}-\d{2}"
N = {k: [common.tre(x, **g) for x in T[k]] for k, g in {
    "n_ecog": {"ecog": r"\d"}, "n_hcg": {"result": r"positive|negative"}, "n_breastfeeding": {},
    "n_weaned": {"date": DATE}, "n_family": {"relative": r".+?", "condition": r".+?"}, "n_ruled_out": {"condition": r".+?"},
    "n_last_dose": {"drug": r".+?", "date": DATE}, "n_vaccine": {"vaccine": r".+?", "date": DATE},
    "n_surgery": {"procedure": r".+?", "category": r"major|minor", "date": DATE}, "n_weight": {"weight": r"[\d.]+ (?:kg|lb)"}}.items()}
NAME2KEY = {n: k for k, ns in track.COND.items() for n in ns}
NAME2KEY.update({n: k + "_near" for k, ns in track.NEAR_COND.items() for n in ns})
NAME2KEY.update({n: "malig" for n in track.MALIG + ["Basal cell carcinoma of the skin", "Squamous cell carcinoma of the skin",
                                                     "Carcinoma in situ of the cervix"]})
NAME2KEY.update({"Gilbert's syndrome": "gilbert", "Non-small cell lung cancer": "primary", "Breast cancer": "primary",
                 "Type 2 diabetes mellitus": "primary"})
LAB2AN = {v[0]: k for k, v in track.UNITS.items()}
PROC = {"Modified radical mastectomy": "mastectomy", "Core needle biopsy, breast": "biopsy",
        "Laparoscopic cholecystectomy": "cholecystectomy"}


def _d(s):
    return dt.date.fromisoformat(s.strip())


def _rows(text):
    return [r.split(" | ") for r in text.splitlines()[1:] if " | " in r]


def match(key, text):
    for rx in N[key]:
        m = rx.fullmatch(text)
        if m:
            return m
    return None


def read_patient(fns, pid, opts=()):
    ch = {s: fns["read_chart"]({"patient_id": pid, "section": s}) for s in track.SECTIONS[:-1]}
    notes, off = [], 0
    while True:
        out = fns["read_notes"]({"patient_id": pid, "offset": off, "limit": 20})
        notes += [r.split(" | ", 1) for r in out.splitlines()[1:] if re.match(DATE + r" \| ", r)]
        m = re.search(r"next offset (\d+)\)$", out)
        if not m:
            break
        off = int(m.group(1))
    sm = dict(re.findall(r"^([A-Za-z ]+): (.+)$", ch["summary"], re.M))
    h = sm["Height"].split()
    P = {"id": pid, "sex": "F" if sm["Sex"] == "female" else "M", "dob": _d(sm["Date of birth"]),
         "screen": _d(sm["Screening date"]), "trial": sm["Assigned trial"].split()[0], "height": (h[0], h[1]),
         "conds": [], "family": [], "ruled_out": [], "meds": [], "procs": [], "vaccines": [], "labs": [], "ecog": [],
         "weights": [], "hcg": [], "lact": [], "markers": {}, "sites": []}
    for name, status, onset, res in _rows(ch["problems"]):
        P["conds"].append({"key": NAME2KEY.get(name, "fill"), "name": name, "status": status, "onset": _d(onset),
                           "resolved": None if res == "-" else _d(res)})
    if ch["oncology"].startswith("Primary"):
        on = dict(re.findall(r"^(.+?): (.+)$", ch["oncology"], re.M))
        P["hist"], P["stage"] = on["Histology"], on["Stage (AJCC 8th edition)"]
        P["markers"] = {k: on[k] for k in ("EGFR", "ALK", "KRAS") if k in on}
        P["sites"] = [] if on["Metastatic sites"] == "none" else on["Metastatic sites"].split(", ")
        P["dx_date"] = _d(on["Diagnosed"])
    for drug, ind, start, stop in _rows(ch["medications"]):
        P["meds"].append({"drug": drug, "ind": ind, "start": _d(start), "stop": None if stop in ("-", "see notes") else _d(stop),
                          "notes": stop == "see notes"})
    for start, end, label, cat, intent, ind in _rows(ch["procedures"]):
        nm = "radiotherapy" if "adiotherapy" in label else PROC.get(label, label[0].lower() + label[1:])
        P["procs"].append({"name": nm, "kind": cat.replace(" ", "_"), "start": _d(start), "date": _d(end),
                           "intent": "" if intent == "-" else intent, "ind": "" if ind == "-" else ind})
    for d, name in _rows(ch["immunizations"]):
        P["vaccines"].append({"name": name, "date": _d(d)})
    for d, test, val, unit, uln in _rows(ch["labs"]):
        if test.startswith("hCG"):
            P["hcg"].append({"date": _d(d), "res": val})
        else:
            r = {"date": _d(d), "an": LAB2AN[test], "val": val, "unit": unit}
            if uln != "-":
                r["uln"] = uln
            P["labs"].append(r)
    for d, w in _rows(ch["vitals"]):
        P["weights"].append({"date": _d(d), "val": w.split()[0], "unit": w.split()[1]})
    for d, e in _rows(ch["performance"]):
        P["ecog"].append({"date": _d(d), "val": int(e)})
    for d, text in notes:
        d = _d(d)
        if "no_notes" in opts:
            continue
        if (m := match("n_ecog", text)):
            P["ecog"].append({"date": d, "val": int(m["ecog"])})
        elif (m := match("n_hcg", text)):
            P["hcg"].append({"date": d, "res": m["result"]})
        elif match("n_breastfeeding", text):
            P["lact"].append({"date": d, "state": "breastfeeding"})
        elif (m := match("n_weaned", text)):
            P["lact"].append({"date": _d(m["date"]), "state": "weaned"})
        elif (m := match("n_family", text)):
            P["family"].append((NAME2KEY.get(m["condition"], "fill"), m["relative"], m["condition"]))
        elif (m := match("n_ruled_out", text)):
            P["ruled_out"].append((m["condition"], d))
        elif (m := match("n_last_dose", text)):
            for x in P["meds"]:
                if x["drug"] == m["drug"] and x["notes"]:
                    x["stop"] = _d(m["date"])
        elif (m := match("n_vaccine", text)):
            P["vaccines"].append({"name": m["vaccine"], "date": _d(m["date"])})
        elif (m := match("n_surgery", text)):
            P["procs"].append({"name": m["procedure"], "kind": m["category"] + "_surgery", "start": None,
                               "date": _d(m["date"]), "intent": "",
                               "ind": {**track.MAJOR, **track.MINOR}.get(m["procedure"], "")})
        elif (m := match("n_weight", text)):
            v, u = m["weight"].split()
            P["weights"].append({"date": d, "val": v, "unit": u})
    if "dx_date" not in P:
        P["dx_date"] = next(c["onset"] for c in P["conds"] if c["key"] == "primary")
    return P


def read_world(task, opts=()):
    fns = {t["name"]: t["fn"] for t in track.tools(task["world_dir"])}
    rows, off = [], 0
    while True:
        out = fns["list_patients"]({"offset": off, "limit": 100})
        rows += [r.split(" | ")[0] for r in out.splitlines()[1:] if re.match(r"P\d{4} \|", r)]
        m = re.search(r"next offset (\d+)\)$", out)
        if not m:
            break
        off = int(m.group(1))
    return [read_patient(fns, pid, opts) for pid in rows]


def answer_of(pats, fn=None):
    out = {}
    for P in pats:
        f = (fn or track.failed_ids)(P, track.BY_NCT[P["trial"]])
        out[P["id"]] = {"eligible": not f, "failed": f}
    return "```json\n" + json.dumps(out) + "\n```"


def oracle(task):
    return answer_of(read_world(task))


def _mut(P, how):
    Q = json.loads(json.dumps(P, default=str))
    for k in ("dob", "screen", "dx_date"):
        Q[k] = _d(Q[k])
    for key in ("conds", "meds", "procs", "vaccines", "labs", "ecog", "weights", "hcg", "lact"):
        for r in Q[key]:
            for f in ("date", "onset", "resolved", "start", "stop"):
                if isinstance(r.get(f), str) and re.fullmatch(DATE, r[f]):
                    r[f] = _d(r[f])
    Q["family"] = [tuple(x) for x in Q["family"]]
    Q["ruled_out"] = [(c, _d(d)) for c, d in Q["ruled_out"]]
    how(Q)
    return Q


def wrong(task):
    pats = read_world(task)
    text = {P["id"]: json.dumps(P, default=str).lower() for P in pats}

    def regex(P, trial):  # inclusion assumed met; an exclusion applies if two of its content words appear in the chart
        out = []
        for c in trial["criteria"]:
            if c["kind"] != "exc":
                continue
            words = {w for w in re.findall(r"[a-z]{5,}", c["text"].lower())} - {"within", "prior", "study", "participant",
                                                                               "history", "patients", "first", "known"}
            if sum(w in text[P["id"]] for w in words) >= 2:
                out.append(c["id"])
        return out

    def oldest(Q):  # uses the first recorded value instead of the most recent one
        for key in ("labs", "ecog", "weights", "hcg", "lact"):
            for r in Q[key]:
                r["date"] = Q["screen"] - (r["date"] - (Q["screen"] - dt.timedelta(days=4000)))

    def no_units(Q):  # compares numbers without converting units and ages by calendar year
        for r in Q["labs"]:
            r["unit"] = track.UNITS[r["an"]][1]
        Q["height"] = (Q["height"][0], "cm")
        for w in Q["weights"]:
            w["unit"] = "kg"
        Q["dob"] = Q["dob"].replace(month=1, day=1)

    def history_any(Q):  # family history, ruled-out and near-name conditions all count as the patient's own
        for key, rel, cond in Q["family"]:
            Q["conds"].append({"key": key, "name": cond, "status": "active", "onset": Q["screen"], "resolved": None})
        for c in Q["conds"]:
            c["key"] = c["key"].replace("_near", "")
            c["status"] = "active"
        for cond, d in Q["ruled_out"]:
            if cond == "brain metastases":
                Q["sites"] = Q["sites"] + ["brain"]
            elif cond in NAME2KEY:
                Q["conds"].append({"key": NAME2KEY[cond], "name": cond, "status": "active", "onset": d, "resolved": None})

    def any_drug(Q):  # every drug or vaccine given in a window counts (no drug-class or route judgment)
        for m in Q["meds"]:
            track.CATS[m["drug"]] = set(track.DRUGS)
        for v in Q["vaccines"]:
            v["name"] = track.LIVE[0]

    import copy
    saved = copy.deepcopy(track.CATS)
    res = {"regex": answer_of(pats, regex), "all_eligible": answer_of(pats, lambda P, t: []),
           "first_values": answer_of([_mut(P, oldest) for P in pats]),
           "no_unit_conversion": answer_of([_mut(P, no_units) for P in pats]),
           "partial_history_any": answer_of([_mut(P, history_any) for P in pats]),
           "partial_skip_notes": answer_of(read_world(task, ("no_notes",)))}
    def careless(Q):
        for f in (oldest, no_units, history_any, any_drug):
            f(Q)
    try:
        res["partial_any_drug"] = answer_of([_mut(P, any_drug) for P in pats])
        res["careless_reader"] = answer_of([_mut(P, careless) for P in read_world(task, ("no_notes",))])
    finally:
        track.CATS.clear()
        track.CATS.update(saved)
    return res


def units(task, k, rng):
    g = common.gold_of(task)["patients"]
    ids = sorted(g)
    rng.shuffle(ids)
    out = []
    for pid in ids[:k]:
        ga = g[pid]
        ex = json.dumps({pid: {"eligible": False, "failed": ["I1", "E2"]}})
        prompt = "\n\n".join([common.CANARY, common.fill(T["unit_prompt"], patient=pid, trial=ga["trial"]),
                              track.rules_text(ex)])

        def check(text, pid=pid, ga=ga):
            pa = track.parse_answer(text).get(pid)
            if pa is None:
                return None
            return track.unit_ok(pa, ga)
        kind = "eligible" if ga["eligible"] else f"fail{len(ga['failed'])}"
        out.append({"id": f"{task['id']}:{pid}", "kind": kind, "prompt": prompt, "gold": ga, "check": check})
    return out
