"""Type 39 trials: clinical-trial eligibility screening of N synthetic patients against real ClinicalTrials.gov criteria.

Unit = one patient screened against its assigned trial. The criterion lines are verbatim registry text (data/trials.json,
see README); their encoding (trialspec.py) and the patient facts are code. Notes and all task prose are K3 templates.
Gold = the encoded predicates evaluated on the generated chart (exact, Decimal arithmetic on the displayed values)."""
import datetime as dt
import json
import random
import re
import threading
from decimal import Decimal as Dm
from pathlib import Path

from .. import common

TYPE_ID = "39_trials"
TIERS = {"easy": (8, 1800), "medium": (24, 1200), "hard": (80, 1200)}  # K3 ~115 s/unit in the 1-unit loop
HERE = Path(__file__).parent
T = common.load_text(HERE)
TRIALS = json.loads((HERE / "data" / "trials.json").read_text())
BY_NCT = {t["nct"]: t for t in TRIALS}
D = dt.date
# analyte: (label, canonical unit, alternate unit, factor alt = canon * f (None = HbA1c formula), has ULN)
UNITS = {"anc": ("Neutrophils, absolute", "x10^9/L", "/uL", 1000, False),
         "plt": ("Platelets", "x10^9/L", "/uL", 1000, False),
         "lym": ("Lymphocytes, absolute", "x10^9/L", "/uL", 1000, False),
         "hgb": ("Hemoglobin", "g/L", "g/dL", Dm("0.1"), False),
         "creat": ("Creatinine", "mg/dL", "umol/L", Dm("88.4"), True),
         "crcl": ("Creatinine clearance", "mL/min", None, None, False),
         "egfr": ("eGFR", "mL/min/1.73m2", None, None, False),
         "hba1c": ("HbA1c", "%", "mmol/mol", None, False),
         "k": ("Potassium", "mmol/L", "mEq/L", 1, False),
         "lvef": ("LVEF (echocardiogram)", "%", None, None, False),
         "bili": ("Total bilirubin", "mg/dL", "umol/L", Dm("17.1"), True),
         "ast": ("AST", "U/L", None, None, True), "alt": ("ALT", "U/L", None, None, True),
         "alp": ("Alkaline phosphatase", "U/L", None, None, True)}
DEC = {"anc": 1, "plt": 0, "lym": 1, "hgb": 0, "creat": 2, "crcl": 0, "egfr": 0, "hba1c": 1, "k": 1, "lvef": 0,
       "bili": 1, "ast": 0, "alt": 0, "alp": 0}
ULNS = {"creat": ["1.00", "1.10", "1.20", "1.30"], "bili": ["1.0", "1.2"], "ast": ["35", "40"], "alt": ["40", "45", "56"],
        "alp": ["120", "130", "147"]}
NORMAL = {"anc": (2.5, 6), "plt": (180, 380), "lym": (1.6, 3), "hgb": (120, 150), "creat": (0.7, 1.0), "crcl": (70, 110),
          "egfr": (70, 100), "hba1c": (7.6, 8.8), "k": (3.9, 4.6), "lvef": (58, 66), "bili": (0.4, 0.8), "ast": (18, 30),
          "alt": (15, 32), "alp": (60, 100)}
CONVERSIONS = ("hemoglobin 1 g/dL = 10 g/L; cell counts 1 x10^9/L = 1,000 /uL; creatinine 1 mg/dL = 88.4 umol/L; "
               "bilirubin 1 mg/dL = 17.1 umol/L; potassium 1 mEq/L = 1 mmol/L; HbA1c mmol/mol = 10.929 x (% - 2.15); "
               "weight 1 lb = 0.4536 kg; height 1 in = 2.54 cm; BMI = kg / m^2")
DRUGS = {"anticancer": ["carboplatin", "cisplatin", "paclitaxel", "docetaxel", "pemetrexed", "gemcitabine", "vinorelbine",
                        "osimertinib", "alectinib", "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab"],
         "checkpoint": ["pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "cemiplimab", "ipilimumab"],
         "immunosuppressive": ["prednisone (oral)", "dexamethasone (oral)", "methylprednisolone (IV)", "methotrexate (oral)",
                               "azathioprine"],
         "steroid_systemic": ["prednisone (oral)", "dexamethasone (oral)", "methylprednisolone (IV)"],
         "insulin": ["insulin glargine", "insulin lispro", "insulin degludec", "insulin aspart"],
         "sulfonylurea": ["glipizide", "glimepiride", "gliclazide", "glyburide"],
         "tzd": ["pioglitazone", "rosiglitazone"],
         "sglt2": ["empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin"]}
DECOY_DRUGS = {"anticancer": ["filgrastim", "ondansetron", "pantoprazole"], "checkpoint": ["bevacizumab", "ramucirumab"],
               "immunosuppressive": ["fluticasone (inhaled)", "triamcinolone (topical cream)", "budesonide (nasal spray)"],
               "steroid_systemic": ["fluticasone (inhaled)", "triamcinolone (topical cream)", "budesonide (nasal spray)"],
               "insulin": ["semaglutide (injection)", "liraglutide (injection)", "metformin"],
               "sulfonylurea": ["sitagliptin", "linagliptin", "metformin"], "tzd": ["sitagliptin", "metformin"],
               "sglt2": ["sitagliptin", "semaglutide (injection)"]}
CATS = {d: {c for c, ds in DRUGS.items() if d in ds} for ds in DRUGS.values() for d in ds}
CATS["dexamethasone (oral)"] = {"immunosuppressive", "steroid_systemic"}  # antiemetic use is still a systemic steroid
BREAST_DRUGS = ["doxorubicin", "capecitabine", "paclitaxel", "tamoxifen", "letrozole", "anastrozole"]
LIVE = ["MMR (measles, mumps, rubella)", "Varicella (Varivax)", "Yellow fever (YF-Vax)", "Influenza, intranasal (FluMist)",
        "Typhoid, oral (Vivotif)"]
NONLIVE = ["Zoster, recombinant (Shingrix)", "Influenza, inactivated (Fluzone)", "Pneumococcal conjugate (Prevnar 20)",
           "Tdap (Boostrix)", "COVID-19 mRNA"]
MAJOR = {"colectomy": "diverticulitis", "total hip arthroplasty": "osteoarthritis of the hip",
         "hysterectomy": "uterine fibroids", "coronary artery bypass grafting": "coronary artery disease",
         "spinal fusion": "lumbar spinal stenosis"}
MINOR = {"skin lesion excision": "seborrheic keratosis", "carpal tunnel release": "carpal tunnel syndrome",
         "cataract surgery": "cataract", "inguinal hernia repair": "inguinal hernia"}
COND = {"hiv": ["HIV infection"], "hbv": ["Chronic hepatitis B"], "hcv": ["Chronic hepatitis C"],
        "ibd": ["Crohn's disease", "Ulcerative colitis"], "allo_transplant": ["Kidney transplant (deceased donor)",
                                                                             "Liver transplant (deceased donor)"],
        "pancreatitis": ["Acute pancreatitis", "Chronic pancreatitis"], "nyha4": ["Heart failure, NYHA class IV"],
        "mtc_men2": ["Medullary thyroid carcinoma", "Multiple endocrine neoplasia type 2"],
        "ckd45": ["Chronic kidney disease stage 4", "Chronic kidney disease stage 5"], "dka": ["Diabetic ketoacidosis"],
        "cns_mets": ["brain metastases"]}
NEAR_COND = {"hiv": [], "hbv": ["Hepatitis B, resolved past infection"], "hcv": ["Hepatitis C, cured after antiviral therapy"],
             "ibd": ["Irritable bowel syndrome"], "allo_transplant": ["Autologous stem cell transplant"],
             "pancreatitis": ["Pancreatic cyst"], "nyha4": ["Heart failure, NYHA class III"],
             "mtc_men2": ["Papillary thyroid carcinoma"], "ckd45": ["Chronic kidney disease stage 3b"], "dka": [],
             "cns_mets": []}
MALIG = ["Prostate adenocarcinoma", "Colon adenocarcinoma", "Melanoma of the skin", "Bladder urothelial carcinoma",
         "Renal cell carcinoma"]
EPISODIC = {"pancreatitis", "dka", "hbv", "hcv"}  # only these may appear as resolved (HIV, IBD, CKD 4-5 ... do not resolve)
RELATIVES = ["mother", "father", "sister", "brother", "maternal aunt", "paternal uncle"]
FILL_COND = ["Hypertension", "Hyperlipidemia", "Osteoarthritis of the knee", "Gastroesophageal reflux disease",
             "Hypothyroidism", "Benign prostatic hyperplasia", "Seasonal allergic rhinitis"]
FILL_MEDS = [("atorvastatin", "hyperlipidemia"), ("amlodipine", "hypertension"), ("omeprazole", "reflux"),
             ("levothyroxine", "hypothyroidism"), ("lisinopril", "hypertension"), ("cetirizine", "allergic rhinitis")]
NSCLC_HIST = ["adenocarcinoma", "squamous cell carcinoma", "large cell carcinoma", "small cell carcinoma",
              "combined small cell carcinoma"]
NSCLC_STAGES = ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IIIC", "IVA", "IVB"]
BREAST_STAGES = ["I", "IIA", "IIB", "IIIA", "IIIB", "IIIC", "IV"]
WIN = (D(2026, 3, 1), D(2026, 9, 30))


def fmt(x, nd):
    return f"{Dm(str(x)):.{nd}f}"


def to_canon(an, val, unit):
    v = Dm(val.replace(",", ""))
    canon, alt, f = UNITS[an][1], UNITS[an][2], UNITS[an][3]
    if unit == canon:
        return v
    if an == "hba1c":
        return v / Dm("10.929") + Dm("2.15")
    return v / Dm(str(f))


def alt_val(an, canon_v):
    if an == "hba1c":
        return fmt(round((float(canon_v) - 2.15) * 10.929), 0)
    f = UNITS[an][3]
    nd = {"anc": 0, "plt": 0, "lym": 0, "hgb": 1, "creat": 0, "bili": 0, "k": 1}[an]
    return fmt(Dm(str(canon_v)) * Dm(str(f)), nd)


# ---------------- predicate evaluation (gold) ----------------
def age_at(dob, day):
    return day.year - dob.year - ((day.month, day.day) < (dob.month, dob.day))


def latest(rows, day):
    rows = [r for r in rows if r["date"] <= day]
    return max(rows, key=lambda r: r["date"]) if rows else None


def lab_latest(P, an):
    return latest([r for r in P["labs"] if r["an"] == an], P["screen"])


def cmp(v, op, t):
    return {"<": v < t, "<=": v <= t, ">": v > t, ">=": v >= t}[op]


def in_range(v, lo, hi, lo_incl, hi_incl):
    if lo is not None and not (v >= Dm(str(lo)) if lo_incl else v > Dm(str(lo))):
        return False
    if hi is not None and not (v <= Dm(str(hi)) if hi_incl else v < Dm(str(hi))):
        return False
    return True


def bmi(P):
    w = latest(P["weights"], P["screen"])
    if w is None:
        return Dm(0)
    kg = Dm(w["val"]) * (Dm("0.4536") if w["unit"] == "lb" else 1)
    h = Dm(P["height"][0]) * (Dm("2.54") if P["height"][1] == "in" else 1) / 100
    return kg / (h * h)


def exposures(P):
    """(category set, end date or None, record) for drugs, procedures and vaccines."""
    out = []
    for m in P["meds"]:
        out.append((CATS.get(m["drug"], set()) | ({"anticancer"} if m["drug"] in BREAST_DRUGS else set()), m["stop"], m))
    for p in P["procs"]:
        out.append(({p["kind"]}, p["date"], p))
    for v in P["vaccines"]:
        out.append(({"live_vaccine"} if v["name"] in LIVE else {"vaccine"}, v["date"], v))
    return out


def has_cond(P, name, scope):
    s = P["screen"]
    if name == "cns_mets":
        return "brain" in P.get("sites", [])
    rows = [c for c in P["conds"] if c["key"] == name]
    if scope == "active":
        return any(c["status"] == "active" for c in rows)
    if isinstance(scope, list):  # an episode counts if any part of it (onset to resolution) lies in the window
        return any(max(c["onset"], c["resolved"] or s) >= s - dt.timedelta(days=scope[1]) for c in rows)
    if scope == "ever_or_family":
        return bool(rows) or any(k == name for k, _, _ in P["family"])
    return bool(rows)


def ev(P, p):
    k, s = p[0], P["screen"]
    if k == "any":
        return any(ev(P, q) for q in p[1:])
    if k == "all":
        return all(ev(P, q) for q in p[1:])
    if k == "not":
        return not ev(P, p[1])
    if k == "age":
        a = age_at(P["dob"], s)
        return (p[1] is None or a >= p[1]) and (p[2] is None or a <= p[2])
    if k == "sex":
        return P["sex"] == p[1]
    if k == "lab":
        r = lab_latest(P, p[1])
        if r is None:
            return False
        return in_range(to_canon(p[1], r["val"], r["unit"]), *p[2:])
    if k == "uln":
        r = lab_latest(P, p[1])
        if r is None:
            return False
        mult = p[3]
        cond = p[5]
        if cond and ((cond == "liver_mets" and "liver" in P.get("sites", []))
                     or (cond == "liver_or_bone_mets" and {"liver", "bone"} & set(P.get("sites", [])))
                     or (cond == "gilbert" and has_cond(P, "gilbert", "ever"))):
            mult = p[4]
        return cmp(Dm(r["val"]), p[2], Dm(str(mult)) * Dm(r["uln"]))
    if k == "ecog":
        e = latest(P["ecog"], s)
        return e is not None and e["val"] <= p[1]
    if k == "bmi":
        return in_range(bmi(P), *p[1:])
    if k == "within":
        cat, days, o = p[1], p[2], p[3]
        for cats, end, rec in exposures(P):
            if cat not in cats:
                continue
            start = rec.get("start", end)
            if end is None or end >= s - dt.timedelta(days=days):
                if o.get("palliative_ok_after") and rec.get("intent") == "palliative" and \
                        end <= s - dt.timedelta(days=o["palliative_ok_after"]):
                    continue
                if o.get("not_for") and rec.get("ind") == o["not_for"]:
                    continue
                if o.get("stable_ok_over") and end is None and start <= s - dt.timedelta(days=o["stable_ok_over"]):
                    continue
                return True
        return False
    if k == "current":
        return any(p[1] in c and e is None for c, e, _ in exposures(P))
    if k == "ever_cat":
        return any(p[1] in c for c, _, _ in exposures(P))
    if k == "ever_for":
        return any(r.get("ind") == p[1] and ("diagnostic" not in c) for c, _, r in exposures(P))
    if k == "cond":
        return has_cond(P, p[1], p[2])
    if k == "proc_ever":
        return any(r["name"] == p[1] for r in P["procs"])
    if k == "preg":
        t = latest(P["hcg"], s)
        return bool(t) and t["res"] == "positive"
    if k == "lact":
        t = latest(P["lact"], s)
        return bool(t) and t["state"] == "breastfeeding"
    if k == "stage":
        return P.get("stage") in p[1]
    if k == "hist":
        return P.get("hist") in p[1]
    if k == "marker":
        return P.get("markers", {}).get(p[1]) == p[2] or (p[2] == "sensitizing" and P.get("markers", {}).get(p[1]) in
                                                          ("exon 19 deletion", "L858R"))
    if k == "dx_days":
        return P["dx_date"] <= s - dt.timedelta(days=p[1])
    if k == "malig":
        lim = s - dt.timedelta(days=p[1])
        return any(c["key"] == "malig" and c["name"] not in p[2] and max(c["onset"], c["resolved"] or s) >= lim
                   for c in P["conds"])
    raise ValueError(k)


def failed_ids(P, trial):
    out = []
    for c in trial["criteria"]:
        v = ev(P, c["pred"])
        if (c["kind"] == "inc" and not v) or (c["kind"] == "exc" and v):
            out.append(c["id"])
    return sorted(out, key=lambda x: (x[0] != "I", int(x[1:])))


# ---------------- generation ----------------
def targets(pred, truth, rng, out):
    """Decompose a criterion predicate with a desired truth into atom targets (first target per atom wins)."""
    k = pred[0]
    if k == "not":
        return targets(pred[1], not truth, rng, out)
    if k in ("all", "any"):
        kids = pred[1:]
        flip = rng.randrange(len(kids))
        for i, q in enumerate(kids):
            if (k == "all") == truth:
                want = truth
            else:
                want = truth if i == flip else not truth
            targets(q, want, rng, out)
        return
    out.setdefault(json.dumps(pred), (pred, truth))


class Gen:
    def __init__(self, i, seed):
        self.rng = random.Random(f"trials:{seed}:{i}")
        self.i = i

    def d(self, back_lo, back_hi):
        return self.P["screen"] - dt.timedelta(days=self.rng.randint(back_lo, back_hi))

    def lab(self, an, val, back=(1, 9), unit=None):
        nd = DEC[an]
        canon = fmt(val, nd)
        r = {"date": self.d(*back), "an": an, "val": canon, "unit": UNITS[an][1]}
        if unit == "alt":
            r["val"], r["unit"] = alt_val(an, canon), UNITS[an][2]
        if UNITS[an][4]:
            r["uln"] = self.P["uln"].setdefault(an, self.rng.choice(ULNS[an]))
            if r["unit"] != UNITS[an][1]:
                r["uln"] = alt_val(an, r["uln"])
        self.P["labs"].append(r)
        return r




def gen_lab(g, atom, t):
    _, an, lo, hi, li, hi_i = atom
    rng, P = g.rng, g.P
    lo_ = lo if lo is not None else NORMAL[an][0] * 0.6
    hi_ = hi if hi is not None else NORMAL[an][1] * 1.15
    inside = lambda: rng.uniform(lo_ * 1.06 if lo is not None else lo_, hi_ * 0.94 if hi is not None else hi_)  # noqa
    outside = lambda: rng.choice([x for x in (lo is not None and rng.uniform(lo * 0.6, lo * 0.93),  # noqa: E731
                                              hi is not None and rng.uniform(hi * 1.07, hi * 1.4)) if x])
    if rng.random() < 0.5:  # superseded older value on the other side
        g.lab(an, round(outside() if t else inside(), DEC[an]), back=(25, 200))
        P["traps"].append("lab_superseded")
    bounds = [(b, inc) for b, inc in ((lo, li), (hi, hi_i)) if b is not None and Dm(fmt(b, DEC[an])) == Dm(str(b))]
    if bounds and rng.random() < 0.3:  # exact boundary value in the canonical unit
        b, inc = rng.choice(bounds)
        g.lab(an, b)
        P["traps"].append("lab_boundary")
        return
    alt = UNITS[an][2] is not None and rng.random() < 0.45
    g.lab(an, round(inside() if t else outside(), DEC[an]), unit="alt" if alt else None)
    if alt:
        P["traps"].append("lab_unit")


def gen_uln(g, atom, t):
    _, an, op, mult, amult, cond = atom
    rng, P = g.rng, g.P
    uln = Dm(P["uln"].setdefault(an, rng.choice(ULNS[an])))
    eff = Dm(str(mult))
    if cond and (cond == "gilbert" or "IV" in (P.get("stage") or "")) and rng.random() < 0.5:
        if cond == "gilbert":
            P["conds"].append({"key": "gilbert", "name": "Gilbert's syndrome", "status": "active",
                               "onset": g.d(2000, 9000), "resolved": None})
        else:
            P["sites"] = sorted(set(P.get("sites", [])) | {"liver" if cond == "liver_mets" or rng.random() < 0.5 else "bone"})
        P["traps"].append("uln_alt_limit")
        lo_r, hi_r = (float(mult) * 1.08, float(amult) * 0.92) if t else (float(amult) * 1.08, float(amult) * 1.5)
        ratio = rng.uniform(lo_r, hi_r)
    else:
        low = t == (op in ("<", "<="))
        ratio = rng.uniform(0.3 * float(eff), 0.92 * float(eff)) if low else rng.uniform(1.08 * float(eff), 1.6 * float(eff))
        exact = eff * uln
        if rng.random() < 0.3 and Dm(fmt(exact, DEC[an])) == exact and cmp(exact, op, exact) == t:
            ratio = None
    if rng.random() < 0.4:
        bad = float(eff) * (1.4 if op in ("<", "<=") else 0.5) if t else float(eff) * (0.5 if op in ("<", "<=") else 1.4)
        g.lab(an, fmt(float(uln) * bad, DEC[an]), back=(30, 200))
        P["traps"].append("lab_superseded")
    if ratio is None:
        g.lab(an, fmt(eff * uln, DEC[an]))
        P["traps"].append("uln_boundary")
    else:
        g.lab(an, fmt(float(uln) * ratio, DEC[an]), unit="alt" if UNITS[an][2] and rng.random() < 0.3 else None)
    if P["uln"][an] != ULNS[an][0]:
        P["traps"].append("uln_nonstandard")


def gen_within(g, atom, t):
    _, cat, days, o = atom
    rng, P = g.rng, g.P
    near_out = lambda: g.d(days + 2, days + 6)  # noqa: E731
    inside = lambda: g.d(1, max(1, days - 2))  # noqa: E731
    if o.get("palliative_ok_after") and rng.random() < 0.5:  # palliative course ending 14-21 days ago: permitted
        end = g.d(o["palliative_ok_after"] + 1, days - 1)
        P["procs"].append({"name": "radiotherapy", "label": "Palliative radiotherapy to spine, 8 Gy in 1 fraction",
                           "kind": "radiotherapy", "start": end, "date": end, "intent": "palliative", "ind": P["disease"]})
        P["traps"].append("rt_palliative")
    if o.get("not_for") and rng.random() < 0.6:  # surgery for the trial cancer inside the window: permitted
        P["procs"].append({"name": "mastectomy", "label": "Modified radical mastectomy", "kind": "major_surgery",
                           "start": None, "date": inside(), "intent": "", "ind": "breast"})
        P["traps"].append("surgery_related")
    if o.get("stable_ok_over") and rng.random() < 0.0 and not t:  # off: dose stability is not charted (audit)
        d0 = rng.choice(DRUGS[cat])
        P["meds"].append({"drug": d0, "ind": "asthma", "start": g.d(o["stable_ok_over"] + 10, 900), "stop": None,
                          "stop_note": False})
        P["traps"].append("steroid_stable")
        return
    if cat in DECOY_DRUGS and rng.random() < 0.5:
        P["meds"].append({"drug": rng.choice(DECOY_DRUGS[cat]), "ind": "symptoms", "start": g.d(days + 20, 400),
                          "stop": None, "stop_note": False})
        P["traps"].append("drug_class_decoy")
    if cat == "live_vaccine" and rng.random() < 0.6:
        P["vaccines"].append({"name": rng.choice(NONLIVE), "date": inside(), "note": rng.random() < 0.4})
        P["traps"].append("vaccine_nonlive")
    if not t and rng.random() < 0.7:
        end = near_out()
        P["traps"].append("window_edge")
    elif not t:
        return
    else:
        end = inside()
        if o.get("stable_ok_over"):
            end = None
    if cat in DRUGS:
        start = (end or P["screen"]) - dt.timedelta(days=rng.randint(20, 300))
        if end is None:  # ongoing and started recently: not a stable long-term dose
            start = g.d(10, o["stable_ok_over"] - 5)
        note = end is not None and rng.random() < 0.5
        P["meds"].append({"drug": rng.choice(DRUGS[cat]), "ind": P["disease"] if cat == "anticancer" else "flare",
                          "start": start, "stop": end, "stop_note": note})
        if note:
            P["traps"].append("stop_in_notes")
    elif cat == "live_vaccine":
        P["vaccines"].append({"name": rng.choice(LIVE), "date": end, "note": rng.random() < 0.5})
    elif cat == "radiotherapy":
        P["procs"].append({"name": "radiotherapy", "label": "Radiotherapy to chest, 30 Gy in 10 fractions",
                           "kind": "radiotherapy", "start": end - dt.timedelta(days=13), "date": end,
                           "intent": "curative", "ind": P["disease"]})
    else:
        table = MAJOR if cat == "major_surgery" else MINOR
        nm = rng.choice(sorted(table))
        P["procs"].append({"name": nm, "label": nm[0].upper() + nm[1:], "kind": cat, "start": None, "date": end,
                           "intent": "", "ind": table[nm], "note": rng.random() < 0.4})


def add_cond(g, key, name, status="active", back=(200, 3000), resolved=None):
    g.P["conds"].append({"key": key, "name": name, "status": status, "onset": g.d(*back), "resolved": resolved})
    return g.P["conds"][-1]


def gen_cond(g, atom, t):
    _, key, scope = atom
    rng, P = g.rng, g.P
    if key == "cns_mets":
        if t:
            P["stage"] = "IVB" if P["disease"] == "nsclc" else "IV"
            P["sites"] = sorted(set(P.get("sites", [])) | {"brain"})
        else:
            P["ruled_out"].append(("brain metastases", g.d(5, 60)))
            P["traps"].append("ruled_out")
        return
    if t:
        if scope == "ever_or_family" and rng.random() < 0.5:
            P["family"].append((key, rng.choice(RELATIVES), rng.choice(COND[key])))
            P["traps"].append("family_counts")
            return
        if isinstance(scope, list):
            c = add_cond(g, key, rng.choice(COND[key]), "resolved", back=(9, scope[1] - 5))
            c["resolved"] = c["onset"] + dt.timedelta(days=4)
            return
        st = "active" if scope == "active" or key not in EPISODIC or rng.random() < 0.5 else "resolved"
        c = add_cond(g, key, rng.choice(COND[key]), st)
        if st == "resolved":
            c["resolved"] = c["onset"] + dt.timedelta(days=rng.randint(20, 300))
            P["traps"].append("resolved_counts")
        return
    r = rng.random()
    if r < 0.3 and scope != "ever_or_family":
        P["family"].append((key, rng.choice(RELATIVES), rng.choice(COND[key])))
        P["traps"].append("family_decoy")
    elif r < 0.5:
        P["ruled_out"].append((rng.choice(COND[key]), g.d(10, 400)))
        P["traps"].append("ruled_out")
    elif r < 0.75 and NEAR_COND[key]:
        nm = rng.choice(NEAR_COND[key])
        add_cond(g, key + "_near", nm, "resolved" if ("resolved" in nm or "cured" in nm) else
                 rng.choice(["active", "resolved"]))["resolved"] = None
        if ("resolved" in nm or "cured" in nm) or P["conds"][-1]["status"] == "resolved":
            P["conds"][-1]["resolved"] = P["conds"][-1]["onset"] + dt.timedelta(days=rng.randint(60, 300))
        P["traps"].append("near_condition")
    elif r < 0.9 and scope == "active" and key in EPISODIC | {"nyha4"}:
        c = add_cond(g, key, rng.choice(COND[key]), "resolved")
        c["resolved"] = c["onset"] + dt.timedelta(days=rng.randint(30, 300))
        P["traps"].append("resolved_not_active")
    elif isinstance(scope, list):
        c = add_cond(g, key, rng.choice(COND[key]), "resolved", back=(scope[1] + 8, scope[1] + 20))
        c["resolved"] = c["onset"] + dt.timedelta(days=4)
        P["traps"].append("window_edge")
    elif scope != "ever_or_family":
        P["family"].append((key, rng.choice(RELATIVES), rng.choice(COND[key])))
        P["traps"].append("family_decoy")
    else:
        P["ruled_out"].append((rng.choice(COND[key]), g.d(10, 400)))
        P["traps"].append("ruled_out")


def gen_malig(g, atom, t):
    _, days, exc = atom
    rng, P = g.rng, g.P
    if exc and rng.random() < 0.5:  # an excepted skin / in-situ cancer inside the window
        c = add_cond(g, "malig", rng.choice(exc), "resolved", back=(100, days - 100))
        c["resolved"] = c["onset"] + dt.timedelta(days=40)
        P["traps"].append("malig_exception")
    if t:
        c = add_cond(g, "malig", rng.choice(MALIG), "resolved", back=(200, days - 100))
        c["resolved"] = c["onset"] + dt.timedelta(days=rng.randint(60, 150))
    elif rng.random() < 0.85:
        c = add_cond(g, "malig", rng.choice(MALIG), "resolved", back=(days + 200, days + 1500))
        c["resolved"] = c["onset"] + dt.timedelta(days=rng.randint(60, 150))
        P["traps"].append("malig_old")


def gen_age(g, atom, t):
    _, lo, hi = atom
    rng, P = g.rng, g.P
    s = P["screen"]
    edge = hi if hi is not None and (lo is None or rng.random() < 0.8) else lo
    if t:
        if rng.random() < 0.6 and hi is not None:  # turns hi+1 shortly after screening: still eligible
            age, after = hi, True
            P["traps"].append("age_birthday")
        else:
            age, after = rng.randint(max(lo or 40, 40), min(hi or 72, 72)), rng.random() < 0.5
            age = min(max(age, lo or age), hi or age)
    else:
        age, after = (hi + 1, False) if edge == hi else (lo - 1, rng.random() < 0.5)
        if rng.random() < 0.6:
            P["traps"].append("age_birthday")
    bday = s + dt.timedelta(days=rng.randint(3, 40)) if after else s - dt.timedelta(days=rng.randint(3, 40))
    year = bday.year - age - (1 if after else 0)
    try:
        P["dob"] = bday.replace(year=year)
    except ValueError:
        P["dob"] = bday.replace(year=year, day=28)


def gen_bmi(g, atom, t):
    _, lo, hi, li, hi_i = atom
    rng, P = g.rng, g.P
    h_cm = Dm(P["height"][0]) * (Dm("2.54") if P["height"][1] == "in" else 1)
    m2 = (h_cm / 100) ** 2
    for _ in range(500):
        b = rng.uniform(16, 50)
        ok = in_range(Dm(str(b)), lo, hi, li, hi_i)
        near = [x for x in (lo, hi) if x is not None and abs(b - x) < 0.4]
        if ok == t and not near:
            break
    if rng.random() < 0.4:  # older weight on the other side
        for _ in range(500):
            ob = rng.uniform(16, 50)
            if in_range(Dm(str(ob)), lo, hi, li, hi_i) != t and all(abs(ob - x) > 0.4 for x in (lo, hi) if x is not None):
                P["weights"].append({"date": g.d(60, 400), "val": fmt(Dm(str(ob)) * m2, 1), "unit": "kg", "note": False})
                P["traps"].append("weight_superseded")
                break
    kg = Dm(str(b)) * m2
    lb = rng.random() < 0.4
    P["weights"].append({"date": g.d(1, 20), "val": fmt(kg / Dm("0.4536"), 0) if lb else fmt(kg, 1),
                         "unit": "lb" if lb else "kg", "note": rng.random() < 0.3})
    if lb or P["height"][1] == "in":
        P["traps"].append("bmi_units")


def gen_ecog(g, atom, t):
    rng, P = g.rng, g.P
    v = rng.randint(0, atom[1]) if t else atom[1] + 1
    if rng.random() < 0.5:
        P["ecog"].append({"date": g.d(30, 120), "val": atom[1] + 1 if t else rng.randint(0, atom[1]), "note": False})
        P["traps"].append("ecog_superseded")
    P["ecog"].append({"date": g.d(1, 6), "val": v, "note": rng.random() < 0.5})


def gen_preg(g, atom, t):
    rng, P = g.rng, g.P
    if t and P["sex"] == "F":
        P["hcg"].append({"date": g.d(1, 10), "res": "positive", "note": rng.random() < 0.5})
    elif P["sex"] == "F" and age_at(P["dob"], P["screen"]) < 52:
        if rng.random() < 0.7:
            P["hcg"].append({"date": g.d(250, 420), "res": "positive", "note": rng.random() < 0.5})
            P["traps"].append("old_pregnancy")
        P["hcg"].append({"date": g.d(1, 10), "res": "negative", "note": rng.random() < 0.5})


def gen_lact(g, atom, t):
    rng, P = g.rng, g.P
    if t and P["sex"] == "F":
        P["lact"].append({"date": g.d(1, 20), "state": "breastfeeding"})
    elif P["sex"] == "F" and rng.random() < 0.7:
        P["lact"].append({"date": g.d(100, 200), "state": "breastfeeding"})
        P["lact"].append({"date": g.d(10, 60), "state": "weaned"})
        P["traps"].append("weaned")


def gen_atom(g, atom, t):
    k, rng, P = atom[0], g.rng, g.P
    if k == "lab":
        gen_lab(g, atom, t)
    elif k == "uln":
        gen_uln(g, atom, t)
    elif k == "within":
        gen_within(g, atom, t)
    elif k == "current":
        if t:
            P["meds"].append({"drug": rng.choice(DRUGS[atom[1]]), "ind": "diabetes", "start": g.d(100, 900), "stop": None,
                              "stop_note": False})
        elif rng.random() < 0.6 and False:  # stopped-drug decoy dropped (contested "who are taking" reading); rng draw kept
            stop = g.d(20, 120)
            P["meds"].append({"drug": rng.choice(DRUGS[atom[1]]), "ind": "diabetes", "start": g.d(300, 900), "stop": stop,
                              "stop_note": rng.random() < 0.5})
            P["traps"].append("stopped_not_current")
        else:
            P["meds"].append({"drug": rng.choice(DECOY_DRUGS[atom[1]]), "ind": "diabetes", "start": g.d(100, 900),
                              "stop": None, "stop_note": False})
            P["traps"].append("drug_class_decoy")
    elif k == "ever_cat":
        if t:
            end = g.d(200, 700)
            P["meds"].append({"drug": rng.choice(DRUGS[atom[1]]), "ind": P["disease"], "start": end - dt.timedelta(days=200),
                              "stop": end, "stop_note": rng.random() < 0.4})
        else:
            end = g.d(200, 700)
            P["meds"].append({"drug": rng.choice(DECOY_DRUGS[atom[1]]), "ind": P["disease"],
                              "start": end - dt.timedelta(days=120), "stop": end, "stop_note": False})
            P["traps"].append("drug_class_decoy")
        P["traps"].append("drug_class")
    elif k == "ever_for":
        P["procs"].append({"name": "biopsy", "label": "Core needle biopsy, breast", "kind": "diagnostic", "start": None,
                           "date": g.d(32, 60), "intent": "diagnostic", "ind": "breast"})
        P["traps"].append("diagnostic_not_therapy")
        if t:
            r = rng.random()
            if r < 0.4:
                end = g.d(300, 1500)
                P["meds"].append({"drug": rng.choice(BREAST_DRUGS), "ind": "breast", "start": end - dt.timedelta(days=150),
                                  "stop": end, "stop_note": False})
            elif r < 0.7:
                P["procs"].append({"name": "mastectomy", "label": "Modified radical mastectomy", "kind": "major_surgery",
                                   "start": None, "date": g.d(300, 1500), "intent": "", "ind": "breast"})
            else:
                end = g.d(300, 1500)
                P["procs"].append({"name": "radiotherapy", "label": "Radiotherapy to left breast, 40 Gy in 15 fractions",
                                   "kind": "radiotherapy", "start": end - dt.timedelta(days=20), "date": end,
                                   "intent": "curative", "ind": "breast"})
    elif k == "cond":
        gen_cond(g, atom, t)
    elif k == "malig":
        gen_malig(g, atom, t)
    elif k == "proc_ever":
        if t:
            P["procs"].append({"name": atom[1], "label": "Laparoscopic cholecystectomy", "kind": "major_surgery",
                               "start": None, "date": g.d(400, 4000), "intent": "", "ind": ""})
        else:
            add_cond(g, "cholecystitis", "Acute cholecystitis, treated with antibiotics", "resolved")["resolved"] = g.d(100, 300)
            P["traps"].append("near_condition")
    elif k == "ecog":
        gen_ecog(g, atom, t)
    elif k == "bmi":
        gen_bmi(g, atom, t)
    elif k == "preg":
        gen_preg(g, atom, t)
    elif k == "lact":
        gen_lact(g, atom, t)
    elif k == "dx_days":
        P["dx_date"] = g.d(atom[1] + 5, atom[1] + 40) if t and rng.random() < 0.5 else g.d(400, 4000) if t else g.d(30, atom[1] - 5)
        if t and rng.random() < 0.5:
            add_cond(g, "prediabetes", "Prediabetes", "resolved", back=(1200, 3000))["resolved"] = P["dx_date"]
            P["traps"].append("near_condition")
    # context atoms (age, sex, stage, hist, marker) are handled in make_patient


def make_patient(i, seed, trial):
    g = Gen(i, seed)
    rng = g.rng
    P = g.P = {"id": f"P{i + 1:04d}", "trial": trial["nct"], "disease": trial["disease"], "conds": [], "family": [],
               "ruled_out": [], "meds": [], "procs": [], "vaccines": [], "labs": [], "ecog": [], "weights": [], "hcg": [],
               "lact": [], "uln": {}, "traps": [], "markers": {}}
    P["screen"] = WIN[0] + dt.timedelta(days=rng.randint(0, (WIN[1] - WIN[0]).days))
    crit = trial["criteria"]
    nfail = rng.choices([0, 1, 2, 3], [0.015, 0.46, 0.365, 0.16])[0]
    fail = set(rng.sample(range(len(crit)), min(nfail, len(crit))))
    tg = {}
    for j, c in enumerate(crit):
        targets(c["pred"], (j not in fail) == (c["kind"] == "inc"), rng, tg)
    want = {k: t for k, (a, t) in tg.items()}
    atoms = [a for a, _ in tg.values()]
    # context: sex, age, disease details
    sex_t = [t for a, t in tg.values() if a[0] == "sex"]
    preg_like = any(t for a, t in tg.values() if a[0] in ("preg", "lact"))
    if sex_t:
        P["sex"] = "F" if sex_t[0] else "M"
    else:
        P["sex"] = "F" if preg_like or trial["disease"] == "breast" or rng.random() < 0.5 else "M"
    P["height"] = (fmt(rng.uniform(60, 74), 0), "in") if rng.random() < 0.35 else (fmt(rng.uniform(152, 190), 0), "cm")
    age_atoms = [(a, t) for a, t in tg.values() if a[0] == "age"]
    if age_atoms:
        gen_age(g, *age_atoms[0])
    else:
        gen_age(g, ["age", 30, 74], True)
        P["traps"] = [x for x in P["traps"] if x != "age_birthday"]
    if preg_like and age_at(P["dob"], P["screen"]) > 44 and P["sex"] == "F":
        P["dob"] = P["dob"].replace(year=P["screen"].year - rng.randint(28, 40))
    d = trial["disease"]
    st = [(a, t) for a, t in tg.values() if a[0] == "stage"]
    hs = [(a, t) for a, t in tg.values() if a[0] == "hist"]
    stages = NSCLC_STAGES if d == "nsclc" else BREAST_STAGES
    if d != "t2d":
        if st:
            a, t = st[0]
            pool = [x for x in stages if (x in a[1]) == t]
            near = [x for x in pool if any(abs(stages.index(x) - stages.index(y)) == 1 for y in a[1])]
            P["stage"] = rng.choice(near if near and rng.random() < 0.6 else pool)
            if near:
                P["traps"].append("stage_near")
        else:
            P["stage"] = rng.choice(stages[2:6])
        if d == "nsclc":
            if hs:
                a, t = hs[0]
                pool = [x for x in NSCLC_HIST if (x in a[1]) == t]
                if P["trial"] == "NCT06623422":  # its line also names large-cell neuroendocrine tumours: keep clear of it
                    pool = [x for x in pool if x != "large cell carcinoma"]
                P["hist"] = rng.choice(pool)
            else:
                P["hist"] = rng.choice(NSCLC_HIST[:3])
            P["markers"] = {"EGFR": "wild-type", "ALK": "negative"}
            for a, t in tg.values():
                if a[0] == "marker" and t:
                    P["markers"][a[1]] = rng.choice(["exon 19 deletion", "L858R"]) if a[2] == "sensitizing" else "positive"
                elif a[0] == "marker":
                    P["markers"]["KRAS"] = "G12C"
                    P["traps"].append("marker_decoy")
        else:
            P["hist"] = "invasive ductal carcinoma"
            P["receptors"] = rng.choice(["ER-negative, PR-negative, HER2-negative", "ER-positive, PR-positive, HER2-negative",
                                         "ER-negative, PR-negative, HER2-positive"])
        P["sites"] = sorted(rng.sample(["bone", "lung", "adrenal"], rng.randint(1, 2))) if P["stage"].startswith("IV") else []
        P["dx_date"] = g.d(20, 90)
        add_cond(g, "primary", "Non-small cell lung cancer" if d == "nsclc" else "Breast cancer", back=(20, 20))["onset"] = P["dx_date"]
    else:
        P["dx_date"] = g.d(400, 5000)
    for a in atoms:
        if a[0] not in ("sex", "age", "stage", "hist", "marker"):
            gen_atom(g, a, want[json.dumps(a)])
    if d == "t2d":
        add_cond(g, "primary", "Type 2 diabetes mellitus")["onset"] = P["dx_date"]
    filler(g, atoms)
    P["failed"] = failed_ids(P, trial)
    return P


def filler(g, atoms):
    rng, P = g.rng, g.P
    for an in ["hgb", "plt", "anc", "creat", "egfr", "hba1c", "k", "ast", "alt", "bili"]:
        if an not in {r["an"] for r in P["labs"]} and (P["disease"] != "t2d" or an in ("creat", "egfr", "hba1c", "k")) \
                and (P["disease"] == "t2d" or an != "hba1c"):
            g.lab(an, round(rng.uniform(*NORMAL[an]), DEC[an]))
    for c in rng.sample([c for c in FILL_COND if P["sex"] == "M" or c != "Benign prostatic hyperplasia"], rng.randint(1, 3)):
        add_cond(g, "fill", c)
    for drug, ind in rng.sample(FILL_MEDS, rng.randint(1, 3)):
        P["meds"].append({"drug": drug, "ind": ind, "start": g.d(200, 2000), "stop": None, "stop_note": False})
    if P["disease"] == "t2d" and not any(m["drug"] == "metformin" for m in P["meds"]):
        P["meds"].append({"drug": "metformin", "ind": "diabetes", "start": P["dx_date"], "stop": None, "stop_note": False})
    P["vaccines"].append({"name": "Influenza, inactivated (Fluzone)", "date": g.d(200, 400), "note": False})
    if not P["weights"]:
        h = Dm(P["height"][0]) * (Dm("2.54") if P["height"][1] == "in" else 1) / 100
        P["weights"].append({"date": g.d(1, 30), "val": fmt(Dm(str(rng.uniform(21, 31))) * h * h, 1), "unit": "kg", "note": False})
    if not P["ecog"] and P["disease"] != "t2d":
        P["ecog"].append({"date": g.d(1, 6), "val": rng.randint(0, 1), "note": False})
    if P["sex"] == "F" and not P["hcg"] and age_at(P["dob"], P["screen"]) < 52:
        P["hcg"].append({"date": g.d(1, 10), "res": "negative", "note": False})


# ---------------- rendering ----------------
def note(rng, key, **kw):
    return common.fill(rng.choice(T[key]), **kw)


def render(P, seed):
    rng = random.Random(f"trials-render:{seed}:{P['id']}")
    s = P["screen"]
    trial = BY_NCT[P["trial"]]
    sec = {}
    sec["summary"] = (f"Patient {P['id']}\nSex: {'female' if P['sex'] == 'F' else 'male'}\nDate of birth: {P['dob']}\n"
                      f"Screening date: {s}\nAssigned trial: {P['trial']} ({trial['title']})\n"
                      f"Height: {P['height'][0]} {P['height'][1]}\n")
    rows = sorted(P["conds"], key=lambda c: c["onset"])
    sec["problems"] = "condition | status | onset | resolved\n" + "\n".join(
        f"{c['name']} | {c['status']} | {c['onset']} | {c['resolved'] or '-'}" for c in rows)
    if P["disease"] != "t2d":
        o = [f"Primary diagnosis: {'Non-small cell lung cancer' if P['disease'] == 'nsclc' else 'Breast cancer'}",
             f"Diagnosed: {P['dx_date']}", f"Histology: {P['hist']}", f"Stage (AJCC 8th edition): {P['stage']}"]
        if P["disease"] == "nsclc":
            o += [f"{k}: {v}" for k, v in sorted(P["markers"].items())]
        else:
            o.append(f"Receptors: {P['receptors']}")
        o.append(f"Metastatic sites: {', '.join(P['sites']) if P['sites'] else 'none'}")
        sec["oncology"] = "\n".join(o)
    sec["medications"] = "drug | indication | start | stop\n" + "\n".join(
        f"{m['drug']} | {m['ind']} | {m['start']} | {'see notes' if m['stop_note'] else (m['stop'] or '-')}"
        for m in sorted(P["meds"], key=lambda m: m["start"]))
    pr = [p for p in P["procs"] if not p.get("note")]
    sec["procedures"] = "start | end | procedure | category | intent | indication\n" + "\n".join(
        f"{p['start'] or p['date']} | {p['date']} | {p['label']} | {p['kind'].replace('_', ' ')} | {p['intent'] or '-'} | "
        f"{p['ind'] or '-'}" for p in sorted(pr, key=lambda p: p["date"]))
    sec["immunizations"] = "date | vaccine\n" + "\n".join(
        f"{v['date']} | {v['name']}" for v in sorted(P["vaccines"], key=lambda v: v["date"]) if not v["note"])
    labs = sorted(P["labs"], key=lambda r: (r["date"], r["an"]))
    sec["labs"] = "date | test | value | unit | upper limit of normal\n" + "\n".join(
        f"{r['date']} | {UNITS[r['an']][0]} | {r['val']} | {r['unit']} | {r.get('uln', '-')}" for r in labs) + \
        "".join(f"\n{h['date']} | hCG (pregnancy test) | {h['res']} | - | -" for h in P["hcg"] if not h["note"])
    sec["vitals"] = "date | weight\n" + "\n".join(f"{w['date']} | {w['val']} {w['unit']}" for w in
                                                  sorted(P["weights"], key=lambda w: w["date"]) if not w["note"])
    sec["performance"] = "date | ECOG performance status\n" + "\n".join(
        f"{e['date']} | {e['val']}" for e in sorted(P["ecog"], key=lambda e: e["date"]) if not e["note"])
    notes = []
    for e in P["ecog"]:
        if e["note"]:
            notes.append((e["date"], note(rng, "n_ecog", ecog=e["val"])))
    for h in P["hcg"]:
        if h["note"]:
            notes.append((h["date"], note(rng, "n_hcg", result=h["res"])))
    for w in P["weights"]:
        if w["note"]:
            notes.append((w["date"], note(rng, "n_weight", weight=f"{w['val']} {w['unit']}")))
    for x in P["lact"]:
        notes.append((x["date"], note(rng, "n_breastfeeding") if x["state"] == "breastfeeding" else
                      note(rng, "n_weaned", date=x["date"])))
    for key, rel, cond in P["family"]:
        notes.append((s - dt.timedelta(days=rng.randint(10, 300)), note(rng, "n_family", relative=rel, condition=cond)))
    for cond, d in P["ruled_out"]:
        notes.append((d, note(rng, "n_ruled_out", condition=cond)))
    for m in P["meds"]:
        if m["stop_note"]:
            notes.append((m["stop"] + dt.timedelta(days=rng.randint(0, 3)), note(rng, "n_last_dose", drug=m["drug"], date=m["stop"])))
    for v in P["vaccines"]:
        if v["note"]:
            notes.append((v["date"] + dt.timedelta(days=rng.randint(0, 5)), note(rng, "n_vaccine", vaccine=v["name"], date=v["date"])))
    for p in P["procs"]:
        if p.get("note"):
            notes.append((p["date"] + dt.timedelta(days=rng.randint(1, 5)),
                          note(rng, "n_surgery", procedure=p["name"], category=p["kind"].split("_")[0], date=p["date"])))
    for _ in range(rng.randint(2, 4)):
        notes.append((s - dt.timedelta(days=rng.randint(0, 200)), rng.choice(T["n_pad"])))
    notes = [(d if d <= s else s, t) for d, t in notes]
    sec["notes"] = [f"{d} | {t}" for d, t in sorted(notes, key=lambda x: (x[0], x[1]))]
    return sec


def generate(n, seed, out_dir):
    pats, world, gold = [], {}, {}
    for i in range(n):
        trial = TRIALS[random.Random(f"trials-pick:{seed}:{i}").randrange(len(TRIALS))]
        P = make_patient(i, seed, trial)
        pats.append(P)
        world[P["id"]] = render(P, seed)
        gold[P["id"]] = {"trial": P["trial"], "eligible": not P["failed"], "failed": P["failed"],
                         "traps": sorted(set(P["traps"]))}
    used = sorted({P["trial"] for P in pats})
    files = {"patients.json": {"rows": [f"{P['id']} | {P['trial']} | {P['screen']}" for P in pats], "charts": world},
             "trials.json": [BY_NCT[t] | {"criteria": [{"id": c["id"], "text": c["text"]} for c in BY_NCT[t]["criteria"]]}
                             for t in used]}
    for t in files["trials.json"]:
        t.pop("disease", None)
    ex = {pats[0]["id"]: {"eligible": False, "failed": ["I2", "E1"]}}
    prompt = "\n\n".join([common.fill(T["intro"], n=n, first=pats[0]["id"], last=pats[-1]["id"], ntrials=len(used)),
                          rules_text(json.dumps(ex))])
    return common.finish(TYPE_ID, n, seed, out_dir, prompt, {"patients": gold}, TIERS, files=files)


def rules_text(example):
    return "\n\n".join([T["rule_scope"], T["rule_windows"], T["rule_latest"],
                        common.fill(T["rule_units"], conversions=CONVERSIONS), T["rule_history"], T["rule_combine"],
                        common.fill(T["answer_line"], example=example)])


# ---------------- tools ----------------
_cache, _lock = {}, threading.Lock()
SECTIONS = ["summary", "problems", "oncology", "medications", "procedures", "immunizations", "labs", "vitals",
            "performance", "notes"]


def _world(world_dir):
    with _lock:
        if world_dir not in _cache:
            p = Path(world_dir)
            _cache[world_dir] = (json.loads((p / "patients.json").read_text()),
                                 {t["nct"]: t for t in json.loads((p / "trials.json").read_text())})
        return _cache[world_dir]


def tools(world_dir):
    def list_patients(a):
        pw, _ = _world(world_dir)
        off, lim = common.page_args(a, 100, 100)
        rows = pw["rows"]
        nxt = off + lim
        return "patient | assigned trial | screening date\n" + "\n".join(rows[off:nxt]) + \
            (f"\n(patients {off}-{nxt - 1} of {len(rows)}; next offset {nxt})" if nxt < len(rows) else "")

    def list_trials(a):
        _, tw = _world(world_dir)
        return "\n".join(f"{k} | {v['title']} | {len(v['criteria'])} criteria" for k, v in tw.items())

    def read_trial(a):
        _, tw = _world(world_dir)
        k = str(a["trial_id"]).strip().upper()
        if k not in tw:
            raise KeyError(f"unknown trial {k}")
        t = tw[k]
        return f"{k}: {t['title']}\n" + "\n".join(f"{c['id']}: {c['text']}" for c in t["criteria"])

    def _pat(a):
        pw, _ = _world(world_dir)
        pid = str(a["patient_id"]).strip().upper()
        if pid not in pw["charts"]:
            raise KeyError(f"unknown patient {pid}")
        return pw["charts"][pid]

    def read_chart(a):
        ch = _pat(a)
        sec = str(a.get("section") or "").strip().lower()
        if sec not in SECTIONS or sec == "notes":
            raise ValueError(f"section must be one of {[x for x in SECTIONS if x != 'notes']} (notes: use read_notes)")
        if sec not in ch:
            return "(no oncology section: this patient has no cancer diagnosis)"
        return ch[sec]

    def read_notes(a):
        ch = _pat(a)
        off, lim = common.page_args(a, 20, 20)
        rows = ch["notes"]
        nxt = off + lim
        return "date | note\n" + "\n".join(rows[off:nxt]) + \
            (f"\n(notes {off}-{nxt - 1} of {len(rows)}; next offset {nxt})" if nxt < len(rows) else "")

    Dd = T["tools"]
    S, I = {"type": "string"}, {"type": "integer"}
    return [common.tool("list_patients", Dd["list_patients"], {"offset": I, "limit": I}, [], list_patients),
            common.tool("list_trials", Dd["list_trials"], {}, [], list_trials),
            common.tool("read_trial", Dd["read_trial"], {"trial_id": S}, ["trial_id"], read_trial),
            common.tool("read_chart", Dd["read_chart"], {"patient_id": S, "section": {"type": "string", "enum": SECTIONS[:-1]}},
                        ["patient_id", "section"], read_chart),
            common.tool("read_notes", Dd["read_notes"], {"patient_id": S, "offset": I, "limit": I}, ["patient_id"], read_notes),
            common.calculator_tool(Dd["calculator"])]


# ---------------- scoring ----------------
_CID = re.compile(r"^\s*([IE])\s*-?\s*(\d{1,2})\s*$", re.I)


def canon_ids(v):
    if not isinstance(v, list):
        return None
    out = set()
    for x in v:
        m = _CID.match(str(x))
        if not m:
            return None
        out.add(f"{m.group(1).upper()}{int(m.group(2))}")
    return sorted(out, key=lambda x: (x[0] != "I", int(x[1:])))


def _elig(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return {"true": True, "yes": True, "eligible": True, "false": False, "no": False, "not eligible": False,
            "ineligible": False}.get(s)


def parse_answer(answer):
    obj = common.last_json(answer, lambda o: isinstance(o, dict) and any(re.fullmatch(r"P\d{4}", str(k).strip().upper())
                                                                         for k in o))
    out = {}
    for k, v in (obj or {}).items():
        k = str(k).strip().upper()
        if re.fullmatch(r"P\d{4}", k) and isinstance(v, dict):
            out[k] = {"eligible": _elig(v.get("eligible")), "failed": canon_ids(v.get("failed"))}
    return out


def unit_ok(pa, ga):
    return bool(pa) and pa["failed"] is not None and pa["failed"] == ga["failed"] and pa["eligible"] == ga["eligible"]


def score(task, answer):
    g = common.gold_of(task)["patients"]
    p = parse_answer(answer)
    ok = sum(unit_ok(p.get(k), v) for k, v in g.items())
    part = sum(1 for k, v in g.items() if p.get(k) and p[k]["eligible"] == v["eligible"])
    n = len(g)
    return {"score": ok / n, "score_exact": float(ok == n), "score_partial": part / n, "n_correct": ok, "n_units": n}
