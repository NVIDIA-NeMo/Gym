"""Hand encoding (Claude, code) of the eligibility lines shown to agents. Each entry: (line index in the registry's
eligibilityCriteria text, "inc"/"exc", predicate). Lines not listed here are never shown. A "reg:" line is a structured
registry field (sex, ages) rendered as "Field: value". Predicates (all evaluated at the patient's screening date):
  ["age", lo, hi]                      whole years, inclusive, None = open
  ["sex", "F"]
  ["lab", analyte, lo, hi, lo_incl, hi_incl]   most recent value in canonical unit (see type.UNITS)
  ["uln", analyte, op, mult, alt_mult, alt_cond]   most recent value / its own ULN; alt_mult applies if alt_cond holds
  ["ecog", max]                        most recent ECOG <= max
  ["bmi", lo, hi, lo_incl, hi_incl]
  ["within", category, days, opts]     exposure of the category ending (or ongoing) within days before screening
  ["current", category]                exposure ongoing at screening
  ["ever_cat", category]               any exposure of the category, ever
  ["ever_for", indication]             any therapy (drug, surgery, radiotherapy) given for that indication
  ["cond", name, scope]                scope: "ever" | "active" | ["within", days] | "ever_or_family"
  ["proc_ever", name]
  ["preg"] / ["lact"]
  ["stage", [stages]] / ["hist", [histologies]] / ["marker", gene, status]
  ["dx_days", days]                    primary diagnosis at least days before screening
  ["malig", days, [exception types]]   another malignancy with onset or resolution within days
  ["any", ...] / ["all", ...] / ["not", p]
"""
MAJOR, MINOR = "major_surgery", "minor_surgery"
TRIALS = [
    ("NCT05654454", "nsclc", [
        (3, "inc", ["age", 18, None]),
        (4, "inc", ["stage", ["IIIB", "IIIC", "IVA", "IVB"]]),
        (5, "inc", ["hist", ["adenocarcinoma", "large cell carcinoma"]]),
        (6, "inc", ["ecog", 1]),
        (9, "inc", ["lab", "anc", 1.5, None, True, None]),
        (10, "inc", ["lab", "plt", 100, None, True, None]),
        (11, "inc", ["lab", "hgb", 90, None, True, None]),
        (12, "inc", ["uln", "bili", "<=", 1.5, None, None]),
        (13, "inc", ["all", ["uln", "ast", "<", 3, 5, "liver_mets"], ["uln", "alt", "<", 3, 5, "liver_mets"]]),
        (14, "inc", ["uln", "alp", "<", 3, 5, "liver_or_bone_mets"]),
        (18, "exc", ["any", ["marker", "EGFR", "sensitizing"], ["marker", "ALK", "positive"]]),
        (20, "exc", ["within", MAJOR, 28, {}]),
        (21, "exc", ["within", MINOR, 7, {}]),
        (24, "exc", ["cond", "cns_mets", "ever"]),
        (25, "exc", ["any", ["preg"], ["lact"]])]),
    ("NCT06225284", "breast", [
        ("reg:sex", "inc", ["sex", "F"]),
        (9, "inc", ["ecog", 1]),
        (12, "inc", ["lab", "hgb", 90, None, True, None]),
        (13, "inc", ["lab", "anc", 1.5, None, True, None]),
        (14, "inc", ["lab", "plt", 100, None, True, None]),
        (15, "inc", ["uln", "bili", "<=", 1.5, None, None]),
        (16, "inc", ["all", ["uln", "ast", "<=", 2.5, None, None], ["uln", "alt", "<=", 2.5, None, None]]),
        (17, "inc", ["any", ["lab", "creat", None, 1.5, None, True], ["lab", "crcl", 50, None, True, None]]),
        (26, "exc", ["ever_for", "breast"]),
        (27, "exc", ["stage", ["IV"]]),
        (28, "exc", ["any", ["preg"], ["lact"]]),
        (35, "exc", ["within", MAJOR, 28, {}])]),
    ("NCT07647263", "breast", [
        ("reg:sex", "inc", ["sex", "F"]),
        (5, "inc", ["not", ["ever_for", "breast"]]),
        (6, "inc", ["ecog", 1]),
        (9, "inc", ["lab", "hgb", 90, None, True, None]),
        (11, "inc", ["lab", "plt", 100, None, True, None]),
        (12, "inc", ["lab", "anc", 1.5, None, True, None]),
        (14, "inc", ["uln", "bili", "<=", 1.5, None, None]),
        (15, "inc", ["uln", "creat", "<=", 1.5, None, None]),
        (24, "exc", ["stage", ["IV"]]),
        (25, "exc", ["malig", 1825, ["basal cell carcinoma of the skin", "squamous cell carcinoma of the skin",
                                      "carcinoma in situ of the cervix"]]),
        (28, "exc", ["within", MAJOR, 28, {}])]),
    ("NCT06611813", "breast", [
        (2, "inc", ["all", ["sex", "F"], ["age", 18, 75]]),
        (3, "inc", ["ecog", 1]),
        (13, "inc", ["all", ["lab", "anc", 1.5, None, False, None], ["lab", "plt", 75, None, False, None],
                     ["lab", "hgb", 90, None, False, None], ["lab", "lym", 1.5, None, True, None]]),
        (15, "inc", ["lab", "lvef", 55, None, False, None]),
        (22, "exc", ["stage", ["IV"]]),
        (26, "exc", ["within", MAJOR, 28, {"not_for": "breast"}])]),
    ("NCT07590531", "nsclc", [
        (3, "inc", ["age", 18, 75]),
        (15, "inc", ["ecog", 1]),
        (25, "exc", ["hist", ["small cell carcinoma", "combined small cell carcinoma"]]),
        (32, "exc", ["within", "live_vaccine", 28, {}]),
        (34, "exc", ["within", "radiotherapy", 21, {"palliative_ok_after": 14}]),
        (35, "exc", ["within", "immunosuppressive", 14, {}]),
        (39, "exc", ["malig", 1825, []]),
        (43, "exc", ["cond", "ibd", "ever"]),
        (50, "exc", ["cond", "hiv", "ever"]),
        (55, "exc", ["any", ["preg"], ["lact"]])]),
    ("NCT06170788", "nsclc", [
        (6, "inc", ["ecog", 1]),
        (12, "exc", ["hist", ["small cell carcinoma", "combined small cell carcinoma"]]),
        (19, "exc", ["within", "anticancer", 28, {}]),
        (21, "exc", ["within", "radiotherapy", 14, {}]),
        (22, "exc", ["within", "live_vaccine", 30, {}]),
        (25, "exc", ["cond", "cns_mets", "ever"]),
        (31, "exc", ["all", ["cond", "hbv", "active"], ["cond", "hcv", "active"]]),
        (33, "exc", ["cond", "allo_transplant", "ever"])]),
    ("NCT06623422", "nsclc", [
        (5, "inc", ["ecog", 1]),
        (16, "exc", ["hist", ["small cell carcinoma", "combined small cell carcinoma"]]),
        (17, "exc", ["marker", "ALK", "positive"]),
        (19, "exc", ["ever_cat", "checkpoint"]),
        (22, "exc", ["within", "radiotherapy", 14, {}]),
        (23, "exc", ["within", "live_vaccine", 30, {}])]),
    ("NCT07365319", "nsclc", [
        (2, "inc", ["age", 18, None]),
        (4, "inc", ["all", ["stage", ["IVA", "IVB"]],
                    ["not", ["hist", ["small cell carcinoma", "combined small cell carcinoma"]]]]),
        (10, "inc", ["ecog", 1]),
        (15, "exc", ["hist", ["small cell carcinoma", "combined small cell carcinoma"]]),
        (17, "exc", ["within", MAJOR, 21, {}]),
        (18, "exc", ["within", "live_vaccine", 30, {}]),
        (19, "exc", ["within", "radiotherapy", 7, {}])]),
    ("NCT05722795", "breast", [
        (3, "inc", ["age", 18, None]),
        (10, "inc", ["ecog", 1]),
        (14, "inc", ["lab", "plt", 100, None, True, None]),
        (16, "inc", ["uln", "bili", "<=", 1.5, 3, "gilbert"]),
        (19, "inc", ["lab", "creat", None, 110 / 88.4, None, False]),
        (33, "exc", ["any", ["preg"], ["lact"]]),
        (35, "exc", ["cond", "hiv", "ever"]),
        (36, "exc", ["any", ["cond", "hbv", "active"], ["cond", "hcv", "active"]])]),
    ("NCT07053306", "t2d", [
        (4, "inc", ["age", 30, 75]),
        (5, "inc", ["bmi", 21, 45, True, True]),
        (6, "inc", ["lab", "hba1c", 7.0, 11.0, True, True]),
        (7, "inc", ["lab", "egfr", 60, None, False, None]),
        (14, "inc", ["not", ["within", "sglt2", 60, {}]]),
        (21, "exc", ["any", ["current", "tzd"], ["current", "insulin"]])]),
    ("NCT07307235", "t2d", [
        (2, "inc", ["age", 18, None]),
        (5, "inc", ["lab", "hba1c", 7.5, 11.0, True, True]),
        (13, "exc", ["not", ["bmi", 20, 40, True, False]]),
        (18, "exc", ["lab", "egfr", None, 30, None, False]),
        (19, "exc", ["any", ["preg"], ["lact"]])]),
    ("NCT06838416", "t2d", [
        (3, "inc", ["age", 18, None]),
        (11, "exc", ["lab", "k", 4.8, None, False, None]),
        (13, "exc", ["lab", "hba1c", 9.0, None, False, None]),
        (19, "exc", ["any", ["uln", "alt", ">", 2.5, None, None], ["uln", "ast", ">", 2.5, None, None],
                     ["uln", "bili", ">", 2, None, None]]),
        (21, "exc", ["malig", 1825, []])]),
    ("NCT07664553", "t2d", [
        (2, "inc", ["dx_days", 90]),
        (4, "inc", ["lab", "hba1c", 7.0, 10.5, True, True]),
        (5, "inc", ["bmi", 23, None, True, None]),
        (14, "exc", ["cond", "pancreatitis", "ever"]),
        (15, "exc", ["cond", "nyha4", "active"]),
        (16, "exc", ["cond", "mtc_men2", "ever_or_family"])]),
    ("NCT07613307", "t2d", [
        ("reg:ages", "inc", ["age", 18, 65]),
        (3, "inc", ["lab", "hba1c", 7.0, 9.5, True, False]),
        (6, "inc", ["bmi", 25, None, True, None]),
        (12, "exc", ["cond", "pancreatitis", "ever"]),
        (15, "exc", ["proc_ever", "cholecystectomy"]),
        (16, "exc", ["cond", "nyha4", "active"])]),
    ("NCT06884059", "t2d", [
        (2, "inc", ["age", 18, 75]),
        (9, "inc", ["lab", "egfr", 45, None, False, None]),
        (15, "exc", ["any", ["current", "insulin"], ["current", "sulfonylurea"], ["lab", "hba1c", 9.0, None, False, None]]),
        (21, "exc", ["any", ["preg"], ["lact"]]),
        (39, "exc", ["cond", "ckd45", "ever"]),
        (40, "exc", ["cond", "hiv", "ever"])]),
    ("NCT06463483", "t2d", [
        (2, "inc", ["age", 18, 70]),
        (4, "inc", ["lab", "hba1c", None, 10.5, None, False]),
        (15, "exc", ["cond", "dka", ["within", 180]]),
        (16, "exc", ["within", "steroid_systemic", 28, {"stable_ok_over": 56}]),
        (19, "exc", ["preg"])]),
]
