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
import ast
import functools
import json
import math
import os
import random
import re
import statistics

SECTORS = ["Industrials", "Technology", "Healthcare", "Consumer", "Energy", "Materials", "Utilities", "Telecom"]
FX = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067, "CHF": 1.12}  # USD per unit of currency
SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CHF": "CHF "}
SUFFIX = {"USD": ["Inc.", "Corp.", "Holdings Inc."], "EUR": ["AG", "SE", "N.V."], "GBP": ["plc", "Group plc"],
          "JPY": ["K.K.", "Corporation"], "CHF": ["AG", "Holding AG"]}
MONTH = {3: ("March", 31), 6: ("June", 30), 9: ("September", 30), 12: ("December", 31)}
METRICS = ("margin", "growth", "net_debt", "eps")
REL_TOL = 0.005
# Absolute floors so answers rounded to the requested decimals still count, even for values near zero.
ABS_TOL = {"margin": 0.02, "growth": 0.02, "net_debt": 0.2, "eps": 0.01}
TOP_K = 20
WEIGHTS = {"cells": 0.4, "precision_at_k": 0.2, "kendall": 0.1, "top_values": 0.1, "sector_medians": 0.2}
MAX_CHARS = 8000


def _r100(x):
    # Ledger amounts are thousands of local currency, kept at multiples of 100 so a "millions, 1 decimal" table is exact.
    return int(round(x / 100.0)) * 100


def _split(total, k, rng):
    weights = [rng.uniform(0.6, 1.8) for _ in range(k)]
    parts = [_r100(total * w / sum(weights)) for w in weights[:-1]]
    return parts + [total - sum(parts)]


def _pnl(rev, op, interest, tax_rate, nci_rate):
    pretax = op - interest
    tax = _r100(pretax * tax_rate)
    ni = pretax - tax
    nci = _r100(ni * nci_rate)
    return {"rev": rev, "op": op, "interest": interest, "pretax": pretax, "tax": tax, "ni": ni, "nci": nci,
            "ni_attr": ni - nci}


def make_ledger(seed, i):
    rng = random.Random(f"finance-ledger:{seed}:{i}")
    cur = rng.choices(list(FX), weights=[50, 20, 15, 8, 7])[0]
    syl = ["Ar", "Bel", "Cor", "Dra", "El", "Fen", "Gal", "Hal", "Ir", "Jun", "Kel", "Lum", "Mar", "Nor", "Or", "Pra",
           "Quin", "Ros", "Sol", "Tor", "Ul", "Ver", "Wes", "Xen", "Yar", "Zan"]
    tail = ["ex", "ion", "ara", "ica", "on", "ent", "is", "ova", "ix", "um", "ant", "ella"]
    rev_prior = _r100(math.exp(rng.uniform(math.log(150), math.log(25000))) / FX[cur] * 1000)
    rev_prior_orig = _r100(rev_prior * rng.uniform(1.05, 1.15)) if rng.random() < 0.25 else rev_prior
    acq = _r100(rev_prior * rng.uniform(0.04, 0.25)) if rng.random() < 0.35 else 0
    rev = _r100(rev_prior * (1 + rng.uniform(-0.15, 0.35))) + acq
    op = _r100(rev * rng.uniform(0.03, 0.30))
    op_orig = op + (_r100(rev * rng.uniform(0.01, 0.04)) if rng.random() < 0.35 else 0)
    st_debt, cur_ltd, ltd = (_r100(rev * rng.uniform(0, hi)) for hi in (0.05, 0.04, 0.5))
    conv = _r100(rev * rng.uniform(0.02, 0.15)) if rng.random() < 0.3 else 0
    split = 1.0
    if rng.random() < 0.3:
        split = float(rng.choice([2, 3, 4, 5, 10]))
    elif rng.random() < 0.07:
        split = 1.0 / rng.choice([4, 5, 10])  # reverse split
    fy_month = rng.choice([12, 12, 12, 3, 6, 9])
    return {
        "id": f"C{i:04d}", "name": f"{rng.choice(syl)}{rng.choice(tail)} {rng.choice(syl)}{rng.choice(tail)} "
                                   f"{rng.choice(SUFFIX[cur])}",
        "sector": rng.choice(SECTORS), "currency": cur, "fy_month": fy_month,
        "fy_year": 2025 if fy_month in (9, 12) else 2026, "unit": rng.choice(["thousands", "millions"]),
        "rev": rev, "rev_prior": rev_prior, "rev_prior_orig": rev_prior_orig, "acq": acq,
        "prior_acq": _r100(rev_prior * rng.uniform(0.03, 0.15)) if not acq and rng.random() < 0.25 else 0,
        "op": op, "op_orig": op_orig, "op_prior": _r100(rev_prior_orig * rng.uniform(0.03, 0.28)),
        "interest": min(_r100((st_debt + cur_ltd + ltd + conv) * rng.uniform(0.02, 0.06)), _r100(op * 0.3)),
        "tax_rate": round(rng.uniform(0.18, 0.30), 3), "nci_rate": round(rng.uniform(0.03, 0.12), 3)
        if rng.random() < 0.3 else 0.0,
        "cash": _r100(rev * rng.uniform(0.03, 0.3)), "sti": _r100(rev * rng.uniform(0, 0.1)),
        "st_debt": st_debt, "cur_ltd": cur_ltd, "ltd": ltd, "conv": conv,
        "leases": _r100(rev * rng.uniform(0.01, 0.1)),
        "shares_k": max(_r100(rev_prior * FX[cur] * rng.uniform(0.02, 0.4)), 1000),  # EPS of roughly 0.3-5 USD
        "split": split, "past_split": split == 1.0 and rng.random() < 0.25,
    }


def company_metrics(L):
    fin = _pnl(L["rev"], L["op"], L["interest"], L["tax_rate"], L["nci_rate"])
    fx = FX[L["currency"]]
    return {
        "margin": 100.0 * L["op"] / L["rev"],
        "growth": 100.0 * (L["rev"] - L["acq"]) / L["rev_prior"] - 100.0,
        "net_debt": (L["st_debt"] + L["cur_ltd"] + L["ltd"] + L["conv"] - L["cash"]) / 1000.0 * fx,
        "eps": fin["ni_attr"] / (L["shares_k"] * L["split"]) * fx,
    }


def aggregate(per_company, sectors):
    ranked = sorted(per_company, key=lambda c: (-per_company[c]["growth"], c))
    by_sector = {}
    for c, m in per_company.items():
        by_sector.setdefault(sectors[c], []).append(m["margin"])
    k = max(1, min(TOP_K, len(ranked) // 2))  # at small n a top-20 would include everyone and be free
    return {"top20": [[c, per_company[c]["growth"]] for c in ranked[:k]],
            "sector_medians": {s: statistics.median(v) for s, v in sorted(by_sector.items())}}


# ---------- filing text ----------

def _fmt(v, unit):
    if isinstance(v, str):
        return v
    s = f"{abs(v):,}" if unit == "thousands" else f"{abs(v) / 1000:,.1f}"
    return f"({s})" if v < 0 else s


def _money(L, v):
    return f"{SYMBOL[L['currency']]}{v / 1000:,.1f} million"


RISK_TOPICS = ["supply chain disruption", "cybersecurity incidents", "interest rate movements", "foreign exchange",
               "changes in regulation", "competitive pressure", "retention of key personnel", "climate-related events",
               "litigation", "pension obligations", "changes in tax law", "customer concentration", "inflation",
               "protection of intellectual property", "product liability", "data privacy", "geopolitical tension",
               "counterparty credit", "technological change", "impairment of goodwill"]
RISK_SENTENCES = [
    "{name} is exposed to {topic}, which could adversely affect its results of operations.",
    "Developments related to {topic} may require us to change how we operate in {region}.",
    "Although we maintain policies to monitor {topic}, these policies may not be effective in all circumstances.",
    "Our competitors in the {sector} sector face similar exposure to {topic}, but may be better capitalised.",
    "A significant deterioration in conditions related to {topic} could reduce demand for our products.",
    "We cannot predict with certainty the timing or magnitude of any impact from {topic}.",
    "Management reviews {topic} quarterly with the Audit Committee of the Board of Directors.",
    "Insurance coverage for losses arising from {topic} may be unavailable or insufficient.",
    "In {region}, {topic} has historically been more pronounced than in our other markets.",
    "Any failure to respond to {topic} could damage our reputation with customers and investors.",
    "Mitigation of {topic} may increase our operating expenses in future periods.",
    "The effects of {topic} may be amplified by other risks described in this section.",
]
MDA_SENTENCES = [
    "Management continued to focus on disciplined capital allocation and operational efficiency during the year.",
    "Pricing actions taken in the first half partly offset input cost inflation in {region}.",
    "Investment in digital capabilities remained a priority across the {sector} portfolio.",
    "Working capital management improved, with lower inventory days compared to the prior year.",
    "The order backlog at year end provides visibility into the first half of the next fiscal year.",
    "Headcount was broadly stable, while productivity per employee increased.",
    "We expect market conditions in {region} to remain mixed over the coming year.",
    "Capital expenditure was directed primarily at capacity expansion and maintenance.",
]
REGIONS = ["North America", "Europe", "Asia-Pacific", "Latin America", "the Middle East", "Japan", "the United Kingdom"]


def _para(rng, pool, L, n, **kw):
    return " ".join(t.format(name=L["name"], sector=L["sector"].lower(), region=rng.choice(REGIONS),
                                            **kw) for t in rng.sample(pool, n))


def render_filing(L, seed):
    rng = random.Random(f"finance-text:{seed}:{L['id']}")
    cur, base = L["currency"], L["unit"]
    month, day = MONTH[L["fy_month"]]
    y = L["fy_year"]
    cur_label, prior_label = f"FY{y} ({month} {day}, {y})", f"FY{y - 1} ({month} {day}, {y - 1})"
    fin_o = _pnl(L["rev"], L["op_orig"], L["interest"], L["tax_rate"], L["nci_rate"])
    fin_f = _pnl(L["rev"], L["op"], L["interest"], L["tax_rate"], L["nci_rate"])
    fin_p = _pnl(L["rev_prior_orig"], L["op_prior"], _r100(L["interest"] * rng.uniform(0.8, 1.2)), L["tax_rate"],
                 L["nci_rate"])
    shares_unit = rng.choice(["thousands", "millions"])
    shares_p = _r100(L["shares_k"] * rng.uniform(0.95, 1.03))
    segs = rng.sample(["Industrial Solutions", "Digital Services", "Consumer Products", "Energy Systems", "Life Sciences",
                       "Materials", "Networks", "Infrastructure", "Software", "Specialty Products"], rng.randint(2, 4))
    note = iter(range(1, 30))
    acq_name = f"{rng.choice(['Nova', 'Brightline', 'Keystone', 'Atlas', 'Meridian'])} {rng.choice(['Labs', 'Systems', 'GmbH', 'Ltd'])}"

    def table(title, rows, unit=None, extra=""):
        # Each table may use the other unit and either column order: both are deliberate reading traps.
        unit = unit or (base if rng.random() < 0.7 else ("millions" if base == "thousands" else "thousands"))
        first = rng.random() < 0.6
        head = [cur_label, prior_label] if first else [prior_label, cur_label]
        out = [f"**{title}** (in {unit} of {cur}{extra})", "", f"| | {head[0]} | {head[1]} |", "|---|---:|---:|"]
        for label, c, p in rows:
            a, b = (c, p) if first else (p, c)
            out.append(f"| {label} | {_fmt(a, unit)} | {_fmt(b, unit)} |")
        return "\n".join(out) + "\n"

    def eps_str(ni, shares):
        return f"{ni / shares:,.2f}"

    def sh(v):
        return _fmt(v, shares_unit)

    s = [f"# {L['name']} - Annual Report for the fiscal year ended {month} {day}, {y}\n",
         f"Company ID: {L['id']}  \nSector: {L['sector']}  \nReporting currency: {cur}  \n"
         f"Fiscal year end: {month} {day}\n"]

    s.append("## 1. Business Overview\n")
    s.append(f"{L['name']} operates in the {L['sector'].lower()} sector through {len(segs)} reportable segments: "
             f"{', '.join(segs)}. " + _para(rng, MDA_SENTENCES, L, 4) + "\n")

    s.append("## 2. Risk Factors\n")
    risk_at = len(s)

    reported_growth = 100.0 * L["rev"] / L["rev_prior_orig"] - 100.0
    s.append("## 3. Management's Discussion and Analysis\n")
    s.append(f"Revenue for fiscal {y} was {_money(L, L['rev'])}, compared with {_money(L, L['rev_prior_orig'])} in "
             f"fiscal {y - 1}, a change of {reported_growth:+.1f}%. On a constant-currency basis revenue changed by "
             f"{reported_growth + rng.uniform(-3, 3):+.1f}%. Operating income was {_money(L, L['op_orig'])}, "
             f"an operating margin of {100.0 * L['op_orig'] / L['rev']:.1f}%. " + _para(rng, MDA_SENTENCES, L, 3) + "\n")
    if L["acq"]:
        s.append(f"Revenue growth benefited from the acquisition of {acq_name} during the year (see the notes on "
                 "business combinations). " + _para(rng, MDA_SENTENCES, L, 2) + "\n")
    s.append(f"Liquidity: cash and cash equivalents at year end were {_money(L, L['cash'])} and short-term investments "
             f"were {_money(L, L['sti'])}. " + _para(rng, MDA_SENTENCES, L, 2) + "\n")

    s.append("## 4. Selected Financial Data\n")
    revs, ops = [L["rev"], L["rev_prior_orig"]], [L["op_orig"], L["op_prior"]]
    for _ in range(3):
        revs.append(_r100(revs[-1] / rng.uniform(0.9, 1.2)))
        ops.append(_r100(revs[-1] * rng.uniform(0.03, 0.25)))
    u = rng.choice(["thousands", "millions"])
    s.append(f"**Five-year summary** (in {u} of {cur})\n\n| Fiscal year | Revenue | Operating income |\n|---|---:|---:|\n"
             + "".join(f"| FY{y - k} | {_fmt(revs[k], u)} | {_fmt(ops[k], u)} |\n" for k in range(5)))

    s.append("## 5. Consolidated Income Statement\n")
    cogs_c, cogs_p = _r100(L["rev"] * rng.uniform(0.35, 0.6)), _r100(L["rev_prior_orig"] * rng.uniform(0.35, 0.6))
    rows = [("Revenue", L["rev"], L["rev_prior_orig"]), ("Cost of revenue", -cogs_c, -cogs_p),
            ("Gross profit", L["rev"] - cogs_c, L["rev_prior_orig"] - cogs_p),
            ("Selling, general and administrative", -(L["rev"] - cogs_c - L["op_orig"]),
             -(L["rev_prior_orig"] - cogs_p - L["op_prior"])),
            ("Operating income", fin_o["op"], fin_p["op"]), ("Interest expense", -fin_o["interest"], -fin_p["interest"]),
            ("Income before income taxes", fin_o["pretax"], fin_p["pretax"]),
            ("Income tax expense", -fin_o["tax"], -fin_p["tax"]), ("Net income", fin_o["ni"], fin_p["ni"])]
    if L["nci_rate"]:
        rows += [("Less: attributable to non-controlling interests", -fin_o["nci"], -fin_p["nci"]),
                 ("Net income attributable to shareholders", fin_o["ni_attr"], fin_p["ni_attr"])]
    rows += [(f"Weighted-average diluted shares (in {shares_unit})", sh(L["shares_k"]), sh(shares_p)),
             (f"Diluted earnings per share ({cur})", eps_str(fin_o["ni_attr"], L["shares_k"]),
              eps_str(fin_p["ni_attr"], shares_p))]
    s.append(table("Consolidated income statement", rows, extra=", except share and per-share data"))

    s.append("## 6. Consolidated Balance Sheet\n")
    vary = lambda v: _r100(v * rng.uniform(0.8, 1.2))
    other_ncl = _r100(L["rev"] * rng.uniform(0.01, 0.06))
    rows = [("Cash and cash equivalents", L["cash"], vary(L["cash"])), ("Short-term investments", L["sti"], vary(L["sti"])),
            ("Trade receivables", _r100(L["rev"] * 0.15), vary(L["rev"] * 0.15)),
            ("Inventories", _r100(L["rev"] * 0.1), vary(L["rev"] * 0.1)),
            ("Property, plant and equipment", _r100(L["rev"] * 0.4), vary(L["rev"] * 0.4)),
            ("Goodwill", _r100(L["rev"] * 0.2 + L["acq"]), vary(L["rev"] * 0.2)),
            ("Short-term borrowings", L["st_debt"], vary(L["st_debt"])),
            ("Current portion of long-term debt", L["cur_ltd"], vary(L["cur_ltd"])),
            ("Trade payables", _r100(L["rev"] * 0.12), vary(L["rev"] * 0.12)),
            ("Long-term debt", L["ltd"], vary(L["ltd"])), ("Lease liabilities", L["leases"], vary(L["leases"])),
            ("Other non-current liabilities", other_ncl + L["conv"], vary(other_ncl))]
    s.append(table("Consolidated balance sheet (selected lines)", rows))

    s.append("## 7. Segment Information\n")
    rc, rp, oc = (_split(t, len(segs), rng) for t in (L["rev"], L["rev_prior_orig"], L["op_orig"]))
    u = rng.choice(["thousands", "millions"])
    s.append(f"**Segment results** (in {u} of {cur})\n\n| Segment | Revenue FY{y} | Revenue FY{y - 1} | "
             f"Operating income FY{y} |\n|---|---:|---:|---:|\n"
             + "".join(f"| {g} | {_fmt(a, u)} | {_fmt(b, u)} | {_fmt(c, u)} |\n" for g, a, b, c in zip(segs, rc, rp, oc)))
    s.append(" ".join(f"{g} revenue was {_money(L, a)} with segment operating income of {_money(L, c)}. "
                      + _para(rng, MDA_SENTENCES, L, 1) for g, a, c in zip(segs, rc, oc)) + "\n")

    s.append("## 8. Quarterly Results (unaudited)\n")
    q = _split(L["rev"], 4, rng)
    u = rng.choice(["thousands", "millions"])
    s.append(f"**Quarterly revenue, fiscal {y}** (in {u} of {cur})\n\n| Q1 | Q2 | Q3 | Q4 |\n|---:|---:|---:|---:|\n| "
             + " | ".join(_fmt(v, u) for v in q) + " |\n")

    s.append("## 9. Non-GAAP Reconciliation\n")
    amort, restr = _r100(L["rev"] * rng.uniform(0.005, 0.03)), _r100(L["rev"] * rng.uniform(0, 0.02))
    s.append(table("Adjusted operating income", [
        ("Operating income (GAAP)", L["op_orig"], L["op_prior"]), ("Amortisation of acquired intangibles", amort,
                                                                     vary(amort)),
        ("Restructuring costs", restr, vary(restr)), ("Adjusted operating income", L["op_orig"] + amort + restr, "n/a")]))
    company_nd = L["st_debt"] + L["cur_ltd"] + L["ltd"] + L["leases"] - L["cash"] - L["sti"]
    s.append(f"Management defines net debt as total borrowings plus lease liabilities, less cash and short-term "
             f"investments. On this basis net debt was {_money(L, company_nd)}. Adjusted diluted EPS, which excludes "
             f"amortisation and restructuring after tax, was {cur} "
             f"{(fin_o['ni_attr'] + (amort + restr) * (1 - L['tax_rate'])) / L['shares_k']:,.2f}.\n")

    s.append("## 10. Notes to the Financial Statements\n")
    s.append(f"### Note {next(note)} - Basis of presentation\nThe consolidated financial statements are presented in "
             f"{cur}. Tables state their own unit (thousands or millions); amounts in the narrative are in millions. "
             f"The fiscal year ends on {month} {day}. " + _para(rng, MDA_SENTENCES, L, 2) + "\n")
    if L["acq"] or L["prior_acq"]:
        u = rng.choice(["thousands", "millions"])
        when, amt = (f"fiscal {y}", L["acq"]) if L["acq"] else (f"fiscal {y - 1}", L["prior_acq"])
        s.append(f"### Note {next(note)} - Business combinations\nDuring {when} the Company acquired 100% of {acq_name} "
                 f"for cash consideration. Since the acquisition date, {acq_name} contributed revenue of "
                 f"{_fmt(amt, u)} (in {u} of {cur}) in {when}, which is included in consolidated revenue. "
                 "Goodwill arising on the acquisition reflects expected synergies.\n")
    u = rng.choice(["thousands", "millions"])
    loans = _split(L["ltd"], 2, rng) if L["ltd"] >= 200 else [L["ltd"], 0]
    debt = f"### Note {next(note)} - Borrowings\n**Borrowings at {month} {day}, {y}** (in {u} of {cur})\n\n" \
           f"| Instrument | Amount |\n|---|---:|\n| Commercial paper and bank overdrafts | {_fmt(L['st_debt'], u)} |\n" \
           f"| Current portion of long-term debt | {_fmt(L['cur_ltd'], u)} |\n| Term loan facility | {_fmt(loans[0], u)} " \
           f"|\n| Senior notes | {_fmt(loans[1], u)} |\n| Total borrowings | " \
           f"{_fmt(L['st_debt'] + L['cur_ltd'] + L['ltd'], u)} |\n"
    if L["conv"]:
        u = rng.choice(["thousands", "millions"])
        debt += (f"\nIn addition, the Company has convertible senior notes outstanding with a carrying amount of "
                 f"{_fmt(L['conv'], u)} (in {u} of {cur}). These notes are presented within 'Other non-current "
                 "liabilities' on the balance sheet and are not included in the table above.\n")
    s.append(debt)
    s.append(f"### Note {next(note)} - Leases\nLease liabilities of {_money(L, L['leases'])} are presented separately "
             "and are not included in borrowings. " + _para(rng, MDA_SENTENCES, L, 1) + "\n")
    s.append(f"### Note {next(note)} - Income taxes\nThe effective tax rate for fiscal {y} was "
             f"{100 * fin_o['tax'] / fin_o['pretax']:.1f}%. " + _para(rng, MDA_SENTENCES, L, 1) + "\n")
    shares = f"### Note {next(note)} - Share capital\n"
    if L["past_split"]:
        shares += (f"In fiscal {y - 1} the Company completed a 2-for-1 stock split. All share and per-share amounts in "
                   "this report have been retroactively adjusted to reflect that split.\n")
    else:
        shares += "There were no changes to the Company's share structure during the year.\n"
    s.append(shares)
    if L["rev_prior_orig"] != L["rev_prior"]:
        u = rng.choice(["thousands", "millions"])
        s.append(f"### Note {next(note)} - Discontinued operations and re-presentation\nDuring fiscal {y} the Company "
                 f"classified a non-core business as discontinued. Revenue for fiscal {y - 1} has been re-presented on "
                 f"a continuing basis as {_fmt(L['rev_prior'], u)} (in {u} of {cur}); the income statement, segment "
                 f"table and selected financial data above show fiscal {y - 1} as originally reported.\n")
    if L["op_orig"] != L["op"]:
        u = rng.choice(["thousands", "millions"])
        s.append(f"### Note {next(note)} - Correction of an error (restatement)\nAfter the statements above were "
                 f"prepared, the Company identified that certain contract costs had been capitalised in error. "
                 f"Operating income for fiscal {y} is restated from {_fmt(L['op_orig'], u)} to {_fmt(L['op'], u)}; "
                 f"income tax expense is restated to {_fmt(fin_f['tax'], u)}; net income attributable to shareholders "
                 f"is restated to {_fmt(fin_f['ni_attr'], u)} (all in {u} of {cur}). Revenue is unaffected. The "
                 "primary statements, segment table and non-GAAP measures in this report have not been updated.\n")
    if L["split"] != 1.0:
        when = f"{MONTH[3 if L['fy_month'] == 12 else L['fy_month'] + 3][0]} 15"
        kind = f"a {int(L['split'])}-for-1 stock split" if L["split"] > 1 else \
            f"a 1-for-{round(1 / L['split'])} reverse stock split"
        s.append(f"### Note {next(note)} - Subsequent events\nOn {when}, after the fiscal year end, the Company "
                 f"effected {kind}. Share counts and per-share amounts in this report have not been adjusted for "
                 "it.\n")

    # Pad the risk factor section until the filing reaches its target length (~4-10k tokens at ~4 chars/token).
    target = rng.randint(16000, 38000)
    risks, topics = [], rng.sample(RISK_TOPICS, len(RISK_TOPICS))
    while sum(map(len, s)) + sum(map(len, risks)) < target:
        t = topics[len(risks) % len(topics)]
        risks.append(f"### Risks related to {t}\n" + _para(rng, RISK_SENTENCES, L, 5, topic=t) + "\n\n"
                     + _para(rng, RISK_SENTENCES, L, 5, topic=t) + "\n")
    s[risk_at:risk_at] = risks
    return "\n".join(s)


def _prompt(n, sectors_present):
    fx = ", ".join(f"1 {c} = {r} USD" for c, r in FX.items())
    return f"""You are given {n} synthetic annual-report filings (one per company). Use the tools list_filings, read_filing, search_filing and calculator.

For EVERY company compute, for its most recent fiscal year (the latest year presented in its filing):
- margin: restated operating margin in percent = operating income / revenue * 100. If the notes correct or restate operating income, use the corrected figure.
- growth: organic revenue growth in percent = (current-year revenue - revenue contributed in the current year by businesses acquired during the current year) / prior-year revenue * 100 - 100. If the notes re-present or restate prior-year revenue, use the re-presented figure. Acquisitions made in earlier years are not excluded.
- net_debt: in USD millions = short-term borrowings + current portion of long-term debt + long-term debt + any other borrowings the notes say are presented within other balance-sheet lines (e.g. convertible notes) - cash and cash equivalents, at the current fiscal year end. Exclude lease liabilities and short-term investments; ignore management's own non-GAAP net debt.
- eps: GAAP diluted EPS in USD = net income attributable to shareholders (corrected, if the notes restate it) / weighted-average diluted shares, adjusted for any stock split effected after year end that the reported figures do not yet reflect.
Watch for tables in thousands vs millions, column order (check the year headers), and notes that override tables.
Convert to USD at these fixed rates: {fx}.

Then aggregate:
- top20: the {max(1, min(TOP_K, n // 2))} companies with the highest organic growth, sorted descending, with their growth values.
- sector_medians: median restated operating margin (percent) per sector over all companies in that sector (mean of the two middle values for even counts). Sectors present: {", ".join(sectors_present)}.

Final answer: a single JSON object, no other text:
{{"per_company": {{"C0000": {{"margin": 12.34, "growth": -3.21, "net_debt": 456.7, "eps": 1.234}}, ...}},
 "top20": [["C0007", 31.42], ...],
 "sector_medians": {{"Energy": 14.2, ...}}}}
Round margin/growth to 2 decimals, net_debt to 1 decimal, eps to 3 decimals. Values within 0.5% (or the rounding) of the exact answer count as correct."""


def generate(n, seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ledgers = [make_ledger(seed, i) for i in range(n)]
    per = {L["id"]: company_metrics(L) for L in ledgers}
    sectors = {L["id"]: L["sector"] for L in ledgers}
    with open(os.path.join(out_dir, "filings.jsonl"), "w") as f:
        for L in ledgers:
            row = {k: L[k] for k in ("id", "name", "sector", "currency")}
            row["fy_end"] = MONTH[L["fy_month"]][0]
            row["text"] = render_filing(L, seed)
            f.write(json.dumps(row) + "\n")
    gold_path = os.path.join(out_dir, "gold.json")
    with open(gold_path, "w") as f:
        json.dump({"n": n, "seed": seed, "ledgers": ledgers, "per_company": per, "sectors": sectors,
                   **aggregate(per, sectors)}, f)
    return {"id": f"finance-n{n}-s{seed}", "prompt": _prompt(n, sorted(set(sectors.values()))), "world_dir": out_dir,
            "n_units": n, "gold_path": gold_path}


# ---------- tools ----------

@functools.lru_cache(maxsize=8)
def _load(world_dir):
    filings = {}
    with open(os.path.join(world_dir, "filings.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            heads = [m.start() for m in re.finditer(r"^## ", row["text"], re.M)] + [len(row["text"])]
            row["sections"] = [(row["text"][a:row["text"].index("\n", a)].lstrip("# ").strip(), a, b)
                               for a, b in zip(heads, heads[1:])]
            filings[row["id"]] = row
    return filings


def _cap(text):
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "\n...[truncated]"


SAFE_FUNCS = {"abs": abs, "round": lambda x, d=0: round(x, int(d)), "min": min, "max": max, "sum": sum, "len": len, "sorted": sorted,
              "median": statistics.median, "mean": statistics.mean}
BIN_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
           ast.Pow: lambda a, b: a ** b}


def _eval(node):
    # Floats only, so results overflow loudly instead of building huge ints; operators never act on lists.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
        a, b = _eval(node.left), _eval(node.right)
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            raise ValueError("operators apply to numbers only")
        return float(BIN_OPS[type(node.op)](float(a), float(b)))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCS and not node.keywords:
        return SAFE_FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"{type(node).__name__} not allowed (arithmetic only; functions: {', '.join(SAFE_FUNCS)})")


def calculate(expr):
    if len(expr) > 20000:
        return "error: expression too long"
    try:
        return _cap(repr(_eval(ast.parse(expr.replace("^", "**"), mode="eval").body)))
    except Exception as e:  # noqa: BLE001 - any evaluation error is reported back to the agent
        return f"error: {type(e).__name__}: {e}"


def tools(world_dir):
    filings = _load(world_dir)
    ids = sorted(filings)

    def get(args):
        f = filings.get(str(args.get("company_id", "")).strip())
        if f is None:
            raise KeyError(f"unknown company_id {args.get('company_id')!r}; use list_filings")
        return f

    def list_filings(args):
        off, lim = int(args.get("offset", 0)), min(int(args.get("limit", 100)), 150)
        rows = [f"{c} | {filings[c]['name']} | {filings[c]['sector']} | {filings[c]['currency']} | FY ends "
                f"{filings[c]['fy_end']}" for c in ids[off:off + lim]]
        return _cap(f"{len(ids)} filings; showing {off}-{off + len(rows) - 1}\n" + "\n".join(rows))

    def read_filing(args):
        f, off = get(args), max(int(args.get("offset", 0)), 0)
        text, toc = f["text"], ""
        sec = str(args.get("section", "") or "").strip().lower()
        if sec:
            hit = [x for x in f["sections"] if x[0].lower().startswith(sec) or sec in x[0].lower()]
            if not hit:
                return "error: no such section; sections: " + "; ".join(x[0] for x in f["sections"])
            text = text[hit[0][1]:hit[0][2]]
        elif off == 0:
            toc = "Sections: " + "; ".join(f"{t} ({b - a} chars)" for t, a, b in f["sections"]) + "\n\n"
        chunk = text[off:off + MAX_CHARS - len(toc) - 100]
        end = off + len(chunk)
        more = f"\n[chars {off}-{end} of {len(text)}; continue with offset={end}]" if end < len(text) else \
            f"\n[end; chars {off}-{end} of {len(text)}]"
        return toc + chunk + more

    def search_filing(args):
        f, q = get(args), str(args.get("text", "")).strip()
        if not q:
            return "error: empty search text"
        text, out = f["text"], []
        for m in re.finditer(re.escape(q), text, re.I):
            sec = next((t for t, a, b in f["sections"] if a <= m.start() < b), "header")
            out.append(f"[{sec} @ offset {m.start()}] ..." + text[max(0, m.start() - 200):m.end() + 200]
                       .replace("\n", " ") + "...")
            if len(out) == 15:
                break
        return _cap(f"{len(out)} match(es) for {q!r}\n" + "\n".join(out)) if out else f"no matches for {q!r}"

    def safe(fn):
        def run(args):
            try:
                return fn(args or {})
            except Exception as e:  # noqa: BLE001 - tool errors go back to the agent as text
                return f"error: {e}"
        return run

    cid = {"company_id": {"type": "string", "description": "e.g. C0001"}}
    return [
        {"name": "list_filings", "description": "List filings (id | name | sector | currency | fiscal year end month).",
         "parameters": {"type": "object", "properties": {"offset": {"type": "integer"}, "limit": {"type": "integer"}}},
         "fn": safe(list_filings)},
        {"name": "read_filing", "description": "Read a filing's text in chunks of ~8k chars. Optional section (number or "
                                               "title, e.g. '10' or 'Notes') and char offset within it.",
         "parameters": {"type": "object", "properties": {**cid, "section": {"type": "string"},
                                                         "offset": {"type": "integer"}}, "required": ["company_id"]},
         "fn": safe(read_filing)},
        {"name": "search_filing", "description": "Case-insensitive substring search in one filing; returns up to 15 "
                                                 "matches with surrounding context and offsets.",
         "parameters": {"type": "object", "properties": {**cid, "text": {"type": "string"}},
                        "required": ["company_id", "text"]},
         "fn": safe(search_filing)},
        {"name": "calculator", "description": "Evaluate an arithmetic expression: numbers, + - * / // % **, lists, and "
                                              "abs round min max sum len sorted median mean. No thousands separators.",
         "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
         "fn": lambda args: calculate(str((args or {}).get("expression", "")))},
    ]


# ---------- scoring ----------

def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, str):
        s = v.strip().replace(",", "").replace("%", "").replace("$", "").replace("USD", "").strip()
        if re.fullmatch(r"\(\d+(\.\d*)?\)", s):
            s = "-" + s[1:-1]
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _close(a, g, metric):
    a = _num(a)
    return a is not None and abs(a - g) <= max(REL_TOL * abs(g), ABS_TOL[metric])


def _parse(answer):
    if isinstance(answer, dict):
        return answer
    # The last JSON object with answer keys wins, so braces in surrounding prose or a draft block do not void the
    # answer. A truncated answer keeps its complete per-company rows.
    text, dec, found, i = str(answer or ""), json.JSONDecoder(), None, 0
    while (i := text.find("{", i)) >= 0:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict) and {"per_company", "top20", "sector_medians"} & set(obj):
            found = obj
        i = end
    if found is None:
        rows = {}
        for cid, body in re.findall(r'"(C\d{4})"\s*:\s*(\{[^{}]*\})', text):
            try:
                rows[cid] = json.loads(body)
            except ValueError:
                pass
        found = {"per_company": rows} if rows else None
    return found


def score(task, answer):
    with open(task["gold_path"]) as f:
        gold = json.load(f)
    obj = _parse(answer)
    zero = {"score": 0.0, "parse_error": obj is None, "n_correct": 0, "n_cells": 4 * gold["n"]}
    if obj is None:
        return zero

    per = obj.get("per_company") if isinstance(obj.get("per_company"), dict) else {}
    per_metric = {m: 0 for m in METRICS}
    for c, g in gold["per_company"].items():
        row = per.get(c) if isinstance(per.get(c), dict) else {}
        for m in METRICS:
            per_metric[m] += _close(row.get(m), g[m], m)
    n_correct = sum(per_metric.values())
    cells = n_correct / (4 * gold["n"])

    k = len(gold["top20"])
    gold_rank = {c: r for r, (c, _) in enumerate(gold["top20"])}
    sub, seen = [], set()
    for item in obj.get("top20") if isinstance(obj.get("top20"), list) else []:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and str(item[0]) not in seen:
            seen.add(str(item[0]))
            sub.append((str(item[0]), item[1]))
    sub = sub[:k]
    overlap = [c for c, _ in sub if c in gold_rank]
    p_at_k = len(overlap) / k
    pairs = [(i, j) for i in range(len(overlap)) for j in range(i + 1, len(overlap))]
    if pairs:
        tau = sum(1 if gold_rank[overlap[i]] < gold_rank[overlap[j]] else -1 for i, j in pairs) / len(pairs)
    else:
        tau = 1.0 if overlap else -1.0
    top_values = sum(c in gold_rank and _close(v, gold["per_company"][c]["growth"], "growth") for c, v in sub) / k

    med = obj.get("sector_medians") if isinstance(obj.get("sector_medians"), dict) else {}
    med_acc = sum(_close(med.get(s), g, "margin") for s, g in gold["sector_medians"].items()) / len(gold["sector_medians"])

    parts = {"cells": cells, "precision_at_k": p_at_k, "kendall": (tau + 1) / 2 * p_at_k, "top_values": top_values,
             "sector_medians": med_acc}
    return {"score": sum(WEIGHTS[x] * parts[x] for x in WEIGHTS), "parse_error": False, "n_correct": n_correct,
            "n_cells": 4 * gold["n"], "cell_accuracy": cells,
            **{f"acc_{m}": per_metric[m] / gold["n"] for m in METRICS}, "precision_at_k": p_at_k,
            "kendall_tau": tau, "top_value_accuracy": top_values, "sector_median_accuracy": med_acc}
