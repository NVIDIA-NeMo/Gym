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
import datetime as dt

from .world import INDUSTRY, T0, W0, W1, at, iso

SOURCES = ["reg", "crm", "sig", "ltr", "evt"]
EXTRA_W = {"crm": 0.25, "sig": 0.3, "ltr": 0.25, "evt": 0.2}
N_REC_W = [(4, 0.35), (5, 0.35), (6, 0.15), (7, 0.1), (8, 0.05)]
LEGAL_SUFFIX = {"inc", "llc", "ltd", "corp", "gmbh", "s.a", "sa"}

THINGS = ["invoice", "delivery schedule", "service contract", "quarterly statement", "renewal quote", "audit request",
          "shipment", "purchase order", "support ticket", "price list", "warranty claim", "onboarding pack"]
SIG_LINES = [
    "Thanks for the quick turnaround on the {t}.", "Following up on the {t} we discussed on {day}.",
    "Could you confirm the figures in the attached {t}? The total should be ${amt}.",
    "I have copied our finance team so they can reconcile the {t} on their side.",
    "We would like to move the review of the {t} to {day} afternoon if that still works.",
    "Our records show the {t} as still open, reference INV-{ref}.",
    "Please disregard my earlier note about the {t}; it was resolved this morning.",
    "Let me know whether you need anything else before the {t} goes out.",
    "The revised {t} is attached, with the changes highlighted in yellow.",
    "I will be out of the office until {day}, so please reply to all on this thread.",
    "Our legal team has reviewed the {t} and has no further comments at this stage.",
    "As agreed on the call, we will split the {t} into two parts and invoice them separately.",
    "The courier has confirmed collection for {day} between nine and twelve.",
    "Apologies for the delay in getting back to you; the {t} needed a second approval.",
    "If the numbers look right to you, I will release the {t} by end of day {day}.",
    "For reference, the original {t} was issued under PO {ref}.",
    "Happy to set up a short call on {day} to walk through the open items.",
    "We noticed a rounding difference of ${amt} between the {t} and our ledger.",
]
LTR_LINES = [
    "I am writing with regard to the {t} I received last month.",
    "According to my records the {t} was settled in full on {day}, yet a reminder arrived again.",
    "I would be grateful if you could review the charges listed under reference INV-{ref}.",
    "On two occasions the courier arrived while nobody was at home, and the parcel was returned to the depot.",
    "I enclose a copy of the {t} together with the receipt from the branch.",
    "Please make sure that all future correspondence is sent to the address given above.",
    "I have been a customer for many years and have always found your staff helpful and courteous.",
    "If there is any outstanding amount, kindly send me an itemised statement so that I can arrange payment.",
    "I would also like to cancel the paper copy of the {t} and receive it electronically instead.",
    "Your colleague at the counter advised me to put this request in writing, which I am now doing.",
    "The amount of ${amt} appears twice on the latest {t}, which I believe is an error.",
    "I trust that this matter can be resolved without further delay.",
    "I telephoned your office on {day} but was told that the person handling my file was away.",
    "The {t} I received was addressed incorrectly and only reached me by chance through a neighbour.",
    "I should be obliged if you would confirm in writing once the correction has been made.",
    "Kindly note that I do not wish to receive marketing material of any kind.",
    "I have also written to the branch manager about the same {t}, so far without reply.",
    "For the avoidance of doubt, I authorise no further direct debits until this is clarified.",
    "The previous reference I was given, INV-{ref}, does not appear on your online portal.",
    "Should you require any further documents, I will gladly provide them.",
]
CO_LTR_LINES = [
    "We are writing to inform our partners of changes to the {t} process effective {day}.",
    "All open orders will continue to be honoured under the existing {t}.",
    "Our accounts team will contact you separately regarding the {t} under reference INV-{ref}.",
    "We thank you for your continued business and look forward to working with you.",
    "Please update your supplier records with the details shown in this letterhead.",
    "A revised price list will follow, with an average adjustment of {pct} percent.",
    "Questions about the {t} may be directed to our customer desk during business hours.",
    "Following a review of our operations, invoices will in future be issued monthly rather than weekly.",
    "Our warehouse will be closed for stocktaking on {day} and deliveries will resume the following day.",
    "We remind partners that the {t} must quote our reference INV-{ref} to be processed.",
    "Payment terms remain unchanged at thirty days from the date of the {t}.",
    "We have appointed a new account team, who will introduce themselves in the coming weeks.",
    "Kindly destroy any stationery bearing our previous details.",
    "We appreciate your patience during the transition of our billing systems.",
]
CRM_NOTES = ["Prefers contact by email.", "Asked for a callback about the {t}.", "Payment terms net 30.",
             "Met at the regional trade fair.", "Escalated a complaint about the {t}; resolved.",
             "Do not call before 10am.", "Duplicate of an older account? not verified.", "Requested {t} copy.",
             "Discount of {pct} percent approved for the next {t}.", "Bounced mail once, re-sent by post.",
             "Interested in the premium tier; follow up next quarter.", "Credit check passed.",
             "Complained about the {t} wording; legal reviewed and closed.", "Card on file expired, asked to update.",
             "Sent the {t} twice by mistake; apology issued.", "Prefers {day} meetings."]
REG_OFFICES = ["Central", "Northern", "Harbour", "Eastern", "Western", "Southern", "Riverside"]
REG_NOTES = ["Filed in person at the registry office.", "Supporting documents verified by the clerk.",
             "Filed electronically.", "Fee paid in full.", "Certified copy issued on request."]
EVT_NOTES = ["source=branch", "source=web-portal", "source=batch-import", "source=call-centre", "source=partner-feed"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
REPS = ["J. Ilves", "M. Okafor", "T. Brandt", "S. Rao", "L. Fontaine", "K. Ward", "A. Mendes", "R. Haas"]


def fill(rng, lines, k):
    return " ".join(line.format(t=rng.choice(THINGS), day=rng.choice(DAYS), amt=f"{rng.randint(80, 99000):,}.00",
                                ref=rng.randint(10000, 99999), pct=rng.randint(2, 9))
                    for line in rng.sample(lines, min(k, len(lines))))


def typo(rng, s):
    words = s.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not idx:
        return s
    i = rng.choice(idx)
    w, j = words[i], rng.randint(1, len(words[i]) - 2)
    op = rng.randrange(4)
    if op == 0:
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    elif op == 1:
        w = w[:j] + w[j + 1:]
    elif op == 2:
        w = w[:j] + w[j] + w[j:]
    else:
        w = w[:j] + rng.choice([c for c in "aeiouy" if c != w[j]]) + w[j + 1:]
    words[i] = w
    return " ".join(words)


OCR = [("m", "rn"), ("l", "1"), ("O", "0"), ("e", "c"), ("i", "l"), ("S", "5"), ("h", "b")]


def ocr(rng, s):
    out = []
    for ch in s:
        sub = [b for a, b in OCR if a == ch]
        out.append(sub[0] if sub and rng.random() < 0.06 else ch)
    return "".join(out)


def phone_fmt(rng, p):
    a, b, c = p[:3], p[3:6], p[6:]
    return rng.choice([f"+1 ({a}) {b}-{c}", f"{a}.{b}.{c}", f"({a}) {b} {c}", f"{a}-{b}-{c}", f"+1 {a} {b} {c}"])


def addr_fmt(rng, a):
    num, name, full, abbr, city = a
    return f"{num} {name} {rng.choice([full, abbr, abbr + '.'])}, {city}"


def addr_key(a):
    return f"addr:{a[0]} {a[1].lower()} {a[3].lower()}"


def person_name(p, d):
    return p["first"], p["middle"], at(p["lasts"], d)


def legal(e, d):
    if e["kind"] == "company":
        return at(e["names"], d)
    f, m, l = person_name(e, d)
    return " ".join(x for x in (f, m, l) if x)


def co_base(name):
    return " ".join(w for w in name.split() if w.lower().rstrip(".") not in LEGAL_SUFFIX)


def name_variant(rng, e, d, style):
    if e["kind"] == "company":
        nm = at(e["names"], d)
        return {"formal": nm, "crm": rng.choice([nm, co_base(nm), typo(rng, co_base(nm))]),
                "ltr": rng.choice([nm, co_base(nm)])}.get(style, nm)
    f, m, l = person_name(e, d)
    nick = e["nick"] or f
    if style == "formal":
        return f"{l}, {f}" + (f" {m[0]}." if m else "")
    opts = {"crm": [f"{f} {l}", f"{nick} {l}", typo(rng, f"{f} {l}"), f"{l}, {f}"],
            "sig": [f"{f} {l}", f"{nick} {l}", f"{nick} {l}", f"{f[0]}. {l}"],
            "ltr": [f"{f} {l}", f"{f[0]}. {l}", f"{nick} {l}", f"{f} {m} {l}" if m else f"{f} {l}"]}[style]
    return rng.choice(opts)


def employer_at(rng, p, d):
    job = at(p["jobs"], d)
    name = at(job["co"]["names"], d)
    return job, rng.choice([name, co_base(name)])


def plan_records(rng, e):
    # Picks record sources and dates for one entity and fills the clean field values the gold rules read.
    person = e["kind"] == "person"
    eid = e["pid"] if person else e["reg"]
    recs = []
    first_filing = rng.randint(T0 + 20, W0 - 20)
    recs.append({"src": "reg", "date": first_filing, "type": "initial registration" if person else "annual return"})
    names = e["lasts"] if person else e["names"]
    if len(names) > 1:
        recs.append({"src": "reg", "date": names[1][0] + rng.randint(5, 60),
                     "type": f"change of name (formerly {legal(e, names[1][0] - 1)})"})
    if len(e["addrs"]) > 1 and rng.random() < 0.2:
        recs.append({"src": "reg", "date": e["addrs"][-1][0] + rng.randint(5, 60), "type": "change of address"})
    events = [("move", d) for d, _ in e["addrs"][1:]] + [("rename", d) for d, _ in names[1:]] + \
             [("phone", d) for d, _ in e["phones"][1:]]
    rng.shuffle(events)
    k = max(len(recs) + 2, rng.choices([k for k, _ in N_REC_W], [w for _, w in N_REC_W])[0])
    while True:
        srcs, ev = [], list(events)
        for _ in range(k - len(recs)):
            pool = {s: w for s, w in EXTRA_W.items() if s != "evt" or len(ev) > len([x for x in srcs if x == "evt"])}
            srcs.append(rng.choices(list(pool), list(pool.values()))[0])
        if ("crm" in srcs or "sig" in srcs) and 3 <= len(set(srcs) | {"reg"}) <= 5:
            break
    for s in srcs:
        if s == "evt":
            kind, d = ev.pop()
            recs.append({"src": "evt", "date": d, "type": kind})
        else:
            recs.append({"src": s, "date": None})
    used = {(r["src"], r["date"]) for r in recs if r["date"]}
    for r in recs:
        while r["date"] is None:
            d = rng.randint(W0, W1)
            if (r["src"], d) not in used:
                used.add((r["src"], d))
                r["date"] = d
    for r in recs:
        d, f = r["date"], {"id": eid}
        r["f"] = f
        if r["src"] == "reg":
            f.update(name=legal(e, d), addr=at(e["addrs"], d), phone=at(e["phones"], d))
        elif r["src"] == "crm":
            c = r["created"] = max(T0, d - rng.randint(100, 1800))
            f.update(name=name_variant(rng, e, d, "crm"), phone=at(e["phones"], d), addr=at(e["addrs"], c))
            if person:
                job, emp = employer_at(rng, e, c)
                f.update(email=job["email"], employer=emp)
            else:
                f.update(website=at(e["domains"], c))
            if rng.random() > 0.35:
                f.pop("id")
        elif r["src"] == "sig":
            f.pop("id")
            f.update(phone=at(e["phones"], d))
            if person:
                job, emp = employer_at(rng, e, d)
                f.update(name=name_variant(rng, e, d, "sig"), email=job["email"], employer=emp, title=job["title"])
            else:
                f.update(name=legal(e, d), website=at(e["domains"], d), addr=at(e["addrs"], d),
                         email="info@" + at(e["domains"], d))
        elif r["src"] == "ltr":
            f.pop("id")
            f.update(name=name_variant(rng, e, d, "ltr"), addr=at(e["addrs"], d))
            if rng.random() < 0.8:
                f["phone"] = at(e["phones"], d)
            if person and len(e["lasts"]) > 1 and d >= e["lasts"][1][0] and rng.random() < 0.5:
                f["nee"] = e["lasts"][0][1]
        else:
            f["name"] = name_variant(rng, e, d - 1, "formal")
            tl = {"move": e["addrs"], "rename": names, "phone": e["phones"]}[r["type"]]
            f["from"], f["to"] = at(tl, d - 1), at(tl, d)
            if r["type"] == "rename":
                f["from"], f["to"] = legal(e, d - 1), legal(e, d)
                if not person:
                    f["reason"] = rng.choice(["rebrand", "acquisition"])
    bridge(recs)
    return recs


def keys(f, src):
    out = set()
    if "id" in f:
        out.add("id:" + f["id"])
    for k in ("phone", "email"):
        if k in f:
            out.add(f"{k}:{f[k]}")
    if "website" in f:
        out.add("web:" + f["website"])
    if "addr" in f:
        out.add(addr_key(f["addr"]))
    if src == "evt":
        for v in (f["from"], f["to"]):
            if isinstance(v, tuple):
                out.add(addr_key(v))
            elif v.isdigit():
                out.add(f"phone:{v}")
    return out


def bridge(recs):
    # Every entity's records must be linkable through strong keys alone; patch strays with a key from the registry.
    while True:
        comp = {0}
        ks = set(keys(recs[0]["f"], "reg"))
        grew = True
        while grew:
            grew = False
            for i, r in enumerate(recs):
                if i not in comp and keys(r["f"], r["src"]) & ks:
                    comp.add(i)
                    ks |= keys(r["f"], r["src"])
                    grew = True
        stray = [i for i in range(len(recs)) if i not in comp]
        if not stray:
            return
        r = recs[stray[0]]
        if r["src"] == "crm":
            r["f"]["id"] = recs[0]["f"]["id"]
        else:
            r["f"]["phone"] = recs[0]["f"]["phone"]


def gold_profile(e, recs):
    latest = lambda src, key: max((r for r in recs if r["src"] == src and key in r["f"]), key=lambda r: r["date"],
                                  default=None)
    pick = lambda key, srcs: next(r["f"][key] for r in (latest(s, key) for s in srcs) if r)
    reg = latest("reg", "name")
    moves = [r for r in recs if r["src"] == "evt" and r["type"] == "move"]
    city = (max(moves, key=lambda r: r["date"])["f"]["to"] if moves else reg["f"]["addr"])[4]
    prof = {"name": reg["f"]["name"], "phone": pick("phone", ["crm", "sig"]), "city": city}
    if e["kind"] == "person":
        prof.update(born=iso(e["dob"]), email=pick("email", ["sig", "crm"]), employer=pick("employer", ["sig", "crm"]))
    else:
        prof.update(founded=str(e["founded"]), website=pick("website", ["crm", "sig"]))
    return prof


def render(rng, e, r):
    f, d, person = r["f"], r["date"], e["kind"] == "person"
    src = r["src"]
    if src == "reg":
        kind = r["type"]
        head = "Civil Registry" if person else "Companies Registry"
        lines = [f"{head} | filing F/{rng.randint(10000, 99999)} | filed {iso(d)} | {kind}",
                 f"{'Full legal name' if person else 'Registered name'}: {f['name']}",
                 f"{'Personal ID' if person else 'Reg. No.'}: {f['id']}",
                 f"Date of birth: {iso(e['dob'])}" if person else f"Founded: {e['founded']}",
                 f"Registered address: {addr_fmt(rng, f['addr'])}",
                 f"Contact phone: {phone_fmt(rng, f['phone'])}",
                 f"Clerk note: {fill(rng, REG_NOTES, 2)}",
                 f"Filing office: {rng.choice(REG_OFFICES)} registry office | documents: "
                 f"{rng.choice(['passport', 'deed', 'certificate', 'affidavit', 'statutory form'])}"]
        if not person:
            lines.insert(5, f"Directors on file: {rng.randint(1, 7)} | Status: active | Sector: {rng.choice(INDUSTRY)}")
        return "\n".join(lines)
    if src == "crm":
        a = f["addr"]
        cols = [("name", f["name"])]
        if person:
            cols += [("email", f["email"]), ("phone", phone_fmt(rng, f["phone"])),
                     ("street", addr_fmt(rng, a).split(",")[0]),
                     ("city", a[4]), ("company", f["employer"]), ("tax_id", f.get("id", "-"))]
        else:
            cols += [("website", f["website"]), ("phone", phone_fmt(rng, f["phone"])),
                     ("street", addr_fmt(rng, a).split(",")[0]), ("city", a[4]), ("reg_no", f.get("id", "-")),
                     ("industry", rng.choice(INDUSTRY))]
        cols += [("segment", rng.choice(["SMB", "enterprise", "public", "retail"])), ("owner", rng.choice(REPS))]
        notes = fill(rng, CRM_NOTES, rng.randint(5, 8))
        return (f"CRM export | account A-{rng.randint(10000, 99999)} | created {iso(r['created'])} | last_updated "
                f"{iso(d)}\ntype: {'contact' if person else 'account'} | " + " | ".join(f"{k}: {v}" for k, v in cols)
                + f"\nnotes: {notes}")
    if src == "sig":
        to = rng.choice(["Customer Desk", "Accounts Payable", "Order Desk", "Partner Support"])
        body = fill(rng, SIG_LINES, rng.randint(7, 11))
        quote = "\n".join("> " + x for x in fill(rng, SIG_LINES, rng.randint(4, 7)).split(". "))
        if person:
            sig = f"{f['name']}\n{f['title']}, {f['employer']}\nT: {phone_fmt(rng, f['phone'])} | {f['email']}"
            sender = f["email"]
        else:
            sig = (f"{f['name']} · {addr_fmt(rng, f['addr'])} · tel {phone_fmt(rng, f['phone'])} · "
                   f"web: www.{f['website']}")
            sender = f["email"]
        return (f"Email | From: {sender} | Date: {iso(d)} | To: {to} | Subject: Re: {rng.choice(THINGS)}\n\n"
                f"{body}\n\n--\n{sig}\n\n> On {iso(d - rng.randint(1, 20))} {to} wrote:\n{quote}")
    if src == "ltr":
        seg = []  # (text, protected): OCR noise never touches addresses and phone numbers
        dated = dt_long(d)
        if person:
            seg.append((f"{dated}\n\nDear Customer Services,\n\n{fill(rng, LTR_LINES, rng.randint(7, 10))} "
                        "Please note that my address is ", False))
            seg += [(addr_fmt(rng, f["addr"]), True), (". ", False)]
            if "phone" in f:
                seg += [("You can reach me by telephone on ", False), (phone_fmt(rng, f["phone"]), True), (". ", False)]
            seg.append((fill(rng, LTR_LINES, rng.randint(7, 10)) + "\n\nYours sincerely,\n" + f["name"]
                        + (f" (née {f['nee']})" if "nee" in f else ""), False))
        else:
            seg += [(f["name"].upper() + " — ", False), (addr_fmt(rng, f["addr"]), True)]
            if "phone" in f:
                seg += [(" — Tel ", False), (phone_fmt(rng, f["phone"]), True)]
            seg.append((f"\n{dated}\n\nDear Partner,\n\n{fill(rng, CO_LTR_LINES, rng.randint(6, 9))}\n\n"
                        f"{fill(rng, CO_LTR_LINES, rng.randint(6, 9))}\n\n"
                        f"Yours faithfully,\n{rng.choice(REPS)}\nfor {f['name']}", False))
        text = "".join(t if prot else ocr(rng, t) for t, prot in seg)
        return f"Scanned letter (OCR) | received {iso(d + rng.randint(2, 9))}\n{text}"
    kind = {"move": "ADDRESS_CHANGE" if person else "RELOCATION", "rename": "NAME_CHANGE",
            "phone": "PHONE_CHANGE"}[r["type"]]
    show = lambda v: addr_fmt(rng, v) if isinstance(v, tuple) else phone_fmt(rng, v) if v.isdigit() else v
    extra = f" | reason: {f['reason']}" if "reason" in f else " | reason: marriage" if r["type"] == "rename" else ""
    return (f"Event log | {iso(d)} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d} | {kind} | subject: {f['name']} | "
            f"{'person_id' if person else 'reg_no'}: {f['id']} | from: {show(f['from'])} | to: {show(f['to'])}"
            f"{extra} | "
            f"{rng.choice(EVT_NOTES)} | batch {rng.randint(100, 999)} | operator {rng.choice(REPS)} | "
            f"verified: {rng.choice(['yes', 'pending', 'yes (document seen)'])}")


def dt_long(d):
    return dt.date.fromordinal(d).strftime("%d %B %Y")
