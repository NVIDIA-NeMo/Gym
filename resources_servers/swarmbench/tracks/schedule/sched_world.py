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
import math
import random

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday"]
ABBR = ["Mon", "Tue", "Wed", "Thu"]
HOURS = list(range(9, 17))  # talk start hours; every talk fills one 1-hour slot
NH = len(HOURS)
NS = len(DAYS) * NH
EQUIP = ["projector", "livestream", "recording", "microphone array", "whiteboard", "hybrid conferencing"]
TIERS = [40, 80, 150, 300, 600]
ITEMS_PER_ROOM = 27  # ~85% of a room's open slots end up used by the hidden plan


def slot_name(s):
    return f"{ABBR[s // NH]} {HOURS[s % NH]:02d}:00"


def day_of(s):
    return s // NH


def hour_of(s):
    return HOURS[s % NH]


def allowed_slots(av):
    out = []
    for d in av["days"]:
        for h in range(av["start"], av["end"]):
            if (d, h) in av["except"]:
                continue
            out.append(d * NH + h - HOURS[0])
    return sorted(out)


def pref_ok(p, s, r, slot_of):
    t, a = p["type"], p.get("arg")
    if t == "morning":
        return hour_of(s) < 12
    if t == "afternoon":
        return hour_of(s) >= 13
    if t == "day":
        return day_of(s) == a
    if t == "room":
        return r == a
    if t == "same_day":
        return a in slot_of and day_of(slot_of[a]) == day_of(s)
    raise ValueError(t)


def _days_text(days):
    names = [DAYS[d] for d in days]
    if len(names) == 1:
        return names[0]
    if len(names) > 2 and days == list(range(days[0], days[-1] + 1)):
        return f"{names[0]} through {names[-1]}"
    return ", ".join(names[:-1]) + " or " + names[-1]


# ---------- world sampling ----------

FIRST = ["Ada", "Bram", "Chiara", "Dmitri", "Elif", "Farid", "Greta", "Hiro", "Ines", "Jonas", "Kavya", "Lior", "Mei",
         "Nadia", "Oskar", "Priya", "Quentin", "Rosa", "Sven", "Tamar", "Ugo", "Vera", "Wen", "Ximena", "Yusuf", "Zora",
         "Anil", "Beatriz", "Cyril", "Dana", "Emeka", "Fiona", "Goran", "Hana", "Ivo", "Joelle", "Kenji", "Lena",
         "Mateo", "Noor", "Olga", "Pavel", "Rania", "Stellan", "Thea", "Umar", "Viktor", "Wanjiru", "Yara", "Zeno",
         "Amara", "Bodhi", "Camille", "Darius", "Esme", "Felix", "Gaia", "Henrik", "Ilse", "Jun"]
LAST = ["Abara", "Bergstrom", "Castellano", "Dubois", "Eriksen", "Fontaine", "Gallo", "Haddad", "Ibsen", "Jaramillo",
        "Kowalczyk", "Lindqvist", "Moreau", "Nakamura", "Okafor", "Petrov", "Quaresma", "Rahman", "Sandoval",
        "Takahashi", "Underwood", "Varga", "Whitlock", "Xu", "Yilmaz", "Zielinski", "Achebe", "Brandt", "Cordero",
        "Delacroix", "Esposito", "Fischer", "Gutierrez", "Holm", "Iqbal", "Jensen", "Kaur", "Laine", "Mbeki", "Novak",
        "Oyelaran", "Pereira", "Quist", "Rossi", "Sato", "Tanaka", "Ulloa", "Vasquez", "Weber", "Yamada", "Zhou",
        "Adeyemi", "Barros", "Chen", "Dahl", "Engel", "Ferreira", "Grant", "Hughes", "Ivanova", "Kim", "Lund", "Maes",
        "Nilsen", "Ortiz", "Park", "Rivera", "Schmidt", "Toivonen", "Vogel"]
AFFIL = ["Northgate University", "Institute for Applied Systems", "Kestrel Labs", "Polytechnic of Varna",
         "Halden Research", "Meridian College", "Oakridge Data Group", "Lakeshore Institute of Technology",
         "Brightwater Analytics", "University of Tarsen", "Corvid Computing", "Elmstead Foundation"]
ADJ = ["Scalable", "Robust", "Sparse", "Adaptive", "Federated", "Incremental", "Probabilistic", "Verified",
       "Low-Latency", "Energy-Aware", "Explainable", "Distributed", "Causal", "Streaming", "Compositional",
       "Self-Healing", "Privacy-Preserving", "Hierarchical", "Lightweight", "Interactive", "Asynchronous",
       "Resilient", "Neural", "Symbolic", "Approximate", "Declarative", "Elastic", "Secure", "Multimodal", "Online"]
NOUN = ["Scheduling", "Retrieval", "Indexing", "Compilation", "Caching", "Replication", "Planning", "Inference",
        "Monitoring", "Partitioning", "Query Optimization", "Program Repair", "Load Balancing", "Consensus",
        "Data Cleaning", "Model Serving", "Graph Mining", "Benchmarking", "Testing", "Simulation", "Forecasting",
        "Routing", "Compression", "Search", "Verification", "Calibration", "Annotation", "Sampling", "Deduplication",
        "Provisioning"]
DOMAIN = ["Edge Devices", "Genomics Pipelines", "Satellite Imagery", "Clinical Records", "Power Grids",
          "Supply Chains", "Legal Archives", "Ocean Sensors", "Retail Logs", "Rail Networks", "Code Repositories",
          "Weather Models", "Financial Ledgers", "Smart Buildings", "Language Corpora", "Video Streams",
          "Autonomous Fleets", "Public Transit", "Crop Monitoring", "Cloud Billing", "Museum Collections",
          "Earthquake Catalogs", "Hospital Staffing", "Air Traffic", "Water Utilities", "Census Data",
          "Wildlife Tracking", "Telecom Traces", "Battery Factories", "Library Catalogs"]
ROOM_NAMES = ["Aurora", "Birch", "Cedar", "Delta", "Ember", "Fjord", "Granite", "Harbor", "Iris", "Juniper", "Kelp",
              "Lumen", "Maple", "Nimbus", "Onyx", "Pine", "Quartz", "Raven", "Summit", "Tundra", "Umber", "Vista",
              "Willow", "Zephyr"]
ROOM_KINDS = ["Hall", "Room", "Studio", "Lounge", "Theatre", "Suite"]


def sample(n, seed):
    rng = random.Random(f"schedule:{seed}:{n}")
    rooms = []

    def open_cells():
        return [(s, r) for r in range(len(rooms)) for s in range(NS) if s not in rooms[r]["closed"]]

    target = max(3, math.ceil(n / ITEMS_PER_ROOM))
    while len(rooms) < target or len(open_cells()) * 0.9 < n:
        r = len(rooms)
        cap = rng.choices(TIERS, weights=[3, 4, 3, 2, 1])[0]
        closed, blocks = set(), []
        for _ in range(rng.choice([0, 1, 1, 2])):
            d = rng.randrange(len(DAYS))
            if rng.random() < 0.15:
                a, b = HOURS[0], HOURS[-1] + 1
            else:
                a = rng.choice(HOURS[:-1])
                b = min(HOURS[-1] + 1, a + rng.randint(1, 3))
            blocks.append((d, a, b))
            closed |= {d * NH + h - HOURS[0] for h in range(a, b)}
        name = f"{ROOM_NAMES[r % len(ROOM_NAMES)]} {ROOM_KINDS[(r // len(ROOM_NAMES)) % len(ROOM_KINDS)]}"
        if r >= len(ROOM_NAMES) * len(ROOM_KINDS):
            name += f" {r // (len(ROOM_NAMES) * len(ROOM_KINDS)) + 1}"
        rooms.append({"id": f"R{r + 1:03d}", "name": name, "cap": cap,
                      "equip": sorted(rng.sample(EQUIP, rng.randint(1, 4)), key=EQUIP.index),
                      "closed": closed, "blocks": blocks, "floor": rng.randint(0, 4)})
    cells = open_cells()
    rng.shuffle(cells)
    cells = cells[:n]

    titles, items = set(), []
    while len(titles) < n:
        titles.add(f"{rng.choice(ADJ)} {rng.choice(NOUN)} for {rng.choice(DOMAIN)}")
    titles = sorted(titles)
    rng.shuffle(titles)
    for i, (s, r) in enumerate(cells):
        room = rooms[r]
        lower = ([10] + TIERS)[TIERS.index(room["cap"])]
        att = rng.randint(lower + 1, room["cap"]) if rng.random() < 0.6 else rng.randint(10, lower)
        att = max(10, min(room["cap"], 5 * round(att / 5)))
        equip = sorted(rng.sample(room["equip"], rng.randint(1, min(2, len(room["equip"])))), key=EQUIP.index) \
            if rng.random() < 0.6 else []
        items.append({"id": f"T{i:04d}", "title": titles[i], "slot": s, "room": room["id"], "min_cap": att,
                      "equip": equip, "after": [], "not_with": [], "not_with_speaker": [], "prefs": []})
        items[-1]["avail"] = _availability(rng, s)
        items[-1]["allowed"] = allowed_slots(items[-1]["avail"])
        assert s in items[-1]["allowed"]

    names = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], len(items))
    speakers, order = {}, list(range(n))
    rng.shuffle(order)
    for k, i in enumerate(order):
        sp = None
        if speakers and rng.random() < 0.45:
            for _ in range(4):
                cand = rng.choice(list(speakers))
                if len(speakers[cand]) < 3 and all(items[j]["slot"] != items[i]["slot"] for j in speakers[cand]):
                    sp = cand
                    break
        sp = sp or f"Dr. {names[k]}"
        speakers.setdefault(sp, []).append(i)
        items[i]["speaker"] = sp
        items[i]["affil"] = rng.choice(AFFIL)

    by_slot = [[] for _ in range(NS)]
    for i, it in enumerate(items):
        by_slot[it["slot"]].append(i)
    for i, it in enumerate(items):
        s = it["slot"]
        if rng.random() < 0.35 and s > 0:
            for _ in range(rng.choice([1, 1, 1, 2])):
                for _ in range(10):
                    lst = by_slot[rng.randint(max(0, s - 10), s - 1)]
                    if lst:
                        j = rng.choice(lst)
                        if items[j]["id"] not in it["after"]:
                            it["after"].append(items[j]["id"])
                        break
        # Conflict partners are drawn from the talk's own window, so the conflict actually removes options.
        others = [t for t in it["allowed"] if t != s and by_slot[t]]
        if others and rng.random() < 0.35:
            j = rng.choice(by_slot[rng.choice(others)])
            if items[j]["speaker"] != it["speaker"]:
                it["not_with"].append(items[j]["id"])
        if others and rng.random() < 0.12:
            sp = items[rng.choice(by_slot[rng.choice(others)])]["speaker"]
            if sp != it["speaker"] and all(items[j]["slot"] != s for j in speakers[sp]):
                it["not_with_speaker"].append(sp)
        it["prefs"] = _prefs(rng, it, items, rooms)
    return {"rooms": rooms, "items": items, "speakers": speakers}


def _availability(rng, s):
    d, h = day_of(s), hour_of(s)
    k = rng.choices([1, 2, 3, 4], weights=[3, 4, 2, 1])[0]
    days = sorted({d} | set(rng.sample(range(len(DAYS)), k - 1)))
    if len(days) < k:
        days = sorted(set(days) | set(rng.sample([x for x in range(len(DAYS)) if x not in days], k - len(days))))
    width = rng.randint(2, 6) if rng.random() < 0.9 else NH
    start = rng.randint(max(HOURS[0], h - width + 1), min(h, HOURS[-1] + 1 - width))
    av = {"days": days, "start": start, "end": start + width, "except": set(), "except_text": None,
          "update": None}
    others = [x for x in days if x != d]
    if others and rng.random() < 0.25 and width >= 3:
        x = rng.choice(others)
        if rng.random() < 0.5:
            cut = rng.randint(start + 1, start + width - 1)  # must be over by `cut` on day x
            av["except"] = {(x, hh) for hh in range(cut, start + width)}
            av["except_text"] = f"On {DAYS[x]} the talk has to be over by {cut:02d}:00."
        else:
            arr = rng.randint(start + 1, start + width - 1)  # arrives at `arr` on day x
            av["except"] = {(x, hh) for hh in range(start, arr)}
            av["except_text"] = f"On {DAYS[x]} the speaker arrives late, so no start before {arr:02d}:00 that day."
    if rng.random() < 0.2:
        added = [x for x in others if not any(e[0] == x for e in av["except"])]
        if added and rng.random() < 0.5:
            x = rng.choice(added)  # never the exception's day: "same hours as above" must stay unambiguous
            av["update"] = ("add", x)  # listed late in the thread; the first message omits it
        else:
            missing = [x for x in range(len(DAYS)) if x not in days]
            if missing:
                av["update"] = ("drop", rng.choice(missing))  # first message lists it, the update withdraws it
    return av


def _prefs(rng, it, items, rooms):
    s = it["slot"]
    out = []
    for _ in range(rng.choice([0, 1, 1, 2])):
        keep = rng.random() < 0.6
        t = rng.choice(["morning", "afternoon", "day", "room", "same_day"])
        w = rng.randint(1, 5)
        if t in ("morning", "afternoon"):
            ok = pref_ok({"type": t}, s, None, {})
            if ok != keep:
                continue
            out.append({"type": t, "w": w})
        elif t == "day":
            days = sorted({day_of(x) for x in it["allowed"]})
            cand = [day_of(s)] if keep else [x for x in days if x != day_of(s)]
            if cand:
                out.append({"type": "day", "arg": rng.choice(cand), "w": w})
        elif t == "room":
            elig = [r["id"] for r in rooms if r["cap"] >= it["min_cap"] and set(it["equip"]) <= set(r["equip"])]
            cand = [it["room"]] if keep else [x for x in elig if x != it["room"]]
            if cand:
                out.append({"type": "room", "arg": rng.choice(cand), "w": w})
        else:
            for _ in range(5):
                j = rng.randrange(len(items))
                if j != int(it["id"][1:]) and (day_of(items[j]["slot"]) == day_of(s)) == keep:
                    out.append({"type": "same_day", "arg": items[j]["id"], "w": w})
                    break
    return out


# ---------- text rendering ----------

ABSTRACT = [
    "This talk presents {adj_l} techniques for {noun_l} in settings dominated by {dom_l}.",
    "We report on three years of deploying {noun_l} systems for {dom_l}, including the failures.",
    "Existing approaches to {noun_l} break down when the workload looks like {dom_l}; we show why.",
    "The core idea is to treat {noun_l} as a {adj_l} process rather than a one-off batch job.",
    "We evaluate on public and proprietary traces and report both median and tail behaviour.",
    "Results show a {pct}% reduction in operator effort with no loss in accuracy.",
    "The talk closes with open problems and a call for shared benchmarks in {dom_l}.",
    "Attendees will leave with a checklist for deciding when {adj_l} {noun_l} is worth the complexity.",
    "We also discuss what did not work, including two designs we abandoned after pilot studies.",
    "A live demonstration shows the system recovering from injected faults.",
    "The work is joint with practitioners who operate {dom_l} at national scale.",
    "We compare against strong baselines and release the evaluation harness.",
    "Particular attention is paid to reproducibility and to the cost of running the pipeline.",
    "Our measurements suggest that most of the gains come from better data handling, not from modelling.",
]
BIO = [
    "{name} leads the {noun_l} group at {affil}.",
    "Before that, {last} spent several years in industry building infrastructure for {dom_l}.",
    "{last} has served on the program committees of several workshops and edits a newsletter on {noun_l}.",
    "Outside work, {last} coaches a youth chess club and restores old bicycles.",
    "{last}'s recent work received a best paper award at a regional symposium.",
    "{last} holds a doctorate from {affil2} and has supervised a dozen graduate students.",
]
REVIEW = [
    "Reviewer {k}: The problem is real and the {dom_l} setting is underserved at this venue; I would attend this talk.",
    "Reviewer {k}: The evaluation is convincing, although the comparison with classical {noun_l} methods is thin.",
    "Reviewer {k}: I worry the {adj_l} framing oversells the contribution; the engineering is the interesting part.",
    "Reviewer {k}: Clear abstract. Please make sure the talk explains the failure cases, not only the headline numbers.",
    "Reviewer {k}: The speaker has a strong track record and the topic fits the practitioner audience well.",
    "Reviewer {k}: The claimed {pct}% improvement needs context; what was the baseline effort before the change?",
    "Reviewer {k}: Strong accept. This is exactly the kind of experience report the program has been missing.",
    "Reviewer {k}: Weak accept. The material overlaps with last year's tutorial on {noun_l}, but the {dom_l} angle is new.",
    "Reviewer {k}: The demo is ambitious for a one-hour slot; a recorded fallback would be wise.",
    "Reviewer {k}: Relevant to several tracks; expect questions from both the systems and the data people.",
]
NOISE = [
    "Slides will be shared on the conference site after the session.",
    "The speaker asked for a glass of water at the lectern.",
    "Catering note: the speaker is vegetarian.",
    "The session chair will be assigned by the track committee later.",
    "A short demo video will be played; audio output is fine through the house system.",
    "Please print two copies of the abstract for the registration desk.",
    "The speaker will bring printed handouts for the audience.",
    "Question time will be moderated from the audience microphone.",
    "Travel reimbursement was approved by the finance office.",
    "The speaker prefers to be introduced by first name.",
    "The talk was accepted with minor revisions; the final abstract is above.",
    "Badge pickup for the speaker is arranged at the north entrance.",
]
DISTRACT = [
    "The co-author, {other}, cannot attend on {day}, but {last} will present alone, so this does not affect scheduling.",
    "{last} will also attend the reception on {day} evening; that has no bearing on the talk slot.",
    "Last year {last} spoke on {day} at {hour}:00; that is irrelevant for this year's program.",
    "An earlier hotel booking mentioned {day}; the travel office confirmed it has nothing to do with the talk time.",
]
PREF_TEXT = {
    "morning": "Soft preference (weight {w}): {last} would rather present in the morning (start before 12:00).",
    "afternoon": "Soft preference (weight {w}): an afternoon slot (start at 13:00 or later) would suit {last} best.",
    "day": "Soft preference (weight {w}): if possible, put the talk on {day}.",
    "room": "Soft preference (weight {w}): {last} has presented in {room} before and would like it again ({rid}).",
    "same_day": "Soft preference (weight {w}): ideally on the same day as \"{title}\", which shares much of the audience.",
}


def _fill(rng, pool, k, **kw):
    return " ".join(x.format(**kw) for x in rng.sample(pool, k))


def render_item(it, items_by_id, rooms_by_id, seed):
    rng = random.Random(f"schedule-text:{seed}:{it['id']}")
    first, last = it["speaker"][4:].split(" ", 1)
    words = it["title"].split(" for ")
    kw = {"adj_l": words[0].split(" ", 1)[0].lower(), "noun_l": words[0].split(" ", 1)[1].lower(),
          "dom_l": words[1].lower(), "pct": rng.randint(11, 63), "name": it["speaker"], "last": f"Dr. {last}",
          "affil": it["affil"], "affil2": rng.choice(AFFIL)}
    av = it["avail"]
    hard = []
    days = list(av["days"])
    if av["update"] and av["update"][0] == "add":
        days.remove(av["update"][1])
    elif av["update"]:
        days = sorted(days + [av["update"][1]])
    if len(days) == len(DAYS):
        hard.append("Any of the four days works for this talk.")
    else:
        hard.append(rng.choice(["This talk can only go on {days}.", "Available on {days} only.",
                                "For this talk, {last} is available on {days} and not on other days."])
                    .format(days=_days_text(days), **kw))
    if av["end"] - av["start"] == NH:
        hard.append("Any time of day works.")
    else:
        hard.append(rng.choice([
            "The talk must start no earlier than {a:02d}:00 and be over by {b:02d}:00.",
            "Acceptable start times are {a:02d}:00 through {b1:02d}:00.",
            "Time of day: not before {a:02d}:00, and {last} has to leave by {b:02d}:00.",
        ]).format(a=av["start"], b=av["end"], b1=av["end"] - 1, **kw))
    if av["except_text"]:
        hard.append(av["except_text"])
    late = []
    if av["update"] and av["update"][0] == "add":
        late.append(f"Update from {kw['last']}: {DAYS[av['update'][1]]} now works as well, same hours as above.")
    elif av["update"]:
        late.append(f"Update from {kw['last']}: {DAYS[av['update'][1]]} is no longer possible after a travel change.")
    need = [f"Expected audience: about {it['min_cap']} people, so the room must seat at least {it['min_cap']}."]
    if it["equip"]:
        need.append(rng.choice(["The talk needs {e}.", "Required equipment: {e}.",
                                "{last} cannot present without {e}."]).format(e=" and ".join(it["equip"]), **kw))
    else:
        need.append("No special equipment is needed.")
    for a in it["after"]:
        o = items_by_id[a]
        hard.append(rng.choice([
            "This talk builds on \"{t}\" ({sp}), so it must be scheduled at a later time than that talk.",
            "Program committee: must come after \"{t}\", which introduces the material.",
            "Ordering constraint: schedule strictly after \"{t}\" by {sp}.",
        ]).format(t=o["title"], sp=o["speaker"]))
    for a in it["not_with"]:
        o = items_by_id[a]
        hard.append(rng.choice([
            "Must not run at the same time as \"{t}\" (same target audience).",
            "Clash rule: not in parallel with \"{t}\" by {sp}.",
        ]).format(t=o["title"], sp=o["speaker"]))
    for sp in it["not_with_speaker"]:
        hard.append(f"{kw['last']} wants to attend every talk by {sp}, so this talk must not overlap any of them.")
    soft = []
    for p in it["prefs"]:
        arg = p.get("arg")
        soft.append(PREF_TEXT[p["type"]].format(
            w=p["w"], day=DAYS[arg] if p["type"] == "day" else "",
            room=rooms_by_id[arg]["name"] if p["type"] == "room" else "", rid=arg,
            title=items_by_id[arg]["title"] if p["type"] == "same_day" else "", **kw))
    noise = rng.sample(NOISE, 5) + [rng.choice(DISTRACT).format(
        other=f"Dr. {rng.choice(FIRST)} {rng.choice(LAST)}", day=rng.choice(DAYS), hour=rng.choice(HOURS), **kw)]
    notes = hard + noise + soft
    rng.shuffle(notes)
    text = (f"# {it['id']}: {it['title']}\nSpeaker: {it['speaker']} ({it['affil']})\n"
            "Format: talk, one 1-hour program slot\n\n## Abstract\n"
            + _fill(rng, ABSTRACT, 6, **kw) + "\n\n" + _fill(rng, ABSTRACT, 5, **kw)
            + "\n\n" + _fill(rng, ABSTRACT, 5, **kw) + "\n\n" + _fill(rng, ABSTRACT, 5, **kw)
            + "\n\n## About the speaker\n" + _fill(rng, BIO, 5, **kw)
            + "\n\n## Committee reviews\n" + "\n".join(
                x.format(k=k + 1, **kw) for k, x in enumerate(rng.sample(REVIEW, 7)))
            + "\n\n## Room needs\n" + " ".join(need)
            + "\n\n## Scheduling notes (program committee thread, oldest first)\n"
            + "\n".join(f"- {x}" for x in notes + late) + "\n")
    return text


def render_room(room, seed):
    rng = random.Random(f"schedule-room:{seed}:{room['id']}")
    closed = []
    for d, a, b in room["blocks"]:
        if b - a == NH:
            closed.append(f"closed all day on {DAYS[d]}")
        else:
            closed.append(f"closed on {DAYS[d]} from {a:02d}:00 to {b:02d}:00 "
                          f"({rng.choice(['maintenance', 'a sponsor event', 'cleaning', 'AV rewiring'])})")
    cal = ("Open 09:00-17:00 on all four days." if not closed else
           "Open 09:00-17:00 on all four days, except: " + "; ".join(closed) + ".")
    return (f"# {room['id']}: {room['name']}\nFloor {room['floor']}, "
            f"{rng.choice(['east', 'west', 'north', 'south'])} wing.\n\n"
            f"Seats: {room['cap']} (fire code limit; a talk expecting more people cannot use this room).\n"
            f"Equipment: {', '.join(room['equip'])}. Nothing else is available here and equipment cannot be moved "
            "between rooms.\n"
            f"Calendar: {cal}\n\n"
            f"Notes: step-free access via the {rng.choice(['main lift', 'ramp by the atrium', 'service lift'])}. "
            f"{rng.choice(['Natural light; blinds available.', 'No windows.', 'Air conditioning can be loud.'])} "
            "Room changeover between talks is handled by venue staff.\n")
