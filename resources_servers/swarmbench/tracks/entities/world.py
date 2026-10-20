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

FIRST = {  # first name -> nicknames used in informal sources
    "Katherine": ["Kate", "Kathy"], "Robert": ["Bob", "Rob"], "William": ["Bill", "Will"], "Elizabeth": ["Liz", "Beth"],
    "Margaret": ["Maggie", "Peggy"], "James": ["Jim", "Jamie"], "Richard": ["Rick", "Dick"], "Jennifer": ["Jen", "Jenny"],
    "Michael": ["Mike"], "Patricia": ["Pat", "Trish"], "Thomas": ["Tom"], "Christopher": ["Chris"], "Daniel": ["Dan"],
    "Alexander": ["Alex", "Sasha"], "Victoria": ["Vicky", "Tori"], "Samuel": ["Sam"], "Rebecca": ["Becky"],
    "Jonathan": ["Jon"], "Nicholas": ["Nick"], "Benjamin": ["Ben"], "Theodore": ["Ted", "Theo"], "Susan": ["Sue"],
    "Anthony": ["Tony"], "Deborah": ["Deb"], "Edward": ["Ed", "Ned"], "Frances": ["Fran"], "Gregory": ["Greg"],
    "Josephine": ["Jo"], "Matthew": ["Matt"], "Natalie": ["Nat"], "Abigail": ["Abby"], "Joseph": ["Joe"],
    "Andrea": [], "Mara": [], "Ingrid": [], "Tomas": [], "Leon": [], "Priya": [], "Yusuf": [], "Elena": [],
    "Oskar": [], "Hana": [], "Rafael": [], "Ines": [], "Dmitri": [], "Amara": [], "Lucia": [], "Kenji": [],
}
FIRSTS = sorted(FIRST)
MIDDLE = ["Ann", "Marie", "Lee", "James", "Rose", "John", "Grace", "Paul", "Jane", "Ray", "Louise", "Edward", "May"]
ONSET = ["b", "br", "c", "d", "dr", "f", "g", "gr", "h", "k", "l", "m", "n", "p", "r", "s", "st", "t", "tr", "v", "w"]
VOWEL = ["a", "e", "i", "o", "u", "ai", "ea", "ou"]
CODA = ["n", "r", "l", "s", "th", "rk", "nd", "x", "m", "ck", "ll", "rn", "ss"]
LAST_END = ["", "er", "on", "ley", "ford", "ton", "berg", "wick", "ard", "en", "is", "ova"]
CO_END = ["Analytics", "Systems", "Logistics", "Foods", "Partners", "Labs", "Holdings", "Robotics", "Media",
          "Freight", "Energy", "Textiles", "Pharma", "Capital", "Works", "Studios", "Dynamics", "Instruments"]
CO_SUFFIX = ["Inc.", "LLC", "Ltd.", "Corp.", "GmbH", "S.A."]
STREET_TYPES = [("Street", "St"), ("Road", "Rd"), ("Avenue", "Ave"), ("Lane", "Ln"), ("Boulevard", "Blvd"),
                ("Drive", "Dr"), ("Court", "Ct"), ("Place", "Pl")]
CITY_PRE = ["", "", "", "Port ", "New ", "East ", "Lake "]
TITLES = ["Account Manager", "Senior Analyst", "Operations Lead", "Procurement Officer", "Engineer",
          "Project Coordinator", "Finance Director", "Sales Executive", "Legal Counsel", "Product Manager"]
INDUSTRY = ["manufacturing", "retail", "healthcare", "transport", "software", "agriculture", "media", "finance"]

T0 = dt.date(2010, 1, 1).toordinal()
W0 = dt.date(2014, 1, 1).toordinal()  # records are dated inside [W0, W1]
W1 = dt.date(2025, 12, 31).toordinal()


def iso(d):
    return dt.date.fromordinal(d).isoformat()


def word(rng, end_pool):
    w = rng.choice(ONSET) + rng.choice(VOWEL) + rng.choice(CODA)
    if rng.random() < 0.5:
        w += rng.choice(VOWEL) + rng.choice(CODA)
    return (w + rng.choice(end_pool)).capitalize()


class Uniq:
    # Hands out values that never repeat anywhere in the world; strong keys rely on this.
    def __init__(self, rng):
        self.rng, self.seen = rng, set()

    def take(self, make):
        while True:
            v = make(self.rng)
            if v not in self.seen:
                self.seen.add(v)
                return v


def at(timeline, d):
    # timeline: list of (start_ordinal, value) sorted by start; value in force at d.
    v = timeline[0][1]
    for start, val in timeline:
        if start <= d:
            v = val
    return v


def change_dates(rng, k):
    return sorted(rng.sample(range(W0 + 200, W1 - 200), k))


def build_entities(n, rng):
    uniq = Uniq(rng)
    n_city = 12 + n // 24
    cities = sorted({rng.choice(CITY_PRE) + word(rng, ["", "a", "ton", "ford", "by"]) for _ in range(n_city * 2)})
    cities = rng.sample(cities, min(n_city, len(cities)))
    street_names = sorted({word(rng, ["", "wood", "field", "hill", "brook"]) for _ in range(400)})
    surnames = sorted({word(rng, LAST_END) for _ in range(max(120, n // 4))})
    rng.shuffle(surnames)
    sur_w = [1 / (i + 5) for i in range(len(surnames))]  # skewed like real surname frequencies
    surname = lambda: rng.choices(surnames, sur_w)[0]

    def addr():
        # number + street + type is the strong key, so it is unique on its own; the city is free.
        _, num, name, (full, abbr) = uniq.take(lambda r: ("addr", r.randint(1, 9999), r.choice(street_names),
                                                          r.choice(STREET_TYPES)))
        return (num, name, full, abbr, rng.choice(cities))

    phone = lambda: uniq.take(lambda r: f"{r.randint(201, 989)}{r.randint(200, 999)}{r.randint(0, 9999):04d}")

    n_co = max(8, round(n * 0.35))
    ents = []
    for i in range(n_co):
        base = uniq.take(lambda r: word(r, [""]) + " " + r.choice(CO_END))
        names = [(T0, f"{base} {rng.choice(CO_SUFFIX)}")]
        if rng.random() < 0.3:  # rebrand or acquisition: new legal name and new domain
            names.append((change_dates(rng, 1)[0], uniq.take(lambda r: word(r, [""]) + " " + r.choice(CO_END))
                          + " " + rng.choice(CO_SUFFIX)))
        domains = [(d, uniq.take(lambda r, nm=nm: domain_of(nm, r))) for d, nm in names]
        ents.append({"kind": "company", "names": names, "domains": domains,
                     "reg": uniq.take(lambda r: f"RC-{r.randint(100000, 999999)}"),
                     "founded": rng.randint(1955, 2012), **moves_and_phones(rng, addr, phone)})
    # Shared names: some companies copy another's legal name (registered elsewhere, different keys).
    for e in rng.sample(ents, max(1, n_co // 10)):
        other = rng.choice(ents)
        if other is not e and len(other["names"]) == 1:
            e["names"] = [(T0, other["names"][0][1])] + e["names"][1:]

    companies = list(ents)
    for i in range(n - n_co):
        first = rng.choice(FIRSTS)
        p = {"kind": "person", "first": first, "middle": rng.choice(MIDDLE) if rng.random() < 0.7 else "",
             "lasts": [(T0, surname())], "pid": uniq.take(
                 lambda r: f"{r.randint(100, 899)}-{r.randint(10, 99)}-{r.randint(1000, 9999)}"),
             "dob": rng.randint(dt.date(1950, 1, 1).toordinal(), dt.date(2000, 12, 31).toordinal()),
             "nick": rng.choice(FIRST[first]) if FIRST[first] and rng.random() < 0.6 else "",
             **moves_and_phones(rng, addr, phone)}
        if rng.random() < 0.25:
            p["lasts"].append((change_dates(rng, 1)[0], surname()))
        ents.append(p)
    persons = [e for e in ents if e["kind"] == "person"]
    for e in rng.sample(persons, max(1, len(persons) // 7)):  # exact namesakes
        other = rng.choice(persons)
        if other is not e:
            e["first"], e["middle"] = other["first"], other["middle"]
            e["lasts"] = [(T0, other["lasts"][0][1])] + e["lasts"][1:]
            e["nick"] = other["nick"]
    for p in persons:
        starts = [T0] + change_dates(rng, rng.choice([0, 1, 1, 2]))
        p["jobs"] = []
        for s in starts:
            co = rng.choice(companies)
            last = at(p["lasts"], s).lower()
            local = rng.choice([f"{p['first'].lower()}.{last}", f"{p['first'][0].lower()}{last}",
                                f"{(p['nick'] or p['first']).lower()}.{last}"])
            dom = at(co["domains"], s)
            email = uniq.take(lambda r: local + (str(r.randint(2, 99)) if r.random() < 0.15 else "") + "@" + dom)
            p["jobs"].append((s, {"co": co, "title": rng.choice(TITLES), "email": email}))
    rng.shuffle(ents)
    return ents, cities


def domain_of(name, rng):
    stem = "".join(c for c in name.lower() if c.isalpha() or c == " ").split()
    stem = stem[0] + (stem[1] if len(stem) > 1 and rng.random() < 0.5 else "")
    return stem + rng.choice(["", "", "-group", "hq", str(rng.randint(1, 99))]) + rng.choice([".com", ".net", ".io",
                                                                                               ".co"])


def moves_and_phones(rng, addr, phone):
    addrs = [(T0, addr())] + [(d, addr()) for d in change_dates(rng, rng.choice([0, 1, 1, 2]))]
    phones = [(T0, phone())] + [(d, phone()) for d in change_dates(rng, rng.choice([0, 0, 1]))]
    return {"addrs": addrs, "phones": phones}
