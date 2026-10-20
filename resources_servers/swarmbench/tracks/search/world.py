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
import random

ONSETS = ["b", "br", "c", "d", "dr", "f", "g", "gr", "h", "k", "kr", "l", "m", "n", "p", "pr", "r", "s", "st", "t", "tr", "th", "v", "z"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ei", "ou", "ia"]
CODAS = ["", "", "n", "r", "l", "s", "th", "nd", "rk", "m", "x", "sk"]
OCCUPATIONS = ["painter", "engineer", "economist", "chemist", "novelist", "architect", "physician", "composer",
               "diplomat", "botanist", "sculptor", "mathematician", "historian", "photographer", "jurist"]
INDUSTRIES = ["shipbuilding", "textiles", "software", "mining", "pharmaceuticals", "publishing", "rail transport",
              "aerospace", "agriculture", "banking", "optics", "ceramics", "brewing", "telecommunications"]
CO_SUFFIXES = ["Works", "Systems", "Holdings", "Industries", "Group", "Labs", "Trading Company", "Foundry", "Mills",
               "Partners", "Instruments", "Shipping"]
EVENT_KINDS = ["Games", "Summit", "Expo", "Regatta", "Film Festival", "Chess Olympiad", "Music Festival", "Trade Fair",
               "Marathon", "Science Congress"]
CITY_PREFIXES = ["Port", "New", "Upper", "Lower", "Saint", "Lake", "North", "East"]
TOPICS = ["trade", "the arts", "local politics", "rail infrastructure", "public health", "education", "printing",
          "the harbor economy", "agricultural reform", "museum collections", "civic architecture"]


class Namer:
    def __init__(self, rng):
        self.rng, self.used = rng, set()

    def word(self, lo=2, hi=3):
        while True:
            s = "".join(self.rng.choice(ONSETS) + self.rng.choice(VOWELS) + self.rng.choice(CODAS)
                        for _ in range(self.rng.randint(lo, hi)))
            w = s.capitalize()
            if w not in self.used and 4 <= len(w) <= 12:
                self.used.add(w)
                return w


def build_world(n, seed):
    rng = random.Random(seed)
    namer = Namer(rng)
    total = 20000 + 20 * n
    n_city, n_event = int(total * 0.08), int(total * 0.15)
    n_co = int(total * 0.22)
    n_person = total - n_city - n_event - n_co - 40
    titles = set()

    countries = [namer.word(2, 3) + rng.choice(["ia", "land", "stan", "a", ""]) for _ in range(40)]
    titles.update(countries)

    cities = []
    roots = []
    while len(cities) < n_city:
        # Variants of an existing root ("Port Valen" next to "Valen") are the city near-duplicates.
        if roots and rng.random() < 0.3:
            name = rng.choice(CITY_PREFIXES) + " " + rng.choice(roots)
        else:
            name = namer.word()
            roots.append(name)
        if name in titles:
            continue
        titles.add(name)
        cities.append({"kind": "city", "title": name, "country": rng.choice(countries),
                       "founded": rng.randint(800, 1850), "population": rng.randint(4, 2000) * 1000})
    for c in cities:
        if rng.random() < 0.2:
            alias = namer.word()
            titles.add(alias)
            c["alias"] = alias

    firsts = [namer.word(2, 2) for _ in range(250)]
    lasts = [namer.word(2, 3) for _ in range(max(1000, n_person // 3))]
    people = []
    while len(people) < n_person:
        name = rng.choice(firsts) + " " + rng.choice(lasts)
        title = f"{name} ({rng.choice(OCCUPATIONS)})"
        if title in titles:
            continue
        titles.add(title)
        bc = rng.randrange(n_city)
        people.append({"kind": "person", "title": title, "name": name, "occupation": title[len(name) + 2:-1],
                       "birth_city": None if rng.random() < 0.15 else bc,
                       "birth_year": None if rng.random() < 0.13 else rng.randint(1850, 1985),
                       "lived_in": [c for c in rng.sample(range(n_city), 2) if c != bc][:rng.randint(1, 2)]})

    companies = []
    while len(companies) < n_co:
        root = namer.word()
        for suffix in rng.sample(CO_SUFFIXES, rng.choice([1, 1, 2, 3])):
            title = f"{root} {suffix}"
            if title in titles or len(companies) >= n_co:
                continue
            titles.add(title)
            founder = None if rng.random() < 0.13 else rng.randrange(n_person)
            by = people[founder]["birth_year"] if founder is not None else None
            year = by + rng.randint(22, 55) if by else rng.randint(1880, 2015)
            companies.append({"kind": "company", "title": title, "industry": rng.choice(INDUSTRIES),
                              "founder": founder, "founded": None if rng.random() < 0.13 else year,
                              "hq": None if rng.random() < 0.11 else rng.randrange(n_city),
                              "hq_on_city_page": rng.random() < 0.3})
    for co in companies:
        if rng.random() < 0.25:
            alias = f"{namer.word()} {rng.choice(CO_SUFFIXES)}"
            if alias not in titles:
                titles.add(alias)
                co["alias"] = alias
    for p in people:
        p["employer"] = rng.randrange(n_co) if rng.random() < 0.6 else None

    events = []
    while len(events) < n_event:
        series = f"{namer.word()} {rng.choice(EVENT_KINDS)}"
        start = rng.randint(1900, 2000)
        gap = rng.choice([2, 3, 4, 5])
        eds = []
        for i in range(rng.randint(2, 4)):
            if len(events) >= n_event:
                break
            year = start + i * gap
            title = f"{year} {series}"
            titles.add(title)
            eds.append(len(events))
            events.append({"kind": "event", "title": title, "series": series, "year": year,
                           "host": None if rng.random() < 0.1 else rng.randrange(n_city),
                           "sponsor": rng.randrange(n_co) if rng.random() < 0.5 else None})
        for e in eds:
            events[e]["editions"] = eds
    return {"countries": countries, "cities": cities, "people": people, "companies": companies, "events": events}


def render_pages(w, rng):
    cities, people, cos, events = w["cities"], w["people"], w["companies"], w["events"]
    city_natives, city_hq, city_events, country_cities = {}, {}, {}, {}
    for i, p in enumerate(people):
        if p["birth_city"] is not None:
            city_natives.setdefault(p["birth_city"], []).append(i)
    for i, co in enumerate(cos):
        if co["hq"] is not None and co["hq_on_city_page"]:
            city_hq.setdefault(co["hq"], []).append(i)
    for i, e in enumerate(events):
        if e["host"] is not None:
            city_events.setdefault(e["host"], []).append(i)
    for i, c in enumerate(cities):
        country_cities.setdefault(c["country"], []).append(i)
    founded_by = {}
    for i, co in enumerate(cos):
        if co["founder"] is not None:
            founded_by.setdefault(co["founder"], []).append(i)

    rc = lambda: cities[rng.randrange(len(cities))]["title"]
    rp = lambda: people[rng.randrange(len(people))]["title"]
    rco = lambda: cos[rng.randrange(len(cos))]["title"]
    ry = lambda: str(rng.randint(1880, 2020))

    def filler(subject, k):
        pool = [
            lambda: f"In {ry()}, {subject} was the subject of a retrospective organised in {rc()}.",
            lambda: f"Contemporary accounts link {subject} with {rp()}, although the details are disputed.",
            lambda: f"A short profile published in {ry()} described {subject} as influential in {rng.choice(TOPICS)}.",
            lambda: f"Archival letters mention {subject} alongside {rco()} and {rco()}.",
            lambda: f"Several later histories of {rng.choice(TOPICS)} in {rc()} cite {subject} only briefly.",
            lambda: f"Records from {rc()} dated {ry()} refer to {subject} in passing.",
            lambda: f"Commentators have compared {subject} with {rp()} and with {rco()}.",
            lambda: f"Much of the surviving documentation about {subject} concerns {rng.choice(TOPICS)}.",
            lambda: f"An exhibition in {rc()} in {ry()} included material on {subject}.",
            lambda: f"{subject} appears in a {ry()} survey of {rng.choice(TOPICS)} compiled by {rp()}.",
            lambda: f"Correspondence held in {rc()} shows that {subject} dealt regularly with {rco()}.",
            lambda: f"A {ry()} dispute involving {subject} and {rco()} was settled out of court.",
            lambda: f"Local newspapers in {rc()} reported on {subject} during the {ry()} season.",
            lambda: f"The papers of {rp()} contain several references to {subject}.",
            lambda: f"Scholars disagree about the role of {subject} in the development of {rng.choice(TOPICS)}.",
        ]
        return [f() for f in rng.sample(pool, k)]

    def assemble(title, lead, facts, n_fill):
        body = filler(title, n_fill)
        for f in facts:
            body.insert(rng.randint(0, len(body)), f)
        paras = [" ".join(body[i:i + 4]) for i in range(0, len(body), 4)]
        return "\n\n".join([f"# {title}", lead] + paras)

    pages = []
    for name in w["countries"]:
        cs = ", ".join(cities[c]["title"] for c in sorted(country_cities.get(name, [])))
        pages.append((name, f"# {name}\n\n{name} is a country. Its cities include {cs}."))
    for i, c in enumerate(cities):
        t = c["title"]
        facts = [f"{t} is a city in {c['country']}."]
        if "alias" in c:
            facts.append(f"{t} was formerly called {c['alias']}.")
        for e in sorted(city_events.get(i, [])):
            facts.append(f"The {events[e]['title']} was held in {t}.")
        for co in sorted(city_hq.get(i, [])):
            facts.append(f"{cos[co]['title']} is headquartered in {t}.")
        natives = city_natives.get(i, [])
        if natives:
            facts.append("Notable people born in " + t + " include " + "; ".join(people[p]["title"] for p in natives) + ".")
        lead = f"{t} is a settlement founded around {c['founded']} with a population of about {c['population']:,}."
        pages.append((t, assemble(t, lead, facts, rng.randint(5, 12))))
    for i, p in enumerate(people):
        nm = p["name"]
        facts = []
        facts.append(f"{nm} was born in {cities[p['birth_city']]['title']}." if p["birth_city"] is not None
                     else f"{nm}'s place of birth is not recorded.")
        facts.append(f"{nm} was born in {p['birth_year']}." if p["birth_year"] is not None
                     else f"{nm}'s year of birth is not recorded.")
        for c in p["lived_in"]:
            facts.append(f"{nm} later moved to {cities[c]['title']}.")
        if p["employer"] is not None:
            facts.append(f"{nm} worked for {cos[p['employer']]['title']} for several years.")
        for co in founded_by.get(i, []):
            facts.append(f"{nm} founded {cos[co]['title']}.")
        lead = f"{nm} is {'an' if p['occupation'][0] in 'aeiou' else 'a'} {p['occupation']}."
        pages.append((p["title"], assemble(p["title"], lead, facts, rng.randint(6, 14))))
    for co in cos:
        t = co["title"]
        facts = []
        facts.append(f"{t} was founded by {people[co['founder']]['title']}." if co["founder"] is not None
                     else f"The founder of {t} is not recorded.")
        facts.append(f"{t} was founded in {co['founded']}." if co["founded"] is not None
                     else f"The founding year of {t} is not recorded.")
        if co["hq"] is None:
            facts.append(f"The location of {t}'s headquarters is not recorded.")
        elif not co["hq_on_city_page"]:
            facts.append(f"{t} is headquartered in {cities[co['hq']]['title']}.")
        facts.append(f"{t} opened a regional office in {rc()}.")
        if "alias" in co:
            facts.append(f"{t} was formerly known as {co['alias']}.")
        lead = f"{t} is a company active in {co['industry']}."
        pages.append((t, assemble(t, lead, facts, rng.randint(6, 14))))
    for e in events:
        t = e["title"]
        facts = [f"The {t} was held in {cities[e['host']]['title']}." if e["host"] is not None
                 else f"The host city of the {t} is not recorded."]
        if e["sponsor"] is not None:
            facts.append(f"The {t} was sponsored by {cos[e['sponsor']]['title']}.")
        others = [events[o]["title"] for o in e["editions"] if events[o] is not e]
        facts.append(f"Other editions of the {e['series']}: {', '.join(others)}.")
        lead = f"The {t} was an edition of the {e['series']}."
        pages.append((t, assemble(t, lead, facts, rng.randint(4, 10))))
    rng.shuffle(pages)
    return [{"id": f"pg{i:06d}", "title": t, "text": x} for i, (t, x) in enumerate(pages)], city_events
