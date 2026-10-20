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
import json
import random
import textwrap
from dataclasses import dataclass, field
from string import Template

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "INR"]
RATES = {"USD": 1000, "EUR": 920, "GBP": 790, "JPY": 149300, "CHF": 880, "CAD": 1360, "INR": 83100}
REGIONS = {"eu": "EUR", "uk": "GBP", "jp": "JPY", "ch": "CHF", "ca": "CAD", "in": "INR"}
NUM_PREFIXES = ["qty", "stock", "score"]
STR_PREFIXES = ["user", "team"]
WORDS = ["amber", "birch", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper",
         "kestrel", "lumen", "maple", "nimbus", "onyx", "pine", "quartz", "raven", "sierra", "tundra"]
VERBS = ["compute", "build", "resolve", "settle", "collect", "render", "plan", "score", "audit", "route",
         "merge", "tally", "project", "stage", "check", "price", "summarize", "rank", "reconcile", "draft"]
NOUNS = ["invoice", "order", "ledger", "batch", "shipment", "account", "session", "report", "window", "payout",
         "refund", "ticket", "budget", "quota", "schedule", "profile", "bundle", "cohort", "manifest", "digest"]
AREAS = ["billing", "fulfillment", "risk", "growth", "payments", "logistics", "support", "catalog", "search", "ads"]
SETTING_KEYS = {
    "db": ["port", "pool", "retries", "shards", "replicas"],
    "net": ["timeout", "backoff", "jitter", "keepalive"],
    "feature": ["beta", "audit", "export", "dark_mode", "fast_path"],
    "app": ["name", "region", "tier", "owner"],
}
LEGACY_NAMES = {
    "money": ["convert", "ConversionError"],
    "store": ["fetch", "put"],
    "clock": ["span", "shift"],
    "events": ["emit"],
    "config": ["get"],
}
MODULE_ALIASES = {
    "money": ["lmoney", "legacy_money", "_fx", "mny"],
    "store": ["lstore", "legacy_store", "_db", "kv"],
    "clock": ["lclock", "legacy_clock", "_clk", "tm"],
    "events": ["levents", "legacy_events", "_bus", "evt"],
    "config": ["lconfig", "legacy_config", "_cfg", "conf"],
}
NAME_ALIASES = {
    "convert": ["convert", "_convert", "to_currency", "fx_convert"],
    "ConversionError": ["ConversionError"],
    "fetch": ["fetch", "_fetch", "fetch_row", "read_row"],
    "put": ["put", "_put", "write_row"],
    "span": ["span", "_span", "seconds_between"],
    "shift": ["shift", "_shift", "move_ts"],
    "emit": ["emit", "_emit", "broadcast"],
    "get": ["get", "_get", "cfg_get", "read_setting"],
}


def T(text, **kw):
    return textwrap.dedent(Template(text).substitute(kw)).strip("\n")


@dataclass
class Snippet:
    old: str
    gold: str
    cases: object  # callable(rng) -> [(fn, args)], called once for hidden and once for visible inputs
    family: str = ""
    functools: bool = False


@dataclass
class World:
    rng: random.Random
    store: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)

    def keys(self, prefix, present=True):
        ids = range(60)
        return [f"{prefix}:{i}" for i in ids if (f"{prefix}:{i}" in self.store) == present]

    def ids(self, prefix, present=True):
        return [int(k.split(":")[1]) for k in self.keys(prefix, present)]

    def setting_keys(self, section, present=True):
        return [k for k in SETTING_KEYS[section] if (f"{section}.{k}" in self.settings) == present]


def make_world_data(rng):
    w = World(rng)
    for prefix in NUM_PREFIXES + STR_PREFIXES:
        for i in range(60):
            if rng.random() < 0.7:
                value = rng.randint(0, 500) if prefix in NUM_PREFIXES else rng.choice(WORDS) + str(rng.randint(1, 99))
                w.store[f"{prefix}:{i}"] = (value, rng.randint(1, 9))
    for key in SETTING_KEYS["db"]:
        if rng.random() < 0.75:
            w.settings[f"db.{key}"] = str(rng.randint(1, 9000))
    for key in SETTING_KEYS["net"]:
        if rng.random() < 0.75:
            w.settings[f"net.{key}"] = str(round(rng.uniform(0.1, 30), 2))
    for key in SETTING_KEYS["feature"]:
        w.settings[f"feature.{key}"] = rng.choice(["true", "false", "yes", "no", "0", "1", "off", "on"])
    for key in SETTING_KEYS["app"]:
        if rng.random() < 0.75:
            w.settings[f"app.{key}"] = rng.choice(WORDS).capitalize() + "-" + str(rng.randint(1, 9))
    # Guarantee at least one missing key per numeric section so fallback paths stay testable.
    for section in ("db", "net"):
        if not w.setting_keys(section, present=False):
            w.settings.pop(f"{section}.{SETTING_KEYS[section][0]}")
    # ... and at least one present net key (all four missing crashed generate at n=64 seed 601; no frozen instance
    # takes this branch, so their bytes are unchanged)
    if not w.setting_keys("net"):
        w.settings[f"net.{SETTING_KEYS['net'][-1]}"] = "5.0"
    return w


def data_module_source(w):
    lines = ["# Seed data for the in-memory store and settings. Do not edit.", f"RATES = {RATES!r}", "", "STORE = {"]
    lines += [f"    {k!r}: {v!r}," for k, v in sorted(w.store.items(), key=lambda kv: (kv[0].split(':')[0], int(kv[0].split(':')[1])))]
    lines += ["}", "", "SETTINGS = {"]
    lines += [f"    {k!r}: {v!r}," for k, v in sorted(w.settings.items())]
    lines += ["}", ""]
    return "\n".join(lines)


class Names:
    def __init__(self, rng, reserved=()):
        self.rng = rng
        self.used = set(reserved)

    def fn(self):
        while True:
            name = f"{self.rng.choice(VERBS)}_{self.rng.choice(NOUNS)}"
            if name not in self.used:
                self.used.add(name)
                return name


def legacy_imports(rng, family):
    style = rng.choice(["module_alias", "names", "dotted"])
    names = LEGACY_NAMES[family]
    if style == "module_alias":
        alias = rng.choice(MODULE_ALIASES[family])
        return [f"from acme.legacy import {family} as {alias}"], {n: f"{alias}.{n}" for n in names}
    if style == "dotted":
        return [f"import acme.legacy.{family}"], {n: f"acme.legacy.{family}.{n}" for n in names}
    refs, parts = {}, []
    for n in names:
        alias = rng.choice(NAME_ALIASES[n])
        refs[n] = alias
        parts.append(n if alias == n else f"{n} as {alias}")
    return [f"from acme.legacy.{family} import {', '.join(parts)}"], refs


def amounts(rng, k):
    out = []
    for _ in range(k):
        r = rng.random()
        if r < 0.3:
            out.append(rng.randint(0, 2000))
        elif r < 0.7:
            out.append(round(rng.uniform(0, 500), 3))
        else:
            out.append(round(rng.randint(0, 50000) / 100 + 0.005, 3))
    return out


def conv(src, dst, expr):
    return f'money.exchange("{src}", "{dst}", round({expr} * 100)).value / 100'


# ---- money ----

def money_direct(rng, nm, r, w):
    fn, dst = nm.fn(), rng.choice(CURRENCIES[1:])
    head = 'def $fn(amount):\n    """USD amount expressed in $dst."""\n'
    old = T(head + '    return $C(amount, "$dst")', fn=fn, dst=dst, C=r["convert"])
    gold = T(head + "    return $G", fn=fn, dst=dst, G=conv("USD", dst, "amount"))
    return Snippet(old, gold, lambda g: [(fn, [a]) for a in amounts(g, 3)])


def money_kwargs(rng, nm, r, w):
    fn = nm.fn()
    src, dst = rng.sample(CURRENCIES, 2)
    markup = rng.choice(["1.05", "1.1", "0.97", "1.2"])
    body = """
        def $fn(qty, unit_price):
            subtotal = qty * unit_price
            converted = $X
            return round(converted * $markup, 2)
    """
    old = T(body, fn=fn, markup=markup, X=f'{r["convert"]}(dst="{dst}", amount=subtotal, src="{src}")')
    gold = T(body, fn=fn, markup=markup, X=conv(src, dst, "subtotal"))
    return Snippet(old, gold, lambda g: [(fn, [g.randint(1, 20), round(g.uniform(0.5, 80), 2)]) for _ in range(3)])


def money_fallback(rng, nm, r, w):
    fn, src = nm.fn(), rng.choice(CURRENCIES)
    fb = rng.choice(["None", "0.0", "-1.0"])
    old = T("""
        def $fn(amount, currency):
            try:
                return $C(amount, currency, src="$src")
            except $E:
                return $fb
    """, fn=fn, src=src, fb=fb, C=r["convert"], E=r["ConversionError"])
    gold = T("""
        def $fn(amount, currency):
            res = money.exchange("$src", currency, round(amount * 100))
            if not res.ok:
                return $fb
            return res.value / 100
    """, fn=fn, src=src, fb=fb)
    def cases(g):
        a = amounts(g, 3)
        return [(fn, [a[0], g.choice(CURRENCIES)]), (fn, [a[1], g.choice(CURRENCIES)]), (fn, [a[2], g.choice(["XYZ", "BTC", "usd"])])]
    return Snippet(old, gold, cases)


def money_partial(rng, nm, r, w):
    fn = nm.fn()
    src, dst = rng.sample(CURRENCIES, 2)
    helper = f"_{fn}_one"
    tail = textwrap.dedent("""


        def $fn(amounts):
            return [$h(a) for a in amounts if a > 0]
    """)
    old = T('$h = functools.partial($C, dst="$dst", src="$src")' + tail, fn=fn, h=helper, C=r["convert"], dst=dst, src=src)
    gold = T("def $h(amount):\n    return $G" + tail, fn=fn, h=helper, G=conv(src, dst, "amount"))
    return Snippet(old, gold, lambda g: [(fn, [amounts(g, 4) + [0, -3.5]]), (fn, [amounts(g, 2)])], functools=True)


def money_strict(rng, nm, r, w):
    fn = nm.fn()
    helper = f"_{fn}_one"
    tail = textwrap.dedent("""


        def $fn(items, currency, strict=False):
            total = 0.0
            for amount in items:
                total += $h(amount, currency, strict)
            return round(total, 2)
    """)
    old = T(textwrap.dedent("""
        def $h(amount, currency, strict):
            if strict:
                return $C(amount, currency)
            try:
                return $C(amount, currency)
            except $E:
                return 0.0
    """) + tail, fn=fn, h=helper, C=r["convert"], E=r["ConversionError"])
    gold = T(textwrap.dedent("""
        def $h(amount, currency, strict):
            res = money.exchange("USD", currency, round(amount * 100))
            if res.ok:
                return res.value / 100
            if strict:
                raise ValueError(res.error)
            return 0.0
    """) + tail, fn=fn, h=helper)
    return Snippet(old, gold, lambda g: [(fn, [amounts(g, 3), g.choice(CURRENCIES), False]), (fn, [amounts(g, 2), "ZZZ", False]),
                                         (fn, [amounts(g, 3), g.choice(CURRENCIES), True])])


def money_table(rng, nm, r, w):
    fn = nm.fn()
    tbl = f"_{fn.upper()}_RATES"
    (k1, d1), (k2, d2) = rng.sample(sorted(REGIONS.items()), 2)
    s2 = rng.choice([c for c in CURRENCIES if c != d2])
    tail = textwrap.dedent("""


        def $fn(region, amount):
            conv = $tbl.get(region)
            if conv is None:
                return None
            return conv(amount)
    """)
    body = '$tbl = {\n    "$k1": lambda amount: $A,\n    "$k2": lambda amount: $B,\n}' + tail
    kw = dict(fn=fn, tbl=tbl, k1=k1, k2=k2)
    old = T(body, A=f'{r["convert"]}(amount, "{d1}")', B=f'{r["convert"]}(amount, "{d2}", "{s2}")', **kw)
    gold = T(body, A=conv("USD", d1, "amount"), B=conv(s2, d2, "amount"), **kw)
    def cases(g):
        a = amounts(g, 3)
        return [(fn, [k1, a[0]]), (fn, [k2, a[1]]), (fn, ["zz", a[2]])]
    return Snippet(old, gold, cases)


# ---- store ----

def store_unpack(rng, nm, r, w):
    fn, p = nm.fn(), rng.choice(NUM_PREFIXES + STR_PREFIXES)
    old = T("""
        def $fn(uid):
            value, version = $F("$p:" + str(uid))
            return "%s@v%d" % (value, version)
    """, fn=fn, p=p, F=r["fetch"])
    gold = T("""
        def $fn(uid):
            rec = store.lookup("$p:" + str(uid))
            return "%s@v%d" % (rec.value, rec.version)
    """, fn=fn, p=p)
    return Snippet(old, gold, lambda g: [(fn, [i]) for i in g.sample(w.ids(p), 2)])


def store_default_none(rng, nm, r, w):
    fn, p, k = nm.fn(), rng.choice(NUM_PREFIXES), rng.randint(2, 9)
    old = T("""
        def $fn(key, scale=$k):
            row = $F(key, None)
            if row is None:
                return -1
            return row[0] * scale
    """, fn=fn, k=k, F=r["fetch"])
    gold = T("""
        def $fn(key, scale=$k):
            rec = store.lookup(key)
            if rec is None:
                return -1
            return rec.value * scale
    """, fn=fn, k=k)
    return Snippet(old, gold, lambda g: [(fn, [g.choice(w.keys(p))]), (fn, [g.choice(w.keys(p)), 3]), (fn, [g.choice(w.keys(p, False))])])


def store_default_tuple(rng, nm, r, w):
    fn, p, dflt = nm.fn(), rng.choice(STR_PREFIXES), rng.choice(["nobody", "unknown", "guest"])
    old = T("""
        def $fn(uid):
            name, _ = $F("$p:%d" % uid, ("$dflt", 0))
            return name.upper()
    """, fn=fn, p=p, dflt=dflt, F=r["fetch"])
    gold = T("""
        def $fn(uid):
            rec = store.lookup("$p:%d" % uid)
            name = rec.value if rec is not None else "$dflt"
            return name.upper()
    """, fn=fn, p=p, dflt=dflt)
    return Snippet(old, gold, lambda g: [(fn, [g.choice(w.ids(p))]), (fn, [g.choice(w.ids(p, False))])])


def store_versions(rng, nm, r, w):
    fn = nm.fn()
    old = T("""
        def $fn(keys):
            found = {}
            for key in keys:
                try:
                    found[key] = $F(key)[1]
                except KeyError:
                    continue
            return found
    """, fn=fn, F=r["fetch"])
    gold = T("""
        def $fn(keys):
            found = {}
            for key in keys:
                rec = store.lookup(key)
                if rec is None:
                    continue
                found[key] = rec.version
            return found
    """, fn=fn)
    p = rng.choice(NUM_PREFIXES + STR_PREFIXES)

    def cases(g):
        keys = g.sample(w.keys(p), 3) + g.sample(w.keys(p, False), 2)
        g.shuffle(keys)
        return [(fn, [keys])]
    return Snippet(old, gold, cases)


def store_bump(rng, nm, r, w):
    fn, p, d = nm.fn(), rng.choice(NUM_PREFIXES), rng.randint(1, 5)
    old = T("""
        def $fn(key, delta=$d):
            value, _ = $F(key, (0, 0))
            return $P(key, value + delta)
    """, fn=fn, d=d, F=r["fetch"], P=r["put"])
    gold = T("""
        def $fn(key, delta=$d):
            rec = store.lookup(key)
            value = rec.value if rec is not None else 0
            return store.save(key, value + delta).version
    """, fn=fn, d=d)
    def cases(g):
        key = g.choice(w.keys(p))
        return [(fn, [key]), (fn, [key, 7]), (fn, [g.choice(w.keys(p, False))])]
    return Snippet(old, gold, cases)


def store_provider(rng, nm, r, w, name):
    head = 'def load_$name(key):\n    """Public row loader; other modules call this."""\n'
    old = T(head + "    return $F(key)", name=name, F=r["fetch"])
    gold = T(head + "    return store.lookup(key)", name=name)
    p = rng.choice(NUM_PREFIXES + STR_PREFIXES)
    return Snippet(old, gold, lambda g: [(f"load_{name}", [k]) for k in g.sample(w.keys(p), 2)])


def dependent(rng, nm, w, ref):
    fn = nm.fn()
    if rng.random() < 0.5:
        th = rng.randint(2, 7)
        old = T("""
            def $fn(key):
                value, version = $L(key)
                return {"key": key, "value": value, "fresh": version > $th}
        """, fn=fn, L=ref, th=th)
        gold = T("""
            def $fn(key):
                rec = $L(key)
                return {"key": key, "value": rec.value, "fresh": rec.version > $th}
        """, fn=fn, L=ref, th=th)
        p = rng.choice(NUM_PREFIXES + STR_PREFIXES)
        return Snippet(old, gold, lambda g: [(fn, [k]) for k in g.sample(w.keys(p), 2)])
    old = T("def $fn(keys):\n    return sum($L(k)[1] for k in keys)", fn=fn, L=ref)
    gold = T("def $fn(keys):\n    return sum($L(k).version for k in keys)", fn=fn, L=ref)
    p = rng.choice(NUM_PREFIXES + STR_PREFIXES)
    return Snippet(old, gold, lambda g: [(fn, [g.sample(w.keys(p), 3)])])


# ---- clock ----

def clock_span(rng, nm, r, w):
    fn, tail = nm.fn(), rng.choice(["", " / 3600", " // 60", " * 2"])
    old = T("def $fn(start, end):\n    return $S(start, end)$tail", fn=fn, S=r["span"], tail=tail)
    gold = T("""
        def $fn(start, end):
            elapsed = clock.elapsed_ms(end * 1000, start * 1000)
            if elapsed < 0:
                raise ValueError("negative span")
            return elapsed // 1000$tail
    """, fn=fn, tail=tail)
    def cases(g):
        starts = [g.randint(1_600_000_000, 1_700_000_000) for _ in range(3)]
        return [(fn, [s, s + g.randint(0, 200_000)]) for s in starts]
    return Snippet(old, gold, cases)


def clock_guarded(rng, nm, r, w):
    fn = nm.fn()
    old = T("""
        def $fn(t0, t1):
            try:
                secs = $S(t0, t1)
            except ValueError:
                secs = 0
            return secs // 60
    """, fn=fn, S=r["span"])
    gold = T("""
        def $fn(t0, t1):
            elapsed = clock.elapsed_ms(t1 * 1000, t0 * 1000)
            secs = elapsed // 1000 if elapsed >= 0 else 0
            return secs // 60
    """, fn=fn)
    def cases(g):
        t = g.randint(1_600_000_000, 1_700_000_000)
        return [(fn, [t, t + g.randint(60, 90_000)]), (fn, [t, t - g.randint(1, 5000)])]
    return Snippet(old, gold, cases)


def clock_ms_floor(rng, nm, r, w):
    fn = nm.fn()
    old = T("def $fn(start_ms, end_ms):\n    return $S(start_ms // 1000, end_ms // 1000) * 1000", fn=fn, S=r["span"])
    gold = T("""
        def $fn(start_ms, end_ms):
            elapsed = clock.elapsed_ms(end_ms // 1000 * 1000, start_ms // 1000 * 1000)
            if elapsed < 0:
                raise ValueError("negative span")
            return elapsed
    """, fn=fn)
    def cases(g):
        starts = [g.randint(1_600_000_000_000, 1_700_000_000_000) for _ in range(3)]
        return [(fn, [s, s + g.randint(1, 900_000)]) for s in starts]
    return Snippet(old, gold, cases)


def clock_shift_kw(rng, nm, r, w):
    fn, m = nm.fn(), rng.choice([5, 15, 30, 45])
    old = T("def $fn(ts, hours):\n    return $H(ts, hours=hours, minutes=$m)", fn=fn, m=m, H=r["shift"])
    gold = T("def $fn(ts, hours):\n    return clock.offset(ts * 1000, (hours * 3600 + $m * 60) * 1000) // 1000", fn=fn, m=m)
    def cases(g):
        t = g.randint(1_600_000_000, 1_700_000_000)
        return [(fn, [t, g.randint(-5, 48)]), (fn, [t + 17, 0])]
    return Snippet(old, gold, cases)


def clock_shift_pos(rng, nm, r, w):
    fn, m = nm.fn(), rng.choice([2, 10, 90])
    old = T("def $fn(stamps):\n    return [$H(ts, $m) for ts in stamps]", fn=fn, m=m, H=r["shift"])
    gold = T("def $fn(stamps):\n    return [clock.offset(ts * 1000, $m * 60 * 1000) // 1000 for ts in stamps]", fn=fn, m=m)
    return Snippet(old, gold, lambda g: [(fn, [[g.randint(1_600_000_000, 1_700_000_000) for _ in range(3)]])])


def clock_table(rng, nm, r, w):
    fn = nm.fn()
    tbl = f"_{fn.upper()}_UNITS"
    tail = textwrap.dedent("""


        def $fn(ts, unit, k):
            return $tbl[unit](ts, k)
    """)
    body = '$tbl = {\n    "m": lambda ts, k: $A,\n    "h": lambda ts, k: $B,\n    "s": lambda ts, k: ts + k,\n}' + tail
    old = T(body, fn=fn, tbl=tbl, A=f'{r["shift"]}(ts, minutes=k)', B=f'{r["shift"]}(ts, hours=k)')
    gold = T(body, fn=fn, tbl=tbl, A="clock.offset(ts * 1000, k * 60000) // 1000",
             B="clock.offset(ts * 1000, k * 3600000) // 1000")
    def cases(g):
        t = g.randint(1_600_000_000, 1_700_000_000)
        return [(fn, [t, u, g.randint(1, 50)]) for u in ("m", "h", "s")]
    return Snippet(old, gold, cases)


# ---- events ----

def events_all(rng, nm, r, w):
    fn, topic = nm.fn(), rng.choice(["audit", "billing", "metrics"])
    old = T("""
        def $fn(record_id, amount):
            replies = $EM("$topic", {"id": record_id, "amount": amount})
            return list(replies)
    """, fn=fn, topic=topic, EM=r["emit"])
    gold = T("""
        def $fn(record_id, amount):
            replies = []
            events.publish("$topic", {"id": record_id, "amount": amount}, replies.append)
            replies.reverse()
            return replies
    """, fn=fn, topic=topic)
    return Snippet(old, gold, lambda g: [(fn, [g.randint(1, 999), g.randint(0, 300)]) for _ in range(2)])


def events_first(rng, nm, r, w):
    fn = nm.fn()
    head = textwrap.dedent("""
        def $fn(user, urgent=False):
            payload = {"user": user, "urgent": urgent}
    """)
    old = T(head + '    return $EM("notify", payload)[0]', fn=fn, EM=r["emit"])
    gold = T(head + '    replies = []\n    events.publish("notify", payload, replies.append)\n    return replies[-1]', fn=fn)
    return Snippet(old, gold, lambda g: [(fn, [g.choice(WORDS)]), (fn, [g.choice(WORDS), True])])


def events_count(rng, nm, r, w):
    fn, topic = nm.fn(), rng.choice(["audit", "billing", "notify", "metrics", "archive"])
    body = """
        def $fn(items):
            sent = 0
            for item in items:
                sent += $X
            return sent
    """
    old = T(body, fn=fn, X=f'len({r["emit"]}("{topic}", {{"id": item}}))')
    gold = T(body, fn=fn, X=f'events.publish("{topic}", {{"id": item}}, lambda reply: None)')
    return Snippet(old, gold, lambda g: [(fn, [[g.randint(1, 99) for _ in range(g.randint(1, 4))]])])


def events_filter(rng, nm, r, w):
    fn = nm.fn()
    old = T('def $fn(amount):\n    return [r for r in $EM("billing", {"amount": amount}) if r]', fn=fn, EM=r["emit"])
    gold = T("""
        def $fn(amount):
            replies = []
            events.publish("billing", {"amount": amount}, replies.append)
            return [r for r in reversed(replies) if r]
    """, fn=fn)
    return Snippet(old, gold, lambda g: [(fn, [g.choice([0, 1, 100, 101])]), (fn, [g.randint(2, 99)]), (fn, [g.randint(102, 900)])])


# ---- config ----

def config_int(rng, nm, r, w):
    fn, dflt, k = nm.fn(), rng.randint(1, 50), rng.randint(0, 9)
    key = rng.choice(SETTING_KEYS["db"])
    old = T('def $fn():\n    return $G("db", "$key", $dflt, int) + $k', fn=fn, key=key, dflt=dflt, k=k, G=r["get"])
    gold = T("""
        def $fn():
            raw = config.setting("db.$key", None)
            return ($dflt if raw is None else int(raw)) + $k
    """, fn=fn, key=key, dflt=dflt, k=k)
    return Snippet(old, gold, lambda g: [(fn, [])])


def config_float_kw(rng, nm, r, w):
    fn, fb = nm.fn(), rng.choice(["0.0", "-1.0", "None"])
    old = T("""
        def $fn(key):
            timeout = $G(key=key, section="net", cast=float)
            if timeout is None:
                return $fb
            return timeout * 2
    """, fn=fn, fb=fb, G=r["get"])
    gold = T("""
        def $fn(key):
            raw = config.setting("net." + key, None)
            if raw is None:
                return $fb
            return float(raw) * 2
    """, fn=fn, fb=fb)
    return Snippet(old, gold, lambda g: [(fn, [g.choice(w.setting_keys("net"))]), (fn, [g.choice(w.setting_keys("net", False))])])


def config_bool(rng, nm, r, w):
    fn = nm.fn()
    on, off = rng.choice([("on", "off"), ("enabled", "disabled"), ("yes", "no")])
    tail = '        return "$on"\n    return "$off"'
    old = T('def $fn(flag):\n    if $G("feature", flag, cast=bool):\n' + tail, fn=fn, on=on, off=off, G=r["get"])
    gold = T('def $fn(flag):\n    if config.setting("feature." + flag, None):\n' + tail, fn=fn, on=on, off=off)
    flags = SETTING_KEYS["feature"]
    falsy = [k for k in flags if w.settings[f"feature.{k}"] in ("false", "no", "0", "off")]
    def cases(g):
        picks = g.sample(flags, 2) + ([g.choice(falsy)] if falsy else []) + [g.choice(["missing_flag", "beta2", "nope"])]
        return [(fn, [f]) for f in picks]
    return Snippet(old, gold, cases)


def config_partial(rng, nm, r, w):
    fn = nm.fn()
    helper = f"_{fn}_app"
    tail = '\n\n\ndef $fn(name):\n    return str($h(name, "unset")).lower()'
    old = T('$h = functools.partial($G, "app")' + tail, fn=fn, h=helper, G=r["get"])
    gold = T('def $h(name, default=None):\n    return config.setting("app." + name, default)' + tail, fn=fn, h=helper)
    keys = SETTING_KEYS["app"]
    return Snippet(old, gold, lambda g: [(fn, [k]) for k in g.sample(keys, 2)] + [(fn, [g.choice(["nope", "zone", "color"])])], functools=True)


def config_dynamic(rng, nm, r, w):
    fn = nm.fn()
    old = T("""
        def $fn(section, key):
            value = $G(section, key)
            return value if value is not None else "<missing>"
    """, fn=fn, G=r["get"])
    gold = T("""
        def $fn(section, key):
            value = config.setting(section + "." + key, None)
            return value if value is not None else "<missing>"
    """, fn=fn)
    def cases(g):
        sections = [g.choice(sorted(SETTING_KEYS)) for _ in range(3)]
        return [(fn, [s, g.choice(SETTING_KEYS[s] + ["zzz"])]) for s in sections]
    return Snippet(old, gold, cases)


TEMPLATES = {
    "money": [money_direct, money_kwargs, money_fallback, money_partial, money_strict, money_table],
    "store": [store_unpack, store_default_none, store_default_tuple, store_versions, store_bump],
    "clock": [clock_span, clock_guarded, clock_ms_floor, clock_shift_kw, clock_shift_pos, clock_table],
    "events": [events_all, events_first, events_count, events_filter],
    "config": [config_int, config_float_kw, config_bool, config_partial, config_dynamic],
}


# ---- fillers (no API use; tests check they keep working) ----

def rand_text(rng):
    parts = [rng.choice(WORDS + ["Q3", "v2", "A&B", "x"]) for _ in range(rng.randint(2, 5))]
    return rng.choice([" ", "-", "  ", ", ", "/"]).join(parts)


def filler_slug(rng, nm):
    fn, sep = nm.fn(), rng.choice(["-", "_", "."])
    code = T("""
        def $fn(text):
            out = []
            for ch in text.lower():
                if ch.isalnum():
                    out.append(ch)
                elif out and out[-1] != "$sep":
                    out.append("$sep")
            return "".join(out).strip("$sep")
    """, fn=fn, sep=sep)
    return code, [(fn, [rand_text(rng)]) for _ in range(2)]


def filler_checksum(rng, nm):
    fn = nm.fn()
    code = T("""
        def $fn(text):
            total = $seed
            for i, ch in enumerate(text):
                total = (total * $mul + ord(ch) * (i + 1)) % $mod
            return total
    """, fn=fn, seed=rng.randint(1, 99), mul=rng.choice([31, 37, 131]), mod=rng.choice([65521, 1000003, 9973]))
    return code, [(fn, [rand_text(rng)]) for _ in range(2)]


def filler_moving_avg(rng, nm):
    fn = nm.fn()
    code = T("""
        def $fn(values, width=$w):
            if width <= 0 or len(values) < width:
                return []
            out = []
            for i in range(len(values) - width + 1):
                window = values[i:i + width]
                out.append(round(sum(window) / width, 3))
            return out
    """, fn=fn, w=rng.randint(2, 4))
    return code, [(fn, [[rng.randint(-50, 200) for _ in range(rng.randint(3, 8))]]), (fn, [[1, 2], 5])]


def filler_clamp(rng, nm):
    fn, lo = nm.fn(), rng.randint(-10, 10)
    code = T("def $fn(values, lo=$lo, hi=$hi):\n    return [min(hi, max(lo, v)) for v in values]", fn=fn, lo=lo, hi=lo + rng.randint(5, 80))
    return code, [(fn, [[rng.randint(-100, 150) for _ in range(5)]])]


def filler_pairs(rng, nm):
    fn, sep = nm.fn(), rng.choice([";", "|", "&"])
    code = T("""
        def $fn(text):
            pairs = {}
            for chunk in text.split("$sep"):
                if "=" not in chunk:
                    continue
                key, _, value = chunk.partition("=")
                pairs[key.strip()] = value.strip()
            return pairs
    """, fn=fn, sep=sep)
    text = sep.join(f"{rng.choice(WORDS)}={rng.randint(0, 99)}" for _ in range(3)) + sep + "junk"
    return code, [(fn, [text])]


def filler_buckets(rng, nm):
    fn = nm.fn()
    tbl = f"_{fn.upper()}_EDGES"
    edges = sorted(rng.sample(range(0, 500, 10), 3))
    code = T("""
        $tbl = $edges


        def $fn(values):
            counts = [0] * (len($tbl) + 1)
            for v in values:
                i = 0
                while i < len($tbl) and v >= $tbl[i]:
                    i += 1
                counts[i] += 1
            return counts
    """, fn=fn, tbl=tbl, edges=repr(tuple(edges)))
    return code, [(fn, [[rng.randint(0, 600) for _ in range(8)]])]


def filler_weights(rng, nm):
    fn = nm.fn()
    tbl = f"_{fn.upper()}_WEIGHTS"
    weights = {wd: rng.randint(1, 9) for wd in rng.sample(WORDS, 5)}
    code = T("""
        $tbl = $weights


        def $fn(tags):
            return sum($tbl.get(t, $miss) for t in tags)
    """, fn=fn, tbl=tbl, weights=repr(weights), miss=rng.randint(0, 2))
    return code, [(fn, [rng.sample(WORDS, 4)])]


def filler_rle(rng, nm):
    fn = nm.fn()
    code = T("""
        def $fn(text):
            out = []
            for ch in text:
                if out and out[-1][0] == ch:
                    out[-1][1] += 1
                else:
                    out.append([ch, 1])
            return "".join("%s%d" % (c, n) for c, n in out)
    """, fn=fn)
    text = "".join(rng.choice("aab") * rng.randint(1, 4) for _ in range(5))
    return code, [(fn, [text])]


FILLERS = [filler_slug, filler_checksum, filler_moving_avg, filler_clamp, filler_pairs, filler_buckets,
           filler_weights, filler_rle]


# ---- modules ----

@dataclass
class Module:
    name: str
    old: str
    gold: str
    cases: list
    visible: list


def build_module(rng, w, name, families, n_snippets, control=False, provider=None, dep_ref=None):
    nm = Names(rng)
    old_imports, refs = [], {}
    for fam in families:
        lines, r = legacy_imports(rng, fam)
        old_imports += lines
        refs[fam] = r
    snippets, used_templates = [], set()

    def add(fam):
        fresh = [t for t in TEMPLATES[fam] if t not in used_templates]
        make = rng.choice(fresh or TEMPLATES[fam])
        used_templates.add(make)
        snippets.append(make(rng, nm, refs[fam], w))
        snippets[-1].family = fam

    for fam in families:
        add(fam)
    while len(snippets) < n_snippets:
        add(rng.choice(families))
    if provider:
        if "store" not in refs:
            lines, refs["store"] = legacy_imports(rng, "store")
            old_imports += lines
        snippets.append(store_provider(rng, nm, refs["store"], w, provider))
        snippets[-1].family = "store"
    dep_import = []
    if dep_ref:
        dep_import, ref = dep_ref
        for _ in range(rng.randint(1, 2)):
            snippets.append(dependent(rng, nm, w, ref))
    fillers = [f(rng, nm) for f in rng.sample(FILLERS, rng.randint(5, 7))]
    vis_rng = random.Random(rng.random())
    blocks = []
    for s in snippets:
        hidden = s.cases(rng)
        blocks.append((s.old, s.gold, hidden, fresh_cases(s.cases, hidden, vis_rng)))
    blocks += [(code, code, cases, []) for code, cases in fillers]
    rng.shuffle(blocks)
    used = sorted({s.family for s in snippets if s.family})
    needs_ft = any(s.functools for s in snippets)

    area, owner = rng.choice(AREAS), rng.choice(WORDS)
    header = [f"# {name}: {area} helpers ({rng.choice(NOUNS)} pipeline).", f"# Owner: team-{owner}."]
    std = ["import functools"] if needs_ft else []
    consts = [f'_SERVICE = "{area}-{owner}"', f'_SCHEMA_VERSION = {rng.randint(1, 12)}']

    def render(code_idx, api_imports):
        imports = [l for l in [*std, *api_imports, *dep_import] if l]
        parts = ["\n".join(header), "\n".join(imports), "\n".join(consts)] + [b[code_idx] for b in blocks]
        return "\n\n\n".join(p for p in parts if p) + "\n"

    gold_api = [f"from acme.api import {', '.join(used)}"] if used else []
    gold = render(1, gold_api)
    old = gold if control else render(0, sorted(old_imports))
    cases = [c for b in blocks for c in b[2]]
    visible = [c for b in blocks for c in b[3]]
    return Module(name, old, gold, cases, visible)


def fresh_cases(make, hidden, rng):
    """Visible inputs that differ from the hidden ones; falls back to repeats only for tiny input domains."""
    seen = {json.dumps(c) for c in hidden}
    for _ in range(20):
        out = [c for c in make(rng) if json.dumps(c) not in seen]
        if out:
            return out
    return make(rng)


MIGRATION_MD = """# Migration: acme.legacy -> acme.api

`acme.legacy` is being deleted. Every module listed in UNITS.txt (under `acme/mods/`) still uses it and must be
rewritten to use `acme.api` instead, with **identical observable behavior**: every public function must return
exactly what it returned before for the same inputs (same values, same types, same fallbacks on error paths).
Each function in `acme/api/*.py` has a docstring section "Replacing legacy ..." with the exact recipe.

## The behavioral changes (all of them)
1. money: `convert(amount, dst, src="USD") -> float` becomes `money.exchange(src, dst, cents) -> Result`.
   - Argument order: **src first**, then dst. Legacy's src defaulted to "USD"; the 3rd positional legacy arg is src.
   - Units: pass `round(amount * 100)` (int cents, rounded like legacy). Return: `res.value / 100` is legacy's float.
   - Errors: legacy raised `ConversionError`; exchange returns `res.ok == False`. Replace `except ConversionError`
     with `if not res.ok:`. If the old code let the error propagate, raise `ValueError(res.error)`.
2. store: `fetch(key) -> (value, version)` becomes `store.lookup(key) -> Record | None`.
   - `Record` has `.value` (old `[0]`) and `.version` (old `[1]`); it cannot be unpacked or indexed.
   - Missing key: legacy raised `KeyError`, or returned the given default as-is (e.g. `None` or `("guest", 0)`).
     lookup returns `None`: handle it exactly like the old except branch / old default.
   - `put(key, value) -> int` becomes `store.save(key, value).version`.
3. clock: `span(start, end)` in seconds becomes `clock.elapsed_ms(end_ms, start_ms)`.
   - Argument order: **end first**. Units: **milliseconds** in and out (`* 1000` on the inputs, `// 1000` on the result).
   - Errors: legacy raised `ValueError` if end < start; elapsed_ms returns a negative number. Where the old code caught
     ValueError, test `< 0`. If the old code let it propagate, raise `ValueError("negative span")`.
   - `shift(ts, minutes=0, hours=0)` (seconds; minutes is the 2nd positional arg) becomes
     `clock.offset(ts * 1000, (minutes * 60 + hours * 3600) * 1000) // 1000`.
4. events: `emit(topic, payload) -> list` becomes `events.publish(topic, payload, on_reply) -> int`.
   - Replies come through the callback (`replies = []; events.publish(topic, payload, replies.append)`).
   - Order: publish delivers in **reverse** registration order. Reverse the collected list to get emit's list
     (emit(...)[0] is replies[-1]). `len(emit(...))` is publish's return value.
5. config: `get(section, key, default=None, cast=None)` becomes `config.setting("section.key", default, kind)`.
   - Path is one string: `section + "." + key`.
   - Missing: legacy returned `default` (None if omitted) and never raised; setting raises `MissingSetting` unless a
     default is passed. Always pass one (`config.setting(path, None)`) and handle `None` like before.
   - Casting: legacy applied `cast(raw)` only when present, never to the default.
   - Bool trap: legacy `cast=bool` is Python `bool(raw)`, so "false", "no" and "0" are True. `kind="bool"` parses
     text and is **not** equivalent. Use `bool(raw)` / `if raw:`.

## Cross-module rule
Public `load_*` functions that return a legacy store row unchanged must now return `store.lookup(key)` (a `Record`,
or `None` when missing) instead of a tuple. Other modules import these loaders (search for `load_`); every caller
must use `.value` / `.version`, consistently with the loader's new return type, even if the loader's module has not
been migrated yet.

## Rules (each is checked by the tests)
- Only edit the existing module files under `acme/mods/`; do not create new files (they are ignored by the tests).
- Remove every import of `acme.legacy` and add `from acme.api import ...` for what you use.
- Keep every function: same names, same signatures, no deletions.
- Every legacy call site must become a call to `acme.api`. Replacing it with hand-written arithmetic or logic
  (e.g. `ts + hours * 3600` instead of `clock.offset`) fails, even if the value is right.
- Migrate each call site in place. Do not add compatibility wrappers: a migrated module may not define any new
  top-level function, class or variable name that the original module did not define (imports are fine, and an
  existing top-level name such as a `functools.partial` helper may be redefined, e.g. as a `def`).
- `run_visible_tests(module)` exercises every function that must change (on different inputs than the hidden
  tests) and checks the rules above. Make it pass completely. A module that imports a not-yet-migrated module fails
  to import during tests until that module is migrated too.
"""
