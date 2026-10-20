"""Type 56 traces: distributed trace analysis over OpenTelemetry-demo-style traces (service and operation names from
the open-telemetry/opentelemetry-demo topology, Apache-2.0; timings simulated once per seed and frozen in the world).
For each request class, name the operation with the largest mean self time. Unit = one request class."""
import json
import random
import re
import threading
from pathlib import Path

from .. import common

TYPE_ID = "56_traces"
TIERS = {"easy": (24, 1800), "medium": (384, 1200), "hard": (1024, 600)}
T = common.load_text(Path(__file__).parent)
TRACES_PER_CLASS = 8
CLIENTS = ["web", "ios", "android", "partner-api", "kiosk", "smart-tv", "batch-export", "load-test"]
REGIONS = ["us-east", "us-west", "eu-west", "eu-central", "ap-south", "ap-northeast", "sa-east", "af-south"]


# Call trees of the demo's user flows: (service, operation, kind, children, mode) with mode "seq", "par" or "async".
def op(svc, name, kids=(), mode="seq", rep=1):
    return {"svc": svc, "op": name, "kids": list(kids), "mode": mode, "rep": rep}


FLOWS = {
    "GET /api/products": op("frontend", "GET /api/products", [
        op("product-catalog", "oteldemo.ProductCatalogService/ListProducts"),
        op("currency", "oteldemo.CurrencyService/Convert", rep=4)], "par"),
    "GET /api/recommendations": op("frontend", "GET /api/recommendations", [
        op("recommendation", "oteldemo.RecommendationService/ListRecommendations", [
            op("flagd", "flagd.evaluation.v1.Service/ResolveBoolean"),
            op("product-catalog", "oteldemo.ProductCatalogService/ListProducts")]),
        op("ad", "oteldemo.AdService/GetAds")], "par"),
    "GET /api/cart": op("frontend", "GET /api/cart", [
        op("cart", "oteldemo.CartService/GetCart", [op("valkey-cart", "HGET")]),
        op("currency", "oteldemo.CurrencyService/Convert", rep=3)], "par"),
    "POST /api/cart": op("frontend", "POST /api/cart", [
        op("cart", "oteldemo.CartService/AddItem", [op("valkey-cart", "HSET"), op("valkey-cart", "EXPIRE")]),
        op("currency", "oteldemo.CurrencyService/Convert", rep=2)], "par"),
    "POST /api/checkout": op("frontend", "POST /api/checkout", [
        op("checkout", "oteldemo.CheckoutService/PlaceOrder", [
            op("cart", "oteldemo.CartService/GetCart", [op("valkey-cart", "HGET")]),
            op("product-catalog", "oteldemo.ProductCatalogService/GetProduct", rep=3),
            op("currency", "oteldemo.CurrencyService/Convert", rep=3),
            op("shipping", "oteldemo.ShippingService/GetQuote", [op("quote", "POST /getquote")]),
            op("payment", "oteldemo.PaymentService/Charge"),
            op("shipping", "oteldemo.ShippingService/ShipOrder"),
            op("email", "POST /send_order_confirmation"),
            op("cart", "oteldemo.CartService/EmptyCart", [op("valkey-cart", "DEL")]),
            op("checkout", "orders publish", [
                op("accounting", "orders process"), op("fraud-detection", "orders process")], "async")])]),
    "GET /api/data": op("frontend", "GET /api/data", [
        op("ad", "oteldemo.AdService/GetAds", [op("flagd", "flagd.evaluation.v1.Service/ResolveBoolean")]),
        op("product-catalog", "oteldemo.ProductCatalogService/GetProduct", rep=2)], "par"),
    "GET /api/shipping": op("frontend", "GET /api/shipping", [
        op("shipping", "oteldemo.ShippingService/GetQuote", [op("quote", "POST /getquote")]),
        op("currency", "oteldemo.CurrencyService/Convert", rep=2)], "par"),
    "GET /api/currency": op("frontend", "GET /api/currency", [
        op("currency", "oteldemo.CurrencyService/GetSupportedCurrencies"),
        op("flagd", "flagd.evaluation.v1.Service/ResolveBoolean", rep=2)], "par"),
}


def flat_ops(node, out=None):
    out = [] if out is None else out
    out.append((node["svc"], node["op"]))
    for k in node["kids"]:
        flat_ops(k, out)
    return out


def build_trace(flow, base, slow, rng, err_op=None):
    """Spans of one trace: [span_id, parent_id, service, operation, start_us, duration_us, status]."""
    spans = []

    def emit(node, parent, start, caller=None):
        if caller is not None:  # an outgoing call: a client span on the caller wraps the callee's server span
            cid = f"{rng.getrandbits(32):08x}"
            lat = rng.randint(150, 900)
            inner = emit(node, cid, start + lat)
            row = [cid, parent, caller, node["op"], start, inner[5] + 2 * lat, "OK"]
            spans.append(row)
            return row
        sid = f"{rng.getrandbits(32):08x}"
        key = (node["svc"], node["op"])
        self_us = int(base[key] * rng.uniform(0.75, 1.25) * (slow.get(key, 1)))
        if key == err_op:
            self_us += rng.randint(2_000_000, 4_000_000)  # a timeout inside an error trace
        pre = self_us // 2
        t, end_children = start + pre, start + pre
        kids = [k for k in node["kids"] for _ in range(k["rep"])]
        rows = []
        if node["mode"] == "par" and kids:
            for k in kids:
                rows.append(emit(k, sid, t + rng.randint(0, 300), node["svc"]))
            end_children = max(r[4] + r[5] for r in rows)
        elif node["mode"] == "async" and kids:
            pass  # consumers start after this producer span ends, see below
        else:
            for k in kids:
                r = emit(k, sid, t + rng.randint(50, 400), node["svc"])
                rows.append(r)
                t = r[4] + r[5]
            end_children = t
        dur = (end_children - start) + (self_us - pre)
        row = [sid, parent, node["svc"], node["op"], start, dur, "OK"]
        spans.append(row)
        if node["mode"] == "async":
            for k in kids:  # async consumers: children of the producer that run after it has finished
                emit(k, sid, start + dur + rng.randint(2_000, 40_000))
        return row

    root = emit(flow, "-", 0)
    spans.sort(key=lambda r: (r[4], r[0]))
    if err_op:
        root[6] = "ERROR"
    return spans


def self_times(spans):
    """Per span: duration minus the union of its direct children's intervals, clipped to the span."""
    kids = {}
    for s in spans:
        kids.setdefault(s[1], []).append((s[4], s[4] + s[5]))
    out = {}
    for s in spans:
        a, b = s[4], s[4] + s[5]
        iv = sorted((max(a, x), min(b, y)) for x, y in kids.get(s[0], []) if min(b, y) > max(a, x))
        covered, cur = 0, None
        for x, y in iv:
            if cur and x <= cur[1]:
                cur[1] = max(cur[1], y)
            else:
                covered += (cur[1] - cur[0]) if cur else 0
                cur = [x, y]
        covered += (cur[1] - cur[0]) if cur else 0
        out[s[0]] = s[5] - covered
    return out


def class_stats(traces, include_errors=False):
    """Mean over the class's traces (non-error unless asked) of each operation's total self time per trace."""
    tot, k = {}, 0
    for spans in traces:
        if not include_errors and any(s[1] == "-" and s[6] == "ERROR" for s in spans):
            continue
        k += 1
        st = self_times(spans)
        for s in spans:
            key = f"{s[2]} {s[3]}"
            tot[key] = tot.get(key, 0) + st[s[0]]
    return {key: v / k for key, v in tot.items()}


def longest_non_root(traces):
    tot = {}
    for spans in traces:
        if any(s[1] == "-" and s[6] == "ERROR" for s in spans):
            continue
        for s in spans:
            if s[1] != "-":
                tot[f"{s[2]} {s[3]}"] = tot.get(f"{s[2]} {s[3]}", 0) + s[5]
    return max(tot.items(), key=lambda x: x[1])[0]


def generate(n, seed, out_dir):
    rng = random.Random(f"56:{seed}:{n}")
    combos = [(f, c, r) for f in FLOWS for c in CLIENTS for r in REGIONS]
    rng.shuffle(combos)
    classes, traces, gold = [], {}, {}
    for i in range(n):
        f, c, r = combos[i % len(combos)]
        cid = f"C{i + 1:04d}"
        label = f"{f} | client={c} | region={r}" + (f" | build={i // len(combos) + 1}" if i >= len(combos) else "")
        flow = FLOWS[f]
        keys = sorted(set(flat_ops(flow)))
        for _ in range(200):  # redraw until the rules give a clear winner and the shortcuts are wrong
            base = {k: rng.randint(400, 9_000) for k in keys}
            root = (flow["svc"], flow["op"])
            target = root if rng.random() < 0.15 else rng.choice([k for k in keys if k != root])
            slow = {target: rng.uniform(3.0, 5.0)}
            n_err = rng.choice([1, 2])
            err_op = rng.choice([k for k in keys if k != target])
            ts = [build_trace(flow, base, slow, rng) for _ in range(TRACES_PER_CLASS - n_err)]
            ts += [build_trace(flow, base, slow, rng, err_op) for _ in range(n_err)]
            rng.shuffle(ts)
            st = sorted(class_stats(ts).items(), key=lambda x: -x[1])
            if st[0][0] != f"{target[0]} {target[1]}" or st[0][1] < 1.4 * st[1][1]:
                continue
            with_err = max(class_stats(ts, True).items(), key=lambda x: x[1])[0]
            if with_err == st[0][0]:
                continue  # excluding the error traces must matter
            break
        else:
            raise RuntimeError("could not plant a clear class")
        tids = []
        for spans in ts:
            tid = f"{rng.getrandbits(64):016x}"
            traces[tid] = spans
            tids.append(tid)
        classes.append({"id": cid, "label": label, "traces": tids})
        gold[cid] = st[0][0]
    prompt = "\n\n".join([T["intro"].replace("{n}", str(n)).replace("{k}", str(TRACES_PER_CLASS)),
                          "\n".join(f"- {x.lstrip('- ')}" for x in T["rules"]),
                          "Tools:\n" + "\n".join(f"- {k}: {v}" for k, v in T["tools"].items()),
                          T["answer"].replace("{example}", json.dumps({"C0001": "cart oteldemo.CartService/GetCart"})),
                          T["score"]])
    return common.finish(TYPE_ID, n, seed, out_dir, prompt, {"answers": gold}, TIERS,
                         files={"classes.json": json.dumps(classes), "traces.json": json.dumps(traces)})


_worlds, _wlock = {}, threading.Lock()


def _load(world_dir):
    with _wlock:
        if world_dir not in _worlds:
            d = Path(world_dir)
            _worlds[world_dir] = (json.loads((d / "classes.json").read_text()),
                                  json.loads((d / "traces.json").read_text()))
        return _worlds[world_dir]


def tools(world_dir):
    def list_classes(a):
        classes, _ = _load(world_dir)
        off, lim = common.page_args(a, 100, 100)
        rows = [f"{c['id']} | {c['label']}" for c in classes[off:off + lim]]
        tail = f"[next: offset={off + lim}]" if off + lim < len(classes) else "[end]"
        return f"{len(classes)} request classes; showing {off}-{off + len(rows) - 1}\n" + "\n".join(rows) + "\n" + tail

    def list_traces(a):
        classes, traces = _load(world_dir)
        cid = str(a.get("class_id", "")).strip().upper()
        c = next((c for c in classes if c["id"] == cid), None)
        if c is None:
            raise KeyError(f"unknown class_id {a.get('class_id')!r}; ids look like C0001")
        rows = []
        for tid in c["traces"]:
            root = next(s for s in traces[tid] if s[1] == "-")
            rows.append(f"{tid} | root status {root[6]} | root duration_us {root[5]} | spans {len(traces[tid])}")
        return f"{cid} {c['label']}\n" + "\n".join(rows)

    def get_trace(a):
        _, traces = _load(world_dir)
        tid = str(a.get("trace_id", "")).strip().lower()
        if tid not in traces:
            raise KeyError(f"unknown trace_id {a.get('trace_id')!r}")
        head = "span_id | parent_id | service | operation | start_us | duration_us | status"
        return f"trace {tid}\n{head}\n" + "\n".join(" | ".join(map(str, s)) for s in traces[tid])

    S, I = {"type": "string"}, {"type": "integer"}
    return [common.tool("list_classes", T["tools"]["list_classes"], {"offset": I, "limit": I}, [], list_classes),
            common.tool("list_traces", T["tools"]["list_traces"], {"class_id": S}, ["class_id"], list_traces),
            common.tool("get_trace", T["tools"]["get_trace"], {"trace_id": S}, ["trace_id"], get_trace)]


CID = re.compile(r"^C\d{4}$")


def _norm(v):
    return re.sub(r"\s+", " ", str(v).replace("`", "").strip())


def parse_answer(answer):
    obj = common.last_json(answer, lambda o: isinstance(o, dict) and any(CID.match(str(k).strip().upper()) for k in o))
    if obj is not None:
        return {str(k).strip().upper(): _norm(v) for k, v in obj.items() if CID.match(str(k).strip().upper())}
    out = {}
    for line in (answer or "").splitlines():  # fallback: "C0001 | cart oteldemo.CartService/GetCart"
        m = re.match(r"^(C\d{4})\s*[|:]\s*(.+)$", common.clean_line(line))
        if m:
            out[m.group(1)] = _norm(m.group(2))
    return out


def score(task, answer):
    gold = common.gold_of(task)["answers"]
    pred = parse_answer(answer)
    ok = sum(pred.get(c) == g for c, g in gold.items())
    return {"score": ok / len(gold), "score_partial": ok / len(gold), "score_exact": float(ok == len(gold)),
            "n_correct": ok, "n_classes": len(gold), "n_answered": len(set(pred) & set(gold))}
