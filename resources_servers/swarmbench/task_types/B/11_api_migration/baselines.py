"""Baselines for type 11. Each takes a generated task and rewrites the unit files in its world."""
import json
import re
from pathlib import Path

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("b11_type", Path(__file__).with_name("type.py"))
T = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(T)
eng = T.eng


def units(task):
    return json.loads((Path(task["gold_path"]) / "meta.json").read_text())["units"]


def oracle(task):
    eng.oracle_fill(task)


ADD = re.compile(r"(?P<recv>[\w.]+)\.add\(\s*(?P<name>[^,]+?),\s*(?P<item>.+?),\s*['\"](?P<loc>[^'\"]+)['\"]\s*\)")
SET = re.compile(r"^(?P<ind>\s*)(?P<recv>[\w.]+)\[(?P<name>[^\]]+)\]\s*=\s*(?P<item>.+)$", re.M)
DEL = re.compile(r"del (?P<recv>[\w.]+)\[(?P<name>[^\]]+)\]")


def regex_text(src):
    """What a sed script would do: textual renames plus a fixed priority per location kind."""
    src = re.sub(r"from (markdown|\.\.)\.?util import etree", "import xml.etree.ElementTree as etree", src)
    src = re.sub(r",\s*md_globals", "", src)
    src = src.replace("self.markdown", "self.md").replace("util.isBlockLevel(", "self.md.is_block_level(")
    prio = {"_begin": "200", "_end": "1"}
    src = ADD.sub(lambda m: f"{m['recv']}.register({m['item']}, {m['name']}, {prio.get(m['loc'], '50')})", src)
    src = SET.sub(lambda m: f"{m['ind']}{m['recv']}.register({m['item']}, {m['name']}, 50)"
                  if m['recv'].endswith(tuple(T.REG_ATTRS)) else m[0], src)
    return DEL.sub(lambda m: f"{m['recv']}.deregister({m['name']})", src)


def regex(task):
    w = Path(task["world_dir"])
    for u in units(task):
        p = w / u["id"]
        p.write_text(regex_text(p.read_text()))


def partial(task):
    """Oracle everywhere except the first registry site of each module, which stays as written."""
    cat = {u["id"]: u for u in T.catalog()["units"]}
    w, g = Path(task["world_dir"]), Path(task["gold_path"]) / "base"
    for u in units(task):
        before = (w / u["id"]).read_bytes()
        eds = cat[u["id"]]["edits"]
        first = next(i for i, e in enumerate(eds) if e["kind"] == "reg-add")
        # rebuild: oracle text with only that one edit applied (legacy form, as in the world)
        src = (g / u["id"]).read_bytes()
        e = eds[first]
        cur = before.decode()
        opt = next((o for o in e["options"] if o in cur), e["options"][0])
        (w / u["id"]).write_text((src[:e["start"]] + opt.encode() + src[e["end"]:]).decode())


def delete(task):
    """Drop every registry call in the legacy modules (a 'remove what breaks' fix)."""
    w = Path(task["world_dir"])
    for u in units(task):
        p = w / u["id"]
        lines = []
        for line in regex_text(p.read_text()).splitlines():
            if re.search(r"\.(register|deregister)\(", line) and line.strip().endswith(")"):
                line = line[:len(line) - len(line.lstrip())] + "pass"
            lines.append(line)
        p.write_text("\n".join(lines) + "\n")


BASELINES = {"oracle": oracle, "regex": regex, "partial": partial, "delete": delete}
