"""Baselines for type 19. Each takes a generated task and rewrites the unit files in its world."""
import ast
import json
import re
from pathlib import Path

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("b19_type", Path(__file__).with_name("type.py"))
T = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(T)
eng, R = T.eng, T.R


def units(task):
    return json.loads((Path(task["gold_path"]) / "meta.json").read_text())["units"]


def oracle(task):
    eng.oracle_fill(task)


def catalog_msgids(world):
    """msgids of the click .pot and the humanize .po catalogs."""
    ids = set()
    texts = [(world / "repos/click/messages.pot").read_text()]
    texts += [p.read_text() for p in (world / "repos/humanize/src/humanize/locale").rglob("*.po")][:1]
    for t in texts:
        for m in re.finditer(r'^msgid ("(?:[^"\\]|\\.)*")', t, re.M):
            v = ast.literal_eval(m.group(1))
            if v:
                ids.add(v)
    return ids


def regex(task):
    """sed-style: wrap every string literal that equals a catalog msgid in _( ... )."""
    w = Path(task["world_dir"])
    ids = catalog_msgids(w)
    for rel in sorted({u["files"][0] for u in units(task)}):
        src = (w / rel).read_text()
        for mid in sorted(ids, key=len, reverse=True):
            for q in ('"', "'"):
                lit = q + mid.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t") + q
                src = re.sub(r"(?<![\w(])" + re.escape(lit), "_(" + lit + ")", src)
        (w / rel).write_text(src)


def after_format(task):
    """Translate after formatting: _(f"...") and ngettext over f-strings (msgids are not literals)."""
    w = Path(task["world_dir"])
    regex(task)
    for rel in sorted({u["files"][0] for u in units(task)}):
        src = (w / rel).read_text()
        src = re.sub(r'(?<![\w(])(f"[^"\n]*")', r"_(\1)", src)
        src = re.sub(r"\((_\(f\"[^\"\n]*\"\)|f\"[^\"\n]*\"|\"[^\"\n]*\") if (.+?) == 1 else (_\(f\"[^\"\n]*\"\)|f\"[^\"\n]*\"|\"[^\"\n]*\")\)",
                     lambda m: f"ngettext({m[1]}, {m[3]}, {m[2]})", src)
        (w / rel).write_text(src)


def partial(task):
    """Oracle, except the first translation call of each unit function stays hard-coded."""
    cat = {u["id"]: u for u in T.catalog()["units"]}
    w, g = Path(task["world_dir"]), Path(task["gold_path"]) / "base"
    by_file = {}
    for u in units(task):
        by_file.setdefault(u["files"][0], []).append(cat[u["id"]]["edits"][0])
    for rel, eds in by_file.items():
        (w / rel).write_text(R.apply((g / rel).read_bytes(), [(s, e, r, None) for s, e, r in eds]))


BASELINES = {"oracle": oracle, "regex": regex, "after_format": after_format, "partial": partial}
