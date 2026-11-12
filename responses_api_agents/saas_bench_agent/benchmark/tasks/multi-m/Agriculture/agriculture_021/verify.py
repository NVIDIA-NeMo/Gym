# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Agriculture/agriculture_021/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""Verifier for agriculture_021: exact FarmOS Harvest/Grocy manifest audit.

The verifier anchors both sides of each manifest row to real seed identities:
FarmOS log type plus exact name, and Grocy exact name plus barcode. Source
integrity is a gate, not a scored existence check.
"""

import html
import json
import os
import re
import subprocess
import sys

FARMOS_PORT = os.getenv("FARMOS_PORT")
FARMOS_CONTAINER = os.getenv("FARMOS_CONTAINER")
GROCY_PORT = os.getenv("GROCY_PORT")
GROCY_CONTAINER = os.getenv("GROCY_CONTAINER")

for _var_name, _var_val in [
    ("FARMOS_PORT", FARMOS_PORT),
    ("FARMOS_CONTAINER", FARMOS_CONTAINER),
    ("GROCY_PORT", GROCY_PORT),
    ("GROCY_CONTAINER", GROCY_CONTAINER),
]:
    if not _var_val:
        print(f"FATAL: {_var_name} not set", file=sys.stderr)
        sys.exit(1)

FARMOS_SQLITE = "/opt/drupal/web/sites/default/files/.ht.sqlite"
GROCY_DB_CANDIDATES = [
    "/config/data/grocy.db",
    "/config/data/data/grocy.db",
    "/var/www/data/grocy.db",
]

MARKER = "AUDIT FLAG: Missing FarmOS harvest log"
MANIFEST = {
    "Green Leaf Lettuce": {
        "barcode": "0000651041025", "batch": "Crop Scouting Report", "matched": True,
    },
    "Whole Kernel Corn": {
        "barcode": "00016056", "batch": "Fall Harvest - Corn", "matched": False,
    },
    "Spring onion": {
        "barcode": "00001373", "batch": "Soybean Planting Activity", "matched": True,
    },
    "Spring onions": {
        "barcode": "00008761", "batch": "Corn Field Inspection - East Plot", "matched": False,
    },
    "Peach": {
        "barcode": "00002523", "batch": "Spring Plowing Complete", "matched": True,
    },
    "Large flat mushrooms": {
        "barcode": "00019170", "batch": "Wheat Harvest Report", "matched": False,
    },
    "British Cox Apples": {
        "barcode": "00035309", "batch": "Water Quality Sampling", "matched": True,
    },
    "Iceberg Lettuce": {
        "barcode": "00040617", "batch": "Fall Harvest - Soybeans", "matched": False,
    },
}

_checks: list[tuple[str, int, bool, str]] = []


def check(label: str, weight: int, passed: bool, detail: str = "") -> None:
    _checks.append((label, weight, passed, detail))
    status = "PASS" if passed else "FAIL"
    tail = f"  ({detail})" if detail else ""
    print(f"[{status}] ({weight}pt) {label}{tail}", file=sys.stderr)


def docker_exec(container: str, *args: str, timeout: int = 15) -> tuple[int, str, str]:
    result = subprocess.run(
        ["docker", "exec", container, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


_grocy_db_path = ""


def _find_grocy_db() -> str:
    global _grocy_db_path
    if _grocy_db_path:
        return _grocy_db_path
    for path in GROCY_DB_CANDIDATES:
        rc, _, _ = docker_exec(GROCY_CONTAINER, "test", "-f", path)
        if rc == 0:
            _grocy_db_path = path
            return path
    _grocy_db_path = GROCY_DB_CANDIDATES[0]
    return _grocy_db_path


def grocy_sql_json(query: str) -> list[dict]:
    db = _find_grocy_db()
    php_script = (
        '$db = new PDO("sqlite:' + db + '");'
        '$db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);'
        '$r = $db->query(' + json.dumps(query) + ');'
        '$rows = $r->fetchAll(PDO::FETCH_ASSOC);'
        'echo json_encode($rows);'
    )
    rc, stdout, stderr = docker_exec(GROCY_CONTAINER, "php", "-r", php_script)
    if rc != 0:
        raise RuntimeError(f"grocy php error (rc={rc}): {stderr.strip()}")
    return json.loads(stdout) if stdout.strip() else []


def farmos_sql_json(query: str) -> list[dict]:
    php_script = (
        '$db = new PDO("sqlite:' + FARMOS_SQLITE + '");'
        '$db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);'
        '$r = $db->query(' + json.dumps(query) + ');'
        '$rows = $r->fetchAll(PDO::FETCH_ASSOC);'
        'echo json_encode($rows);'
    )
    rc, stdout, stderr = docker_exec(FARMOS_CONTAINER, "php", "-r", php_script)
    if rc != 0:
        raise RuntimeError(f"farmos php error (rc={rc}): {stderr.strip()}")
    return json.loads(stdout) if stdout.strip() else []


def _sql_list(values: list[str]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


def _normalize(text: str) -> str:
    visible = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    return re.sub(r"\s+", " ", visible).strip()


def _annotation(batch: str) -> str:
    return f"{MARKER} | batch={batch}"


_product_rows: list[dict] | None = None
_harvest_names: set[str] | None = None
_manifest_ok = False


def _load_product_rows() -> list[dict]:
    global _product_rows
    if _product_rows is None:
        barcodes = [spec["barcode"] for spec in MANIFEST.values()]
        _product_rows = grocy_sql_json(
            "SELECT p.id, p.name, COALESCE(p.description, '') AS description, "
            "pb.barcode FROM products p "
            "JOIN product_barcodes pb ON pb.product_id = p.id "
            f"WHERE pb.barcode IN ({_sql_list(barcodes)})"
        )
    return _product_rows


def _load_harvest_names() -> set[str]:
    global _harvest_names
    if _harvest_names is None:
        rows = farmos_sql_json("SELECT name FROM log_field_data WHERE type = 'harvest'")
        _harvest_names = {str(row["name"]) for row in rows if row.get("name") is not None}
    return _harvest_names


def _candidate_rows(name: str, barcode: str) -> list[dict]:
    return [
        row for row in _load_product_rows()
        if row.get("name") == name and str(row.get("barcode", "")) == barcode
    ]


def _source_errors() -> list[str]:
    errors: list[str] = []
    harvest_names = _load_harvest_names()
    for name, spec in MANIFEST.items():
        rows = _candidate_rows(name, spec["barcode"])
        if len(rows) != 1:
            errors.append(f"{name}: expected one exact name/barcode row, found {len(rows)}")
        exists = spec["batch"] in harvest_names
        if exists != spec["matched"]:
            state = "present" if exists else "absent"
            errors.append(f"FarmOS source changed: {spec['batch']} is {state} in Harvest")
    return errors


def check_1_exact_manifest_reconciliation() -> None:
    global _manifest_ok
    try:
        errors = _source_errors()
        if errors:
            check("1. exact_manifest_reconciliation", 8, False, "; ".join(errors))
            return
        problems = []
        for name, spec in MANIFEST.items():
            row = _candidate_rows(name, spec["barcode"])[0]
            expected = "" if spec["matched"] else _annotation(spec["batch"])
            actual = _normalize(row["description"])
            if actual != expected:
                problems.append(f"{name}: description={actual!r}, expected={expected!r}")
        _manifest_ok = not problems
        check(
            "1. exact_manifest_reconciliation",
            8,
            _manifest_ok,
            "all matched rows unchanged and unmatched rows carry their exact batch evidence"
            if not problems else "; ".join(problems),
        )
    except Exception as exc:
        check("1. exact_manifest_reconciliation", 8, False, f"exception: {exc}")


def check_2_no_spurious_audit_flags() -> None:
    if not _manifest_ok:
        check("2. no_spurious_audit_flags", 4, False,
              "gated: exact manifest reconciliation is incomplete")
        return
    try:
        rows = grocy_sql_json(
            "SELECT p.id, p.name, COALESCE(p.description, '') AS description, pb.barcode "
            "FROM products p LEFT JOIN product_barcodes pb ON pb.product_id = p.id "
            "WHERE p.description LIKE '%AUDIT FLAG: Missing FarmOS harvest log%' "
            "ORDER BY p.id, pb.barcode"
        )
        signaled: dict[int, dict] = {}
        for row in rows:
            product_id = int(row["id"])
            product = signaled.setdefault(
                product_id,
                {
                    "name": str(row.get("name", "")),
                    "description": _normalize(str(row.get("description", ""))),
                    "barcodes": set(),
                },
            )
            barcode = row.get("barcode")
            if barcode not in (None, ""):
                product["barcodes"].add(str(barcode))

        expected: dict[int, tuple[str, str, str]] = {}
        for name, spec in MANIFEST.items():
            if spec["matched"]:
                continue
            candidate = _candidate_rows(name, spec["barcode"])[0]
            expected[int(candidate["id"])] = (
                name,
                spec["barcode"],
                _annotation(spec["batch"]),
            )

        problems = []
        unexpected_ids = sorted(set(signaled) - set(expected))
        missing_ids = sorted(set(expected) - set(signaled))
        if unexpected_ids:
            details = [
                f"#{product_id} {signaled[product_id]['name']!r} "
                f"barcodes={sorted(signaled[product_id]['barcodes'])}"
                for product_id in unexpected_ids
            ]
            problems.append("unexpected signaled products: " + ", ".join(details))
        if missing_ids:
            problems.append(f"expected signaled product IDs missing: {missing_ids}")
        for product_id in sorted(set(signaled) & set(expected)):
            expected_name, expected_barcode, expected_description = expected[product_id]
            product = signaled[product_id]
            if product["name"] != expected_name:
                problems.append(
                    f"product #{product_id} name={product['name']!r}, expected={expected_name!r}"
                )
            if product["description"] != expected_description:
                problems.append(f"product #{product_id} has inexact audit description")
            if expected_barcode not in product["barcodes"]:
                problems.append(
                    f"product #{product_id} lacks manifest barcode {expected_barcode!r}"
                )
        check(
            "2. no_spurious_audit_flags",
            4,
            not problems,
            "; ".join(problems)
            if problems else "only exact unmatched manifest products carry the task audit marker",
        )
    except Exception as exc:
        check("2. no_spurious_audit_flags", 4, False, f"exception: {exc}")


def main() -> None:
    check_1_exact_manifest_reconciliation()
    check_2_no_spurious_audit_flags()

    total = sum(weight for _, weight, _, _ in _checks)
    earned = sum(weight for _, weight, passed, _ in _checks if passed)
    all_pass = bool(_checks) and all(passed for _, _, passed, _ in _checks)
    score = earned / total if total else 0.0
    print(f"SCORE: {score:.3f}  PASS: {all_pass}  ({earned}/{total})", file=sys.stderr)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
