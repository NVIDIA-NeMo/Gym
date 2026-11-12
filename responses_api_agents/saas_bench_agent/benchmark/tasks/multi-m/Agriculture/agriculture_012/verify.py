#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Agriculture/agriculture_012/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""Verify the e-label-to-Grocy causal chain for agriculture_012."""

import html
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request


HOST = os.getenv("SERVER_HOSTNAME", "localhost")
E_LABEL_PORT = os.getenv("E_LABEL_PORT")
E_LABEL_CONTAINER = os.getenv("E_LABEL_CONTAINER")
GROCY_PORT = os.getenv("GROCY_PORT")
GROCY_CONTAINER = os.getenv("GROCY_CONTAINER")

for _name, _value in [
    ("E_LABEL_PORT", E_LABEL_PORT),
    ("E_LABEL_CONTAINER", E_LABEL_CONTAINER),
    ("GROCY_PORT", GROCY_PORT),
    ("GROCY_CONTAINER", GROCY_CONTAINER),
]:
    if not _value:
        print(f"FATAL: {_name} not set", file=sys.stderr)
        sys.exit(1)

BASE_URL = f"http://{HOST}:{E_LABEL_PORT}"
GROCY_DB_CANDIDATES = [
    "/config/data/grocy.db",
    "/config/data/data/grocy.db",
    "/var/www/data/grocy.db",
]
ELABEL_NAME = "Boutique Organic Chardonnay"
GROCY_NAME = "Boutique Organic Chardonnay 2023 - 750 mL"

_checks: list[tuple[str, int, bool, str]] = []


def check(label: str, weight: int, passed: bool, detail: str = "") -> None:
    _checks.append((label, weight, passed, detail))
    status = "PASS" if passed else "FAIL"
    tail = f"  ({detail})" if detail else ""
    print(f"[{status}] ({weight}pt) {label}{tail}", file=sys.stderr)


def docker_exec(container: str, *args: str, timeout: int = 20) -> tuple[int, str, str]:
    result = subprocess.run(
        ["docker", "exec", container, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def _resolve_elabel_db() -> str:
    candidates = [
        os.getenv("E_LABEL_DB_CONTAINER", ""),
        E_LABEL_CONTAINER.replace("-app", "-db") if "-app" in E_LABEL_CONTAINER else "",
        E_LABEL_CONTAINER + "-db",
        E_LABEL_CONTAINER,
        "elabel-db",
    ]
    for candidate in dict.fromkeys(c for c in candidates if c):
        try:
            rc, _, _ = docker_exec(candidate, "echo", "ok", timeout=5)
            if rc == 0:
                return candidate
        except Exception:
            continue
    return E_LABEL_CONTAINER


E_LABEL_DB = _resolve_elabel_db()


def elabel_rows(query: str) -> list[list[str]]:
    last_error = "sqlcmd not found"
    for binary in ["/opt/mssql-tools18/bin/sqlcmd", "/opt/mssql-tools/bin/sqlcmd"]:
        args = [
            binary, "-S", "localhost", "-U", "sa", "-P", "Elabel2024!Strong",
            "-d", "elabel", "-h", "-1", "-W", "-s", "|",
            "-Q", "SET NOCOUNT ON; " + query,
        ]
        if "mssql-tools18" in binary:
            args.insert(9, "-C")
        rc, out, err = docker_exec(E_LABEL_DB, *args)
        if rc == 0:
            return [
                [part.strip() for part in line.strip().split("|")]
                for line in out.splitlines()
                if line.strip() and "rows affected" not in line.lower()
            ]
        last_error = err.strip() or out.strip()
        if "not found" not in last_error.lower():
            break
    raise RuntimeError(last_error)


_grocy_db_path = ""


def _find_grocy_db() -> str:
    global _grocy_db_path
    if _grocy_db_path:
        return _grocy_db_path
    for candidate in GROCY_DB_CANDIDATES:
        rc, _, _ = docker_exec(GROCY_CONTAINER, "test", "-f", candidate, timeout=5)
        if rc == 0:
            _grocy_db_path = candidate
            return candidate
    _grocy_db_path = GROCY_DB_CANDIDATES[0]
    return _grocy_db_path


def grocy_query(sql: str) -> list[dict]:
    php = (
        '$db=new PDO("sqlite:' + _find_grocy_db() + '");'
        "$db->setAttribute(PDO::ATTR_ERRMODE,PDO::ERRMODE_EXCEPTION);"
        "$s=$db->query($argv[1]);"
        "echo json_encode($s->fetchAll(PDO::FETCH_ASSOC));"
    )
    rc, out, err = docker_exec(GROCY_CONTAINER, "php", "-r", php, "--", sql)
    if rc != 0:
        raise RuntimeError(f"Grocy query failed: {err.strip()[:300]}")
    return json.loads(out) if out.strip() else []


def strip_html(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"</p\s*>", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def description_lines(value: str) -> list[str]:
    value = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>", "\n", value or "")
    value = html.unescape(re.sub(r"<[^>]+>", "", value))
    return [line.strip() for line in value.splitlines() if line.strip()]


def http_get(path: str) -> tuple[int, str, str]:
    try:
        with urllib.request.urlopen(BASE_URL + path, timeout=15) as response:
            return (
                response.status,
                response.read().decode("utf-8", errors="replace"),
                response.geturl(),
            )
    except urllib.error.HTTPError as exc:
        return exc.code, "", exc.geturl()
    except Exception:
        return 0, "", ""


def label_code_from_final_url(final_url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(final_url)
        base = urllib.parse.urlparse(BASE_URL)
        same_origin = (
            parsed.scheme.casefold() == base.scheme.casefold()
            and (parsed.hostname or "").casefold() == (base.hostname or "").casefold()
            and parsed.port == base.port
        )
    except (TypeError, ValueError):
        return ""
    if (not same_origin or parsed.username is not None or parsed.password is not None
            or parsed.params or parsed.query or parsed.fragment):
        return ""
    match = re.fullmatch(r"/l/([^/]+)/?", parsed.path)
    if match is None:
        return ""
    code = urllib.parse.unquote(match.group(1)).strip()
    if not code or "/" in code or "\\" in code:
        return ""
    return code


_elabel_rows: list[list[str]] = []
_product: dict[str, str] = {}
_label_code = ""
_public_html = ""
_grocy_rows: list[dict] = []
_core_ok = False
_compliance_ok = False
_public_ok = False
_grocy_link_ok = False


def load_context() -> None:
    global _elabel_rows, _product, _label_code, _public_html, _public_ok, _grocy_rows
    _elabel_rows = elabel_rows(
        "SELECT CAST(Id AS NVARCHAR(36)), Name, CAST(Volume AS NVARCHAR(20)), "
        "CAST(WineVintage AS NVARCHAR(10)), ISNULL(WineAppellation,''), "
        "CAST(WineAlcohol AS NVARCHAR(20)), CAST(WineType AS NVARCHAR(10)), "
        "ISNULL(FBOName,''), CAST(ISNULL(Certifications_Organic,0) AS NVARCHAR(5)), "
        "ISNULL(Sku,''), ISNULL(CAST(Ean AS NVARCHAR(30)),''), "
        "ISNULL(RedirectUrl,''), ISNULL(ExternalShortUrl,'') "
        "FROM Product "
        "WHERE LOWER(LTRIM(RTRIM(Name))) = 'boutique organic chardonnay' "
        "ORDER BY CreatedOn"
    )
    if len(_elabel_rows) == 1 and len(_elabel_rows[0]) >= 13:
        row = _elabel_rows[0]
        _product = {
            "id": row[0], "name": row[1], "volume": row[2], "vintage": row[3],
            "appellation": row[4], "alcohol": row[5], "type": row[6],
            "fbo": row[7], "organic": row[8], "sku": row[9], "ean": row[10],
            "redirect_url": row[11], "external_short_url": row[12],
        }
        locator = urllib.parse.quote(_product["id"], safe="")
        status, body, final_url = http_get(f"/l/{locator}")
        final_code = label_code_from_final_url(final_url)
        if (status == 200 and len(body) > 100
                and final_code == _product["id"]):
            _label_code = final_code
            _public_html = body
            text = strip_html(body)
            _public_ok = (
                ELABEL_NAME in text
                and "2023" in text
                and re.search(r"12[.,]5\s*%?\s*vol", text) is not None
            )

    safe_name = GROCY_NAME.replace("'", "''")
    _grocy_rows = grocy_query(
        f"SELECT id, name, description FROM products WHERE name = '{safe_name}' ORDER BY id"
    )


def check_1_elabel_core_fields() -> None:
    global _core_ok
    problems = []
    if len(_elabel_rows) != 1 or not _product:
        problems.append(f"expected one exact e-label product, found {len(_elabel_rows)}")
    else:
        if _product.get("name") != ELABEL_NAME:
            problems.append(f"name='{_product.get('name', '')}'")
        try:
            volume = float(_product["volume"])
            if not (abs(volume - 0.75) < 0.01 or abs(volume - 750) < 1):
                problems.append(f"volume={volume}")
            if int(_product["vintage"]) != 2023:
                problems.append(f"vintage={_product['vintage']}")
            if abs(float(_product["alcohol"]) - 12.5) >= 0.05:
                problems.append(f"alcohol={_product['alcohol']}")
            if int(_product["type"]) != 1:
                problems.append(f"WineType={_product['type']} (expected White=1)")
        except (TypeError, ValueError) as exc:
            problems.append(f"numeric field error: {exc}")
        if _product.get("appellation", "").strip().casefold() != "loire":
            problems.append(f"appellation='{_product.get('appellation', '')}'")
        for field in ("sku", "ean", "redirect_url", "external_short_url"):
            if _product.get(field, "").strip():
                problems.append(f"{field} must be blank")
    _core_ok = not problems
    check("1. exact Chardonnay core fields", 3, _core_ok, "; ".join(problems))


def check_2_exact_compliance() -> None:
    global _compliance_ok
    if not _core_ok:
        check("2. exact organic compliance", 2, False, "gated: core product invalid")
        return
    try:
        ingredients = elabel_rows(
            "SELECT i.Name, CAST(ISNULL(i.Allergen,0) AS NVARCHAR(5)) "
            "FROM ProductIngredient pi JOIN Ingredient i ON i.Id = pi.IngredientId "
            f"WHERE pi.ProductId = '{_product['id']}' ORDER BY i.Name"
        )
    except Exception as exc:
        _compliance_ok = False
        check("2. exact organic compliance", 2, False,
              f"compliance read failed: {exc}")
        return
    names = [row[0].strip() for row in ingredients if row]
    sulphites = [
        row for row in ingredients
        if row and row[0].strip().casefold() == "sulphites"
        and len(row) > 1 and row[1] in ("1", "True", "true")
    ]
    _compliance_ok = (
        _product.get("fbo", "").strip() == "Boutique Organic Farm"
        and _product.get("organic") in ("1", "True", "true")
        and len(ingredients) == 1
        and len(sulphites) == 1
    )
    check(
        "2. exact organic compliance",
        2,
        _compliance_ok,
        f"FBO='{_product.get('fbo', '')}', organic={_product.get('organic')}, ingredients={names}",
    )


def check_3_public_label() -> None:
    if not (_core_ok and _compliance_ok):
        check("3. public Chardonnay label", 5, False, "gated: core or compliance invalid")
        return
    check(
        "3. public Chardonnay label",
        5,
        _public_ok,
        f"code='{_label_code}'" if _public_ok
        else "public /l/ page missing exact name, vintage, or 12.5 % vol",
    )


def check_4_grocy_generated_code_link() -> None:
    global _grocy_link_ok
    if not (_core_ok and _compliance_ok and _public_ok):
        check("4. Grocy links generated e-label code", 6, False,
              "gated: complete compliant public label unavailable")
        return
    if len(_grocy_rows) != 1:
        check(
            "4. Grocy links generated e-label code",
            6,
            False,
            f"expected one exact Grocy product, found {len(_grocy_rows)}",
        )
        return
    lines = description_lines(_grocy_rows[0].get("description") or "")
    _grocy_link_ok = (
        len(lines) == 3
        and lines[0] == f"e-label code: {_label_code}"
        and lines[1].startswith("125 mL servings per bottle:")
        and lines[2] == "Producer: Boutique Organic Farm"
    )
    check(
        "4. Grocy links generated e-label code",
        6,
        _grocy_link_ok,
        f"lines={lines}",
    )


def check_5_calculation_and_exclusive_trace() -> None:
    if not (_core_ok and _compliance_ok and _public_ok and _grocy_link_ok):
        check("5. calculated servings and exclusive trace", 4, False,
              "gated: complete public/Grocy chain missing")
        return
    safe_code = _label_code.replace("'", "''")
    target_id = int(_grocy_rows[0]["id"])
    try:
        other_rows = grocy_query(
            "SELECT id, name FROM products "
            f"WHERE description LIKE '%{safe_code}%' AND id <> {target_id}"
        )
    except Exception as exc:
        check("5. calculated servings and exclusive trace", 4, False,
              f"exclusive-trace read failed: {exc}")
        return
    lines = description_lines(_grocy_rows[0].get("description") or "")
    serving_ok = len(lines) == 3 and lines[1] == "125 mL servings per bottle: 6"
    passed = serving_ok and not other_rows
    check(
        "5. calculated servings and exclusive trace",
        4,
        passed,
        f"serving_line='{lines[1] if len(lines) > 1 else ''}', "
        f"other products carrying code={len(other_rows)}",
    )


def main() -> None:
    try:
        load_context()
    except Exception as exc:
        print(f"WARNING: context load failed: {exc}", file=sys.stderr)
    check_1_elabel_core_fields()
    check_2_exact_compliance()
    check_3_public_label()
    check_4_grocy_generated_code_link()
    check_5_calculation_and_exclusive_trace()

    total = sum(weight for _, weight, _, _ in _checks)
    earned = sum(weight for _, weight, passed, _ in _checks if passed)
    all_pass = bool(_checks) and all(passed for _, _, passed, _ in _checks)
    score = earned / total if total else 0.0
    print(f"SCORE: {score:.3f}  PASS: {all_pass}  ({earned}/{total})", file=sys.stderr)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
