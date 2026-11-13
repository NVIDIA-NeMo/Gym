#!/usr/bin/env python3
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/UniPat-AI/SaaS-Bench/blob/48c22deb18b98c0ed78f81e7f3be82bc162de2c8/tasks/multi-m/Agriculture/agriculture_016/verify.py
# Original license: CC-BY-4.0; Copyright: SaaS-Bench Authors; see bundled licenses.
# NVIDIA modifications: Register Drupal Unicode NOCASE_UTF8 and route PHP failures to stderr; grading SQL unchanged.
# Adaptation: register the shipped Drupal collation for database reads; task checks unchanged.
"""Verifier for agriculture_016: FarmOS application chain -> Grocy deficit."""

import base64
import datetime as dt
import json
import os
import re
import subprocess
import sys


FARMOS_CONTAINER = os.getenv("FARMOS_CONTAINER")
GROCY_CONTAINER = os.getenv("GROCY_CONTAINER")
FARMOS_DB = "/opt/drupal/web/sites/default/files/.ht.sqlite"
GROCY_DB = "/config/data/grocy.db"

INPUT_NAME = "Neem Oil Application — Certified Garlic Plot"
MAINTENANCE_NAME = "Post-Application Rinse — Backpack Sprayer #2"
PLANT_NAME = "Certified Garlic Plot"
EQUIPMENT_NAME = "Backpack Sprayer #2"
PRODUCT_NAME = "Neem Oil Concentrate"

for name, value in [
    ("FARMOS_CONTAINER", FARMOS_CONTAINER),
    ("GROCY_CONTAINER", GROCY_CONTAINER),
]:
    if not value:
        print(f"FATAL: {name} not set", file=sys.stderr)
        sys.exit(1)


_checks: list[tuple[str, int, bool, str]] = []
_input_log: dict | None = None
_maintenance_log: dict | None = None
_product: dict | None = None
_related_shopping_rows: list[dict] = []
_input_ok = False
_farmos_chain_ok = False
_grocy_amount_ok = False
_required_amount: float | None = None


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


def farmos_query(sql: str) -> list[dict]:
    encoded = base64.b64encode(sql.encode()).decode()
    php = (
        "require '/opt/drupal/vendor/autoload.php';"
        "$sql=base64_decode('" + encoded + "');"
        "$db=new SQLite3('" + FARMOS_DB + "');"
        "$db->createCollation('NOCASE_UTF8',['Drupal\\Component\\Utility\\Unicode','strcasecmp']);"
        "$result=$db->query($sql);"
        "if($result===false){fwrite(STDERR,$db->lastErrorMsg());exit(2);}"
        "$rows=[];while($row=$result->fetchArray(SQLITE3_ASSOC)){$rows[]=$row;}"
        "echo json_encode($rows);"
    )
    rc, stdout, stderr = docker_exec(FARMOS_CONTAINER, "php", "-d", "display_errors=stderr", "-r", php)
    if rc != 0:
        raise RuntimeError(f"FarmOS query failed: {stderr.strip()}")
    return json.loads(stdout) if stdout.strip() else []


def grocy_query(sql: str) -> list[dict]:
    php = (
        '$db=new PDO("sqlite:' + GROCY_DB + '");'
        "$db->setAttribute(PDO::ATTR_ERRMODE,PDO::ERRMODE_EXCEPTION);"
        "$rows=$db->query(" + json.dumps(sql) + ")->fetchAll(PDO::FETCH_ASSOC);"
        "echo json_encode($rows);"
    )
    rc, stdout, stderr = docker_exec(GROCY_CONTAINER, "php", "-r", php)
    if rc != 0:
        raise RuntimeError(f"Grocy query failed: {stderr.strip()}")
    return json.loads(stdout) if stdout.strip() else []


def strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", text).strip()


def is_today(value: object) -> bool:
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return False
    today_local = dt.datetime.now().date()
    today_utc = dt.datetime.now(dt.timezone.utc).date()
    return (
        dt.datetime.fromtimestamp(timestamp).date() == today_local
        or dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).date() == today_utc
    )


def check_1_complete_input_log() -> None:
    global _input_log, _input_ok, _required_amount
    try:
        plant_rows = farmos_query(
            "SELECT id FROM asset_field_data "
            "WHERE type='plant' AND name='Certified Garlic Plot' ORDER BY id"
        )
        equipment_rows = farmos_query(
            "SELECT id FROM asset_field_data "
            "WHERE type='equipment' AND name='Backpack Sprayer #2' ORDER BY id"
        )
        log_rows = farmos_query(
            "SELECT id,name,timestamp,notes__value FROM log_field_data "
            "WHERE type='input' "
            "AND name COLLATE BINARY='Neem Oil Application — Certified Garlic Plot' "
            "ORDER BY id"
        )
        problems = []
        if len(plant_rows) != 1:
            problems.append(f"expected one exact plant asset, found {len(plant_rows)}")
        if len(equipment_rows) != 1:
            problems.append(f"expected one exact equipment asset, found {len(equipment_rows)}")
        if len(log_rows) != 1:
            problems.append(f"expected one exact Input log, found {len(log_rows)}")
        if problems:
            check("1. complete exact FarmOS Input log", 4, False, "; ".join(problems))
            return

        candidate = log_rows[0]
        log_id = int(candidate["id"])
        plant_id = int(plant_rows[0]["id"])
        equipment_id = int(equipment_rows[0]["id"])
        asset_links = farmos_query(
            "SELECT asset_target_id FROM log__asset "
            f"WHERE entity_id={log_id} AND deleted=0 ORDER BY asset_target_id"
        )
        equipment_links = farmos_query(
            "SELECT equipment_target_id FROM log__equipment "
            f"WHERE entity_id={log_id} AND deleted=0 ORDER BY equipment_target_id"
        )
        linked_assets = [int(row["asset_target_id"]) for row in asset_links]
        linked_equipment = [int(row["equipment_target_id"]) for row in equipment_links]
        if linked_assets != [plant_id]:
            problems.append(f"asset links={linked_assets}, expected [{plant_id}]")
        if linked_equipment != [equipment_id]:
            problems.append(f"equipment links={linked_equipment}, expected [{equipment_id}]")
        if not is_today(candidate.get("timestamp")):
            problems.append("Input log is not dated today")

        notes = strip_html(candidate.get("notes__value") or "")
        required_fragments = [
            "Neem Oil",
            "OMRI-2024-NO-007",
            "150 mL/acre",
            "4 acres",
            "applied during cooler morning hours to avoid leaf burn",
        ]
        missing = [fragment for fragment in required_fragments if fragment not in notes]
        if missing:
            problems.append("missing exact note fragments: " + ", ".join(missing))
        rate_matches = re.findall(r"\b(\d+(?:\.\d+)?)\s*mL\s*/\s*acre\b", notes)
        area_matches = re.findall(r"\b(\d+(?:\.\d+)?)\s*acres?\b", notes)
        if len(rate_matches) != 1 or len(area_matches) != 1:
            problems.append("expected one unambiguous mL/acre rate and one acreage value")

        if not problems:
            _input_log = candidate
            _input_ok = True
            _required_amount = float(rate_matches[0]) * float(area_matches[0])

        check(
            "1. complete exact FarmOS Input log",
            4,
            not problems,
            f"log_id={log_id}" if not problems else "; ".join(problems),
        )
    except Exception as exc:
        check("1. complete exact FarmOS Input log", 4, False, f"exception: {exc}")


def check_2_ordered_maintenance_log() -> None:
    global _maintenance_log, _farmos_chain_ok
    if not _input_ok or not _input_log:
        check("2. exact ordered FarmOS Maintenance log", 4, False, "gated: Input log is invalid")
        return
    try:
        equipment_rows = farmos_query(
            "SELECT id FROM asset_field_data "
            "WHERE type='equipment' AND name='Backpack Sprayer #2' ORDER BY id"
        )
        log_rows = farmos_query(
            "SELECT id,name,timestamp,notes__value FROM log_field_data "
            "WHERE type='maintenance' "
            "AND name COLLATE BINARY='Post-Application Rinse — Backpack Sprayer #2' "
            "ORDER BY id"
        )
        problems = []
        if len(equipment_rows) != 1:
            problems.append(f"expected one exact equipment asset, found {len(equipment_rows)}")
        if len(log_rows) != 1:
            problems.append(f"expected one exact Maintenance log, found {len(log_rows)}")
        if problems:
            check("2. exact ordered FarmOS Maintenance log", 4, False, "; ".join(problems))
            return

        _maintenance_log = log_rows[0]
        log_id = int(_maintenance_log["id"])
        equipment_id = int(equipment_rows[0]["id"])
        links = farmos_query(
            "SELECT asset_target_id FROM log__asset "
            f"WHERE entity_id={log_id} AND deleted=0 ORDER BY asset_target_id"
        )
        linked_assets = [int(row["asset_target_id"]) for row in links]
        if linked_assets != [equipment_id]:
            problems.append(f"asset links={linked_assets}, expected [{equipment_id}]")
        if not is_today(_maintenance_log.get("timestamp")):
            problems.append("Maintenance log is not dated today")
        try:
            if int(_maintenance_log["timestamp"]) < int(_input_log["timestamp"]):
                problems.append("Maintenance timestamp precedes Input timestamp")
        except (TypeError, ValueError, KeyError):
            problems.append("invalid log timestamps")
        notes = strip_html(_maintenance_log.get("notes__value") or "")
        if "triple-rinse clean with clean water" not in notes:
            problems.append("missing exact rinse phrase in notes")

        _farmos_chain_ok = not problems
        check(
            "2. exact ordered FarmOS Maintenance log",
            4,
            _farmos_chain_ok,
            f"log_id={log_id}" if _farmos_chain_ok else "; ".join(problems),
        )
    except Exception as exc:
        check("2. exact ordered FarmOS Maintenance log", 4, False, f"exception: {exc}")


def load_related_shopping_rows() -> list[dict]:
    return grocy_query(
        "SELECT sl.id,sl.product_id,sl.amount,COALESCE(sl.note,'') AS note "
        "FROM shopping_list sl "
        "WHERE COALESCE(sl.note,'') LIKE '%FarmOS input log%' "
        "OR COALESCE(sl.note,'') LIKE '%Certified Garlic Plot%' "
        "ORDER BY sl.id"
    )


def check_3_exact_conditional_grocy_amount() -> None:
    global _product, _related_shopping_rows, _grocy_amount_ok
    if not _farmos_chain_ok or not _input_log or _required_amount is None:
        check("3. exact conditional Grocy product and deficit", 2, False,
              "gated: complete FarmOS chain is invalid")
        return
    try:
        products = grocy_query(
            "SELECT p.id,p.name,COALESCE(qu.name,'') AS unit_name,"
            "COALESCE(qu.name_plural,'') AS unit_name_plural "
            "FROM products p LEFT JOIN quantity_units qu ON qu.id=p.qu_id_stock "
            "WHERE p.name='Neem Oil Concentrate' ORDER BY p.id"
        )
        if len(products) != 1:
            check("3. exact conditional Grocy product and deficit", 2, False,
                  f"expected one exact product, found {len(products)}")
            return
        _product = products[0]
        product_id = int(_product["id"])
        units = {
            re.sub(r"[^a-z]", "", (_product.get("unit_name") or "").lower()),
            re.sub(r"[^a-z]", "", (_product.get("unit_name_plural") or "").lower()),
        }
        valid_units = {"ml", "milliliter", "milliliters", "millilitre", "millilitres"}
        if not units & valid_units:
            check("3. exact conditional Grocy product and deficit", 2, False,
                  f"stock unit must be milliliters, got {sorted(units)}")
            return
        stock_rows = grocy_query(
            f"SELECT COALESCE(SUM(amount),0) AS amount FROM stock WHERE product_id={product_id}"
        )
        stock = float(stock_rows[0]["amount"] or 0) if stock_rows else 0.0
        deficit = max(0.0, _required_amount - stock)
        _related_shopping_rows = load_related_shopping_rows()
        problems = []
        if deficit > 0:
            if len(_related_shopping_rows) != 1:
                problems.append(
                    f"expected one task-related entry, found {len(_related_shopping_rows)}"
                )
            else:
                row = _related_shopping_rows[0]
                if int(row.get("product_id") or 0) != product_id:
                    problems.append(f"entry uses product_id={row.get('product_id')}, expected {product_id}")
                try:
                    amount = float(row.get("amount") or 0)
                except (TypeError, ValueError):
                    amount = 0.0
                if abs(amount - deficit) > 1e-6:
                    problems.append(f"entry amount={amount:g}, expected deficit={deficit:g}")
        elif _related_shopping_rows:
            problems.append(
                f"stock={stock:g} requires no entry, found {len(_related_shopping_rows)}"
            )
        _grocy_amount_ok = not problems
        check(
            "3. exact conditional Grocy product and deficit",
            2,
            _grocy_amount_ok,
            f"stock={stock:g}, deficit={deficit:g}" if not problems else "; ".join(problems),
        )
    except Exception as exc:
        check("3. exact conditional Grocy product and deficit", 2, False, f"exception: {exc}")


def check_4_exact_log_traceability() -> None:
    if (not _farmos_chain_ok or not _input_log or not _product
            or not _grocy_amount_ok or _required_amount is None):
        check("4. shopping entry has exact upstream log traceability", 10, False,
              "gated: FarmOS chain or exact Grocy amount is invalid")
        return
    try:
        product_id = int(_product["id"])
        stock_rows = grocy_query(
            f"SELECT COALESCE(SUM(amount),0) AS amount FROM stock WHERE product_id={product_id}"
        )
        stock = float(stock_rows[0]["amount"] or 0) if stock_rows else 0.0
        deficit = max(0.0, _required_amount - stock)
        rows = _related_shopping_rows or load_related_shopping_rows()
        exact_note = f"FarmOS input log {int(_input_log['id'])} — Certified Garlic Plot"
        problems = []
        if deficit > 0:
            if len(rows) != 1:
                problems.append(f"expected one traceability entry, found {len(rows)}")
            elif rows[0].get("note") != exact_note:
                problems.append(
                    f"note='{rows[0].get('note')}', expected exact numeric log ID {int(_input_log['id'])}"
                )
        elif rows:
            problems.append("task traceability entry exists despite zero deficit")
        check(
            "4. shopping entry has exact upstream log traceability",
            10,
            not problems,
            exact_note if not problems and deficit > 0 else (
                "no entry required" if not problems else "; ".join(problems)
            ),
        )
    except Exception as exc:
        check("4. shopping entry has exact upstream log traceability", 10, False,
              f"exception: {exc}")


def main() -> None:
    check_1_complete_input_log()
    check_2_ordered_maintenance_log()
    check_3_exact_conditional_grocy_amount()
    check_4_exact_log_traceability()
    total = sum(weight for _, weight, _, _ in _checks)
    earned = sum(weight for _, weight, passed, _ in _checks if passed)
    all_pass = bool(_checks) and all(passed for _, _, passed, _ in _checks)
    print(f"SCORE: {earned / total if total else 0:.3f}  PASS: {all_pass}  ({earned}/{total})",
          file=sys.stderr)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
