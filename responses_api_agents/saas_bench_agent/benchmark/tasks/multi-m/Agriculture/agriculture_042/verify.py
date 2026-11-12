# SPDX-License-Identifier: CC-BY-4.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/tasks/multi-m/Agriculture/agriculture_042/verify.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""Verifier for agriculture_042: image-selected three-system traceability.

The source FarmOS record is selected by hashing the image attached to the
actual record, not by reading a fixed task input. Each downstream system is
gated by the preceding validated state, so copied trace text cannot replace
the image-selected FarmOS source.
"""

import hashlib
import html
import json
import os
import re
import subprocess
import sys
import tempfile

FARMOS_PORT = os.getenv("FARMOS_PORT")
FARMOS_CONTAINER = os.getenv("FARMOS_CONTAINER")
GROCY_PORT = os.getenv("GROCY_PORT")
GROCY_CONTAINER = os.getenv("GROCY_CONTAINER")
E_LABEL_PORT = os.getenv("E_LABEL_PORT")
E_LABEL_CONTAINER = os.getenv("E_LABEL_CONTAINER")

for _var_name, _var_val in [
    ("FARMOS_PORT", FARMOS_PORT),
    ("FARMOS_CONTAINER", FARMOS_CONTAINER),
    ("GROCY_PORT", GROCY_PORT),
    ("GROCY_CONTAINER", GROCY_CONTAINER),
    ("E_LABEL_PORT", E_LABEL_PORT),
    ("E_LABEL_CONTAINER", E_LABEL_CONTAINER),
]:
    if not _var_val:
        print(f"FATAL: {_var_name} not set", file=sys.stderr)
        sys.exit(1)

FARMOS_SQLITE = "/opt/drupal/web/sites/default/files/.ht.sqlite"
FARMOS_FILES_ROOT = "/opt/drupal/web/sites/default/files"
GROCY_DB_CANDIDATES = [
    "/config/data/grocy.db",
    "/config/data/data/grocy.db",
    "/var/www/data/grocy.db",
]

SOURCE_LOG_NAME = "Spring Plowing Complete"
SOURCE_ASSET_NAME = "Vineyard Block 1"
BASE_NOTES = "Plowed 120 acres. Soil conditions excellent. Ready for planting."
CANDIDATE_METHOD_BY_SHA256 = {
    "0184ddd5515204b69cc87f1db84ffefa722fb0f4046a7dd0f742b01e35bd8922": "Manual",
    "317c3ce01ab6cb8598a43335a61d518c926e906341946100236c8a252321e633": "Drone-assisted",
}
METHOD_RANK = {"Manual": 0, "Tractor-only": 1, "Drone-assisted": 2}
EXPECTED_ATTACHMENT_SHA256 = max(
    CANDIDATE_METHOD_BY_SHA256,
    key=lambda digest: METHOD_RANK[CANDIDATE_METHOD_BY_SHA256[digest]],
)
EXPECTED_BATCH = "VINO-2025-081"
EXPECTED_METHOD = CANDIDATE_METHOD_BY_SHA256[EXPECTED_ATTACHMENT_SHA256]
EXPECTED_PRODUCT_NAME = f"{EXPECTED_METHOD} Estate Wine 2025"
EXPECTED_BRAND = f"{EXPECTED_METHOD} Estate"
EXPECTED_VINTAGE = "2025"
EXPECTED_GROCY_LINES = (
    f"TRACEABILITY BATCH: {EXPECTED_BATCH}",
    f"FIELD METHOD: {EXPECTED_METHOD}",
    "FARMOS SOURCE: Spring Plowing Complete | Vineyard Block 1",
)
EXPECTED_ELABEL_INFO = (
    f"FIELD METHOD: {EXPECTED_METHOD}; FARMOS SOURCE: Spring Plowing Complete; "
    "Vineyard Block 1"
)
EXPECTED_FARMOS_LINES = (
    BASE_NOTES,
    f"TRACEABILITY BATCH: {EXPECTED_BATCH}",
    f"FIELD METHOD: {EXPECTED_METHOD}",
)

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


def _resolve_elabel_db() -> str:
    candidates = [
        os.getenv("E_LABEL_DB_CONTAINER", ""),
        E_LABEL_CONTAINER.replace("-app", "-db") if "-app" in E_LABEL_CONTAINER else "",
        E_LABEL_CONTAINER + "-db",
        E_LABEL_CONTAINER,
        "elabel-db",
    ]
    for candidate in dict.fromkeys(value for value in candidates if value):
        try:
            rc, _, _ = docker_exec(candidate, "echo", "ok", timeout=5)
            if rc == 0:
                return candidate
        except Exception:
            continue
    return E_LABEL_CONTAINER


E_LABEL_DB_CONTAINER = _resolve_elabel_db()


def elabel_rows(query: str) -> list[list[str]]:
    last_error = "sqlcmd not found"
    for binary in ["/opt/mssql-tools18/bin/sqlcmd", "/opt/mssql-tools/bin/sqlcmd"]:
        rc, stdout, stderr = docker_exec(
            E_LABEL_DB_CONTAINER,
            binary,
            "-S", "localhost",
            "-U", "sa",
            "-P", "Elabel2024!Strong",
            "-d", "elabel",
            "-C",
            "-h", "-1",
            "-W",
            "-s", "|",
            "-Q", "SET NOCOUNT ON; " + query,
        )
        if rc == 0:
            rows = []
            for line in stdout.splitlines():
                line = line.strip()
                if not line or "rows affected" in line.lower():
                    continue
                rows.append([part.strip() for part in line.split("|")])
            return rows
        last_error = stderr.strip() or stdout.strip()
        if "not found" not in last_error.lower():
            break
    raise RuntimeError(last_error)


def _normalize(text: str) -> str:
    visible = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    return re.sub(r"\s+", " ", visible).strip()


def _text_lines(text: str) -> list[str]:
    visible = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>", "\n", text or "")
    visible = html.unescape(re.sub(r"<[^>]+>", "", visible))
    return [re.sub(r"\s+", " ", line).strip()
            for line in visible.splitlines() if line.strip()]


def _container_file_path(uri: str) -> str:
    if uri.startswith("public://"):
        return FARMOS_FILES_ROOT + "/" + uri.removeprefix("public://").lstrip("/")
    if uri.startswith("/"):
        return uri
    raise ValueError(f"unsupported FarmOS file URI: {uri!r}")


def _attachment_sha256(uri: str) -> str:
    source_path = _container_file_path(uri)
    with tempfile.TemporaryDirectory(prefix="agriculture_042_") as temp_dir:
        local_path = os.path.join(temp_dir, os.path.basename(source_path) or "attachment")
        result = subprocess.run(
            ["docker", "cp", f"{FARMOS_CONTAINER}:{source_path}", local_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"docker cp failed for {source_path}: {result.stderr.strip()}")
        digest = hashlib.sha256()
        with open(local_path, "rb") as attachment:
            for chunk in iter(lambda: attachment.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


_candidate_logs: list[dict] | None = None
_attachments_by_log: dict[int, list[dict]] = {}
_visual_source: tuple[dict, dict, str] | None = None


def _load_candidate_logs() -> list[dict]:
    global _candidate_logs
    if _candidate_logs is None:
        _candidate_logs = farmos_sql_json(
            "SELECT DISTINCT lfd.id, lfd.name, lfd.type, "
            "COALESCE(lfd.notes__value, '') AS notes, afd.name AS asset_name "
            "FROM log_field_data lfd "
            "JOIN log__asset la ON la.entity_id = lfd.id AND la.deleted = 0 "
            "JOIN asset_field_data afd ON afd.id = la.asset_target_id "
            "WHERE lfd.type = 'harvest' "
            "AND lfd.name = 'Spring Plowing Complete' "
            "AND afd.name = 'Vineyard Block 1' ORDER BY lfd.id"
        )
    return _candidate_logs


def _load_attachments(log_id: int) -> list[dict]:
    if log_id in _attachments_by_log:
        return _attachments_by_log[log_id]
    rows = farmos_sql_json(
        "SELECT fm.fid, fm.uri, fm.filename, fm.filemime, 'image' AS field_name "
        "FROM log__image li JOIN file_managed fm ON fm.fid = li.image_target_id "
        f"WHERE li.entity_id = {log_id} AND li.deleted = 0"
    )
    file_rows = farmos_sql_json(
        "SELECT fm.fid, fm.uri, fm.filename, fm.filemime, 'file' AS field_name "
        "FROM log__file lf JOIN file_managed fm ON fm.fid = lf.file_target_id "
        f"WHERE lf.entity_id = {log_id} AND lf.deleted = 0 "
        "AND fm.filemime LIKE 'image/%'"
    )
    unique = {str(row["fid"]): row for row in rows + file_rows}
    _attachments_by_log[log_id] = list(unique.values())
    return _attachments_by_log[log_id]


def _find_visual_source() -> tuple[dict, dict, str]:
    global _visual_source
    if _visual_source is not None:
        return _visual_source

    recognized: list[tuple[dict, dict, str, str]] = []
    for log in _load_candidate_logs():
        attachments = _load_attachments(int(log["id"]))
        if len(attachments) != 1:
            raise RuntimeError(
                f"candidate log #{log['id']} must have exactly one classification image, "
                f"found {len(attachments)}"
            )
        attachment = attachments[0]
        digest = _attachment_sha256(str(attachment["uri"]))
        method = CANDIDATE_METHOD_BY_SHA256.get(digest)
        if method is None:
            raise RuntimeError(
                f"candidate log #{log['id']} has an unrecognized classification image"
            )
        recognized.append((log, attachment, method, digest))

    digests = {match[3] for match in recognized}
    if digests != set(CANDIDATE_METHOD_BY_SHA256):
        raise RuntimeError(
            f"candidate classification image set changed: found {sorted(digests)}"
        )

    highest_rank = max(METHOD_RANK[match[2]] for match in recognized)
    highest = [match for match in recognized if METHOD_RANK[match[2]] == highest_rank]
    if len(highest) != 1:
        raise RuntimeError(f"expected one uniquely higher-ranked visual source, found {len(highest)}")

    source_log, attachment, method, _ = highest[0]
    _visual_source = (source_log, attachment, method)
    return _visual_source


_farmos_ok = False
_grocy_ok = False
_elabel_ok = False
_source_log_id = 0
_source_method = ""
_grocy_product_id = ""
_elabel_product_id = ""


def check_1_farmos_visual_source_and_notes() -> None:
    global _farmos_ok, _source_log_id, _source_method
    try:
        candidates = _load_candidate_logs()
        if len(candidates) != 2:
            check(
                "1. FarmOS visual source and preserved notes",
                5,
                False,
                f"expected two exact source candidates, found {len(candidates)}",
            )
            return
        source_log, attachment, _source_method = _find_visual_source()
        _source_log_id = int(source_log["id"])
        problems = []
        if _source_method != EXPECTED_METHOD:
            problems.append(
                f"selected method={_source_method!r}, expected controlled class {EXPECTED_METHOD!r}"
            )
        if _text_lines(str(source_log.get("notes", ""))) != list(EXPECTED_FARMOS_LINES):
            problems.append("selected record does not preserve baseline notes plus both exact trace lines")
        for candidate in candidates:
            if int(candidate["id"]) == _source_log_id:
                continue
            if _normalize(str(candidate.get("notes", ""))) != BASE_NOTES:
                problems.append(f"non-selected candidate #{candidate['id']} was modified")
        if not str(attachment.get("filemime", "")).startswith("image/"):
            problems.append("selected attachment is not an image")
        _farmos_ok = not problems
        check(
            "1. FarmOS visual source and preserved notes",
            5,
            _farmos_ok,
            f"log #{_source_log_id}, method={_source_method}, "
            f"attachment={attachment.get('filename', '')}"
            if _farmos_ok else "; ".join(problems),
        )
    except Exception as exc:
        check("1. FarmOS visual source and preserved notes", 5, False, f"exception: {exc}")


def check_2_grocy_exact_trace_product() -> None:
    global _grocy_ok, _grocy_product_id
    if not _farmos_ok:
        check("2. Grocy exact trace product", 5, False,
              "gated: image-selected FarmOS source is invalid")
        return
    try:
        safe_name = EXPECTED_PRODUCT_NAME.replace("'", "''")
        safe_batch = EXPECTED_BATCH.replace("'", "''")
        rows = grocy_sql_json(
            "SELECT id, name, COALESCE(description, '') AS description FROM products "
            f"WHERE name = '{safe_name}' "
            f"OR description LIKE '%{safe_batch}%' ORDER BY id"
        )
        problems = []
        if len(rows) != 1:
            problems.append(f"expected one exact/associated product, found {len(rows)}")
        else:
            row = rows[0]
            if row.get("name") != EXPECTED_PRODUCT_NAME:
                problems.append(f"wrong product name: {row.get('name')!r}")
            if _text_lines(str(row.get("description", ""))) != list(EXPECTED_GROCY_LINES):
                problems.append("description is not the exact three-field traceability record")
            if not problems:
                _grocy_product_id = str(row["id"])
        _grocy_ok = not problems
        check(
            "2. Grocy exact trace product",
            5,
            _grocy_ok,
            f"product id={_grocy_product_id}" if _grocy_ok else "; ".join(problems),
        )
    except Exception as exc:
        check("2. Grocy exact trace product", 5, False, f"exception: {exc}")


def check_3_elabel_exact_trace_record() -> None:
    global _elabel_ok, _elabel_product_id
    if not _farmos_ok or not _grocy_ok:
        check("3. e-label exact trace record", 10, False,
              "gated: FarmOS-to-Grocy traceability chain is invalid")
        return
    try:
        safe_info = (
            "REPLACE(REPLACE(ISNULL(FBOAdditionalInfo,''), CHAR(13), ' '), CHAR(10), ' ')"
        )
        safe_name = EXPECTED_PRODUCT_NAME.replace("'", "''")
        safe_batch = EXPECTED_BATCH.replace("'", "''")
        rows = elabel_rows(
            "SELECT CAST(Id AS NVARCHAR(36)), Name, ISNULL(Sku,''), ISNULL(Brand,''), "
            "CAST(WineVintage AS NVARCHAR(10)), " + safe_info + " FROM Product "
            f"WHERE Name = N'{safe_name}' OR Sku = '{safe_batch}' ORDER BY CreatedOn"
        )
        problems = []
        if len(rows) != 1 or len(rows[0]) < 6:
            problems.append(f"expected one exact/associated wine row, found {len(rows)}")
        else:
            row = rows[0]
            values = {
                "id": row[0], "name": row[1], "sku": row[2], "brand": row[3],
                "vintage": row[4], "info": _normalize(row[5]),
            }
            expected = {
                "name": EXPECTED_PRODUCT_NAME,
                "sku": EXPECTED_BATCH,
                "brand": EXPECTED_BRAND,
                "vintage": EXPECTED_VINTAGE,
                "info": EXPECTED_ELABEL_INFO,
            }
            mismatches = [
                f"{key}={values[key]!r}" for key in expected if values[key] != expected[key]
            ]
            problems.extend(mismatches)
            if not problems:
                _elabel_product_id = values["id"]
        _elabel_ok = not problems
        check(
            "3. e-label exact trace record",
            10,
            _elabel_ok,
            f"wine id={_elabel_product_id}" if _elabel_ok else "; ".join(problems),
        )
    except Exception as exc:
        check("3. e-label exact trace record", 10, False, f"exception: {exc}")


def main() -> None:
    check_1_farmos_visual_source_and_notes()
    check_2_grocy_exact_trace_product()
    check_3_elabel_exact_trace_record()

    total = sum(weight for _, weight, _, _ in _checks)
    earned = sum(weight for _, weight, passed, _ in _checks if passed)
    all_pass = bool(_checks) and all(passed for _, _, passed, _ in _checks)
    score = earned / total if total else 0.0
    print(f"SCORE: {score:.3f}  PASS: {all_pass}  ({earned}/{total})", file=sys.stderr)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
