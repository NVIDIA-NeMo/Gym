# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh application and editor checks without model inference or old task state."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile

import requests
import yaml


def credentials(root: Path, app: str) -> tuple[str, str]:
    from saas_bench.loader import load_tasks

    for task in load_tasks(str(root / "tasks")):
        match = re.search(
            r"^-\s*" + re.escape(app) + r":\s*(\S+)\s*/\s*(\S+)\s*$",
            task["description_md"],
            re.I | re.M,
        )
        if match:
            return match.group(1), match.group(2)
    raise ValueError(f"No benchmark fixture login for {app}")


def response(response):
    response.raise_for_status()
    return response.json()


def runtime_exec(root):
    if os.environ.get("SAAS_NRT_ENROOT") == "1":
        return [sys.executable, str(root / "scripts/nrt/docker_enroot_shim.py"), "exec"]
    return ["podman", "exec"]


def xlsx_a1(data: bytes) -> str:
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        cell = sheet.find('.//s:c[@r="A1"]', ns)
        if cell is None:
            return ""
        if cell.get("t") == "inlineStr":
            return "".join(cell.itertext())
        value = cell.find("s:v", ns)
        if value is None:
            return ""
        if cell.get("t") == "s":
            shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            return "".join(list(shared)[int(value.text)].itertext())
        return value.text or ""


async def onlyoffice_browser(root, base, output, session, *, editor):
    from saas_bench.opencode.browser import BrowserTools

    browser = BrowserTools(
        {
            "allowed_origins": [base],
            "input_files": [],
            "artifacts": str(output / "browser"),
            "viewport": {"width": 1280, "height": 720},
        }
    )
    key = []
    report = {}
    try:
        await browser.start()
        page = browser.page
        report["failed_browser_requests"] = []
        page.on(
            "requestfailed",
            lambda request: report["failed_browser_requests"].append(
                {"url": request.url, "failure": request.failure}
            ),
        )

        def websocket_opened(ws):
            match = re.search(r"/doc/([^/]+)/", ws.url)
            if match:
                key.append(match.group(1))

        page.on("websocket", websocket_opened)
        username, password = credentials(root, "onlyoffice")
        await page.goto(base + "/auth.aspx", wait_until="domcontentloaded")
        await page.locator("#login").fill(username)
        await page.locator("#pwd").fill(password)
        await page.locator("#loginButton").click()
        await page.wait_for_url(lambda url: "auth.aspx" not in url, timeout=60000)
        report["browser_login"] = True
        if editor:
            folder = response(session.get(base + "/api/2.0/files/@my", timeout=30))[
                "response"
            ]["current"]["id"]
            file = response(
                session.post(
                    base + f"/api/2.0/files/{folder}/file",
                    json={"title": "Harness editor proof.xlsx"},
                    timeout=30,
                )
            )["response"]
            file_id = file["id"]
            await page.goto(
                base + f"/products/files/doceditor.aspx?fileid={file_id}",
                wait_until="domcontentloaded",
            )
            frame = None
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                for candidate in page.frames:
                    if await candidate.locator("#ce-cell-content").count():
                        frame = candidate
                        break
                if frame:
                    break
                await asyncio.sleep(1)
            if frame is None:
                report["frame_paths"] = [
                    __import__("urllib.parse", fromlist=["urlsplit"])
                    .urlsplit(f.url)
                    .path
                    for f in page.frames
                ]
                raise RuntimeError("OnlyOffice spreadsheet editor did not become ready")
            report["editor_frame_found"] = True
            (output / "editor-progress.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            await frame.wait_for_function(
                """() => {
                  const name = document.querySelector('#ce-cell-name');
                  const masks = document.querySelectorAll('#loading-mask, .asc-loadmask');
                  return name?.value === 'A1' && !name.disabled &&
                    ![...masks].some(mask => mask.offsetParent !== null);
                }""",
                timeout=120000,
            )
            report["document_load_complete"] = True
            (output / "editor-progress.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            await browser.observe(timeout=5000)
            await frame.locator("#ce-cell-name").fill("A1")
            await frame.locator("#ce-cell-name").press("Enter")
            value = "SaaS harness saved editor proof"
            await frame.locator("#ce-cell-content").fill("")
            await frame.locator("#ce-cell-content").press_sequentially(value)
            await frame.locator("#ce-cell-content").press("Enter")
            report["editor_input"] = True
            deadline = time.monotonic() + 45
            forced = None
            while time.monotonic() < deadline:
                if key:
                    forced = response(
                        session.post(
                            base + "/ds-vpath/coauthoring/CommandService.ashx",
                            json={"c": "forcesave", "key": key[-1]},
                            timeout=30,
                        )
                    )
                    if forced.get("error") == 0:
                        break
                await asyncio.sleep(2)
            report["document_socket_keys_observed"] = len(key)
            report["force_save_reply"] = forced
            if not forced or forced.get("error") != 0:
                raise RuntimeError("Document Server did not accept force-save")
            report["force_save_accepted"] = True
            deadline = time.monotonic() + 60
            saved = False
            while time.monotonic() < deadline and not saved:
                for route in [
                    f"/Products/Files/HttpHandlers/filehandler.ashx?action=download&fileid={file_id}",
                    f"/api/2.0/files/file/{file_id}/download",
                ]:
                    downloaded = await page.request.get(base + route, timeout=15000)
                    data = await downloaded.body()
                    if downloaded.status == 200 and data.startswith(b"PK"):
                        saved = xlsx_a1(data) == value
                        if saved:
                            (output / "saved-editor-proof.xlsx").write_bytes(data)
                            report["saved_workbook_sha256"] = hashlib.sha256(
                                data
                            ).hexdigest()
                            break
                if not saved:
                    await asyncio.sleep(2)
            if not saved:
                raise RuntimeError(
                    "Saved application workbook did not contain the edited A1 value"
                )
            report["callback_persisted_cell"] = True
        return report
    finally:
        (output / "editor-progress.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if browser.page is not None:
            try:
                await browser.observe(timeout=3000)
            except Exception:
                pass
        await browser.close()


def check_app(root, app, manager, output, *, editor):
    base = "http://localhost:" + str(manager.get_port(app))
    username, password = credentials(root, app)
    with requests.Session() as session:
        if app == "openproject":
            from html.parser import HTMLParser

            class LoginForm(HTMLParser):
                token = None

                def handle_starttag(self, tag, attrs):
                    attributes = dict(attrs)
                    if tag == "input" and attributes.get("name") == "authenticity_token":
                        self.token = attributes.get("value")

            login = session.get(base + "/login", timeout=30)
            login.raise_for_status()
            form = LoginForm()
            form.feed(login.text)
            if not form.token:
                raise RuntimeError("OpenProject login form has no CSRF token")
            logged_in = session.post(
                base + "/login", data={
                    "username": username, "password": password,
                    "authenticity_token": form.token,
                }, timeout=30,
            )
            logged_in.raise_for_status()
            user = response(session.get(base + "/api/v3/users/me", timeout=30))
            assert isinstance(user.get("id"), int) and user["id"] > 0
            projects = response(session.get(base + "/api/v3/projects", timeout=30))
            count = projects.get("total", len(projects.get("_embedded", {}).get("elements", [])))
            assert count > 0
            return {"seeded_login": True, "seeded_projects": count}
        if app == "onlyoffice":
            from saas_bench.opencode.verifier import probe_onlyoffice_api

            assert probe_onlyoffice_api(base)["available"]
            token = response(
                session.post(
                    base + "/api/2.0/authentication",
                    json={"userName": username, "password": password},
                    timeout=30,
                )
            )["response"]["token"]
            session.headers["Authorization"] = token
            assert (
                response(session.get(base + "/api/2.0/people/@self", timeout=30))[
                    "status"
                ]
                == 0
            )
            loopback = subprocess.run(
                [
                    *runtime_exec(root),
                    manager.get_container_name(app) + "-community",
                    "curl",
                    "-fsS",
                    "--max-time",
                    "10",
                    base + "/auth.aspx",
                ],
                capture_output=True,
                timeout=20,
            )
            if loopback.returncode:
                raise RuntimeError(
                    "OnlyOffice cannot reach its published URL from the application namespace"
                )
            result = asyncio.run(
                onlyoffice_browser(root, base, output, session, editor=editor)
            )
            # This runs after editor activity, across Elasticsearch's startup
            # disk checks, rather than treating green cluster health as proof
            # that indices accept writes.
            for number in range(3):
                write = subprocess.run(
                    [
                        *runtime_exec(root),
                        manager.get_container_name(app) + "-elasticsearch",
                        "curl",
                        "-fsS",
                        "--max-time",
                        "10",
                        "-X",
                        "PUT",
                        f"http://127.0.0.1:9200/saas-harness-runtime-proof/_doc/{number}",
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        '{"runtime_probe":true}',
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if write.returncode or json.loads(write.stdout).get("result") not in {
                    "created",
                    "updated",
                }:
                    raise RuntimeError(
                        "Elasticsearch rejected a runtime probe write: "
                        + write.stderr[-500:]
                    )
                time.sleep(1)
            result["elasticsearch_writes_after_editor"] = 3
            result["seeded_api_login"] = True
            return result
        if app == "booklore":
            login = response(
                session.post(
                    base + "/api/v1/auth/login",
                    json={"username": username, "password": password},
                    timeout=30,
                )
            )
            session.headers["Authorization"] = "Bearer " + login["accessToken"]
            books = response(session.get(base + "/api/v1/books", timeout=30))
            count = (
                len(books)
                if isinstance(books, list)
                else books.get("totalElements", len(books.get("content", [])))
            )
            assert count > 0
            return {"seeded_login": True, "seeded_books": count}
        if app == "mattermost":
            login = session.post(
                base + "/api/v4/users/login",
                json={"login_id": username, "password": password},
                timeout=30,
            )
            if login.status_code == 401:
                command = [
                    *runtime_exec(root),
                    manager.get_container_name(app) + "-postgres",
                    "env",
                    "PGPASSWORD=mmuser_password",
                    "psql",
                    "-h",
                    "127.0.0.1",
                    "-U",
                    "mmuser",
                    "-d",
                    "mattermost",
                    "-t",
                    "-A",
                    "-c",
                    "SELECT username,email,authservice,deleteat FROM users ORDER BY username",
                ]
                proc = subprocess.run(
                    command, capture_output=True, text=True, timeout=30
                )
                (output / "fixture_users.txt").write_text(proc.stdout)
                diagnostics = {
                    "username_status": login.status_code,
                    "username_error": login.json().get("id"),
                    "fixture_query_status": proc.returncode,
                    "fixture_query_error": proc.stderr[-1000:],
                }
                for line in proc.stdout.splitlines():
                    parts = line.split("|")
                    if len(parts) > 1 and parts[0] == username:
                        alternate = session.post(
                            base + "/api/v4/users/login",
                            json={"login_id": parts[1], "password": password},
                            timeout=30,
                        )
                        diagnostics.update(
                            email_status=alternate.status_code,
                            email_error=alternate.json().get("id"),
                        )
                (output / "login_diagnostics.json").write_text(
                    json.dumps(diagnostics, indent=2)
                )
            user = response(login)
            assert user.get("id")
            return {"seeded_login": True}
        if app == "owncloud":
            assert (
                response(session.get(base + "/status.php", timeout=30))["installed"]
                is True
            )
            session.auth = (username, password)
            target = (
                base
                + "/remote.php/dav/files/"
                + requests.utils.quote(username, safe="")
                + "/harness-runtime-proof.txt"
            )
            session.put(
                target, data=b"SaaS runtime proof", timeout=30
            ).raise_for_status()
            readback = session.get(target, timeout=30)
            readback.raise_for_status()
            assert readback.content == b"SaaS runtime proof"
            return {"installed": True, "seeded_dav_write_read": True}
        if app == "pretix":
            result = session.get(base + "/control/login/", timeout=30)
            result.raise_for_status()
            return {"login_page": result.status_code}
        if app == "bigcapital":
            command = [
                *runtime_exec(root),
                manager.get_container_name(app),
                "mysql",
                "-u",
                "bigcapital",
                "-pbigcapital123",
                "-N",
                "-B",
            ]
            proc = subprocess.run(
                [
                    *command,
                    "-e",
                    "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME LIKE 'bigcapital_tenant_%' ORDER BY SCHEMA_NAME LIMIT 1",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode:
                raise RuntimeError(
                    "BigCapital schema discovery failed: " + proc.stderr[-600:]
                )
            database = proc.stdout.strip()
            if not re.fullmatch(r"bigcapital_tenant_[A-Za-z0-9_]+", database):
                raise RuntimeError("BigCapital tenant database was not discovered")
            query = "SELECT ID,AMOUNT,DELIVERED_AT FROM SALES_ESTIMATES LIMIT 0; SELECT DESCRIPTION,QUANTITY,RATE FROM ITEMS_ENTRIES LIMIT 0;"
            proc = subprocess.run(
                [*command, "-D", database, "-e", query],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode:
                raise RuntimeError(
                    "BigCapital estimate schema check failed: " + proc.stderr[-600:]
                )
            return {"estimate_and_line_schema": True}
    raise ValueError("Unsupported runtime check")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--apps",
        nargs="+",
        choices=[
            "onlyoffice",
            "bigcapital",
            "mattermost",
            "booklore",
            "owncloud",
            "pretix",
            "openproject",
        ],
        required=True,
    )
    parser.add_argument("--editor", action="store_true")
    parser.add_argument("--backend", choices=["enroot", "podman"], default="enroot")
    parser.add_argument("--base-port", type=int, default=44000)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run live application probes in a Slurm allocation")
    if not 1024 <= args.base_port <= 64000:
        parser.error("Use a base port in 1024..64000")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="saas-runtime-check-") as temporary:
        scratch = Path(temporary)
        for name, suffix in [
            ("ENROOT_DATA_PATH", "data"),
            ("ENROOT_CACHE_PATH", "cache"),
            ("ENROOT_RUNTIME_PATH", "runtime"),
            ("ENROOT_TEMP_PATH", "tmp"),
        ]:
            path = scratch / suffix
            path.mkdir()
            os.environ[name] = str(path)
        os.environ.update(
            SAAS_NRT_ENROOT="1" if args.backend == "enroot" else "0",
            SAAS_SKIP_ENV_FILE="1",
            SAAS_ENROOT_STATE_DIR=str(args.output / "diagnostics"),
            SAAS_ENROOT_IMAGE_DIR=str(args.assets / "assets/enroot_images"),
            SAAS_ENROOT_RUNTIME_DIR=str(args.assets / "runtime/enroot"),
            SAAS_UID_SHIM=str(scratch / "uid-shim.so"),
            SAAS_BASE_PORT=str(args.base_port),
            SAAS_SLOT_PREFIX="v21_" + os.environ["SLURM_JOB_ID"],
            SAAS_NRT_START_ATTEMPTS="1",
        )
        from saas_bench.slot import SlotManager

        registry = yaml.safe_load((root / "saas_bench/apps.yaml").read_text())["apps"]
        reports = []
        for app in args.apps:
            output = args.output / app
            output.mkdir()
            manager = SlotManager(registry, 0)
            report = {
                "app": app,
                "job_id": os.environ["SLURM_JOB_ID"],
                "host": socket.gethostname(),
                "backend": args.backend,
                "passed": False,
                "runtime_sha256": {
                    name: hashlib.sha256(
                        (root / "saas_bench" / name).read_bytes()
                    ).hexdigest()
                    for name in [
                        "slot.py",
                        "enroot_slot.py",
                        "apps.yaml",
                        "oo_community_init.sh",
                        "oo_documentserver_init.sh",
                        "opencode/browser.py",
                        "opencode/runtime_check.py",
                    ]
                },
            }
            try:
                manager.start_apps([app], hostname="localhost")
                report.update(check_app(root, app, manager, output, editor=args.editor))
                report["passed"] = True
            except Exception:
                report["error"] = traceback.format_exc()
                network = {}
                port = str(manager.get_port(app))
                for version in ["-4", "-6"]:
                    probe = subprocess.run(
                        [
                            "curl",
                            version,
                            "-sS",
                            "--max-time",
                            "5",
                            "-o",
                            "/dev/null",
                            "-w",
                            "%{http_code}",
                            "http://localhost:"
                            + port
                            + registry[app].get("health_path", "/"),
                        ],
                        capture_output=True,
                        text=True,
                    )
                    network[version] = {
                        "exit": probe.returncode,
                        "status": probe.stdout,
                        "error": probe.stderr[-500:],
                    }
                (output / "failure-network.json").write_text(
                    json.dumps(network, indent=2)
                )
            finally:
                if args.backend == "podman":
                    for container in manager._compose_containers(
                        app, registry[app], manager.get_container_name(app)
                    ):
                        log = subprocess.run(
                            ["podman", "logs", container],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        (output / (container + ".log")).write_text(
                            log.stdout + log.stderr
                        )
                manager.stop_apps([app])
                (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                reports.append(report)
                print(json.dumps({"app": app, "passed": report["passed"]}), flush=True)
    (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    raise SystemExit(0 if all(r["passed"] for r in reports) else 1)


if __name__ == "__main__":
    main()
