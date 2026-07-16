# SPDX-License-Identifier: Apache-2.0
"""Acceptance checks a-f for the Brian-design auto-exposure spike.

Runs six live uvicorn servers (exposed + unmodified-main-style pair per test server), drives them
with the official MCP client (streamable HTTP) and aiohttp for the HTTP door, and prints
PASS/FAIL per check with captured output.

  a. R2: verbatim main handlers execute over MCP; MCP + HTTP-door calls mutate the SAME
     per-session state (core invariant).
  b. R1/R3: tools/list = finance's 5 typed tools (harvested schemas), workplace's 27 (override),
     aviary step+close. Zero decorators.
  c. R4: HTTP-door byte-parity vs an unmodified main-style app (happy + error paths).
  d. Error mapping over MCP (unseeded 400, unknown tool, malformed args, missing/invalid token).
  e. The aviary hazard: step with a DIFFERENT rollout's env_id over MCP.
  f. Concurrency sanity: interleaved sessions, no state cross-talk.
  g. (bonus) allowed_tools claim restricts tools/list and tools/call.

Run:  ../../.venv/bin/python run_checks.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import sys
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import aiohttp
import uvicorn

import servers as spike_servers  # sets sys.path for main_snapshot + vendor
from autoexpose import TOKEN_HEADER, TOKEN_SALT, mint_session_cookie
from itsdangerous import URLSafeSerializer
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

SPIKE_DIR = Path(__file__).resolve().parent
OUT = SPIKE_DIR / "out"
OUT.mkdir(exist_ok=True)

PORTS = {
    "finance": 18871,
    "finance_plain": 18872,
    "workplace": 18873,
    "workplace_plain": 18874,
    "aviary": 18875,
    "aviary_plain": 18876,
}
PAGE_PORT = 18899
PAGE_URL = f"http://127.0.0.1:{PAGE_PORT}/page.html"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


# ---------------------------------------------------------------- infrastructure


def start_page_server() -> ThreadingHTTPServer:
    page_dir = Path(tempfile.mkdtemp(prefix="spike_pages_"))
    (page_dir / "page.html").write_text(
        "<html><body><p>NVIDIA reported record data center revenue of $30.77B in Q3 FY2025.</p></body></html>"
    )

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", PAGE_PORT), partial(Quiet, directory=str(page_dir)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


async def start_uvicorn(app, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    asyncio.get_event_loop().create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    return server


def mcp_url(key: str) -> str:
    return f"http://127.0.0.1:{PORTS[key]}/mcp"


async def mcp_list_tools(url: str, token: str | None):
    headers = {TOKEN_HEADER: token} if token else {}
    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return (await session.list_tools()).tools


async def mcp_call(url: str, token: str | None, name: str, args: dict):
    headers = {TOKEN_HEADER: token} if token else {}
    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(name, args)


def result_text(result) -> str:
    return "".join(c.text for c in result.content if getattr(c, "type", None) == "text")


async def http_post_json(client: aiohttp.ClientSession, port: int, path: str, payload: dict) -> tuple[int, dict]:
    async with client.post(f"http://127.0.0.1:{port}{path}", json=payload) as resp:
        return resp.status, await resp.json()


async def raw_post(
    client: aiohttp.ClientSession, port: int, path: str, body: bytes, cookie: str | None = None
) -> tuple[int, bytes, list[tuple[bytes, bytes]]]:
    headers = {"content-type": "application/json"}
    if cookie:
        headers["cookie"] = cookie
    async with client.post(f"http://127.0.0.1:{port}{path}", data=body, headers=headers) as resp:
        return resp.status, await resp.read(), list(resp.raw_headers)


VOLATILE_HEADERS = {b"date", b"set-cookie"}


def filter_headers(raw: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    return [(k, v) for k, v in raw if k.lower() not in VOLATILE_HEADERS]


def parity_compare(label: str, a: tuple, b: tuple, expect_equal: bool = True) -> None:
    """a = exposed response, b = plain-main response: (status, body, headers)."""
    same = a[0] == b[0] and a[1] == b[1] and filter_headers(a[2]) == filter_headers(b[2])
    if expect_equal:
        check(
            f"c. byte-parity: {label}",
            same,
            f"status {a[0]}=={b[0]}, body {len(a[1])}B" if same else f"exposed={a[:2]!r} plain={b[:2]!r}",
        )
    else:
        return


# ---------------------------------------------------------------- checks


async def check_b_tools_list(tokens: dict[str, str]) -> None:
    section("CHECK b (R1/R3): tools/list — harvested typed schemas, dispatcher override, plumbing routes")

    fin_tools = await mcp_list_tools(mcp_url("finance"), tokens["finance"])
    fin_names = sorted(t.name for t in fin_tools)
    expected_fin = sorted(
        ["sec_filing_search", "parse_html_page", "retrieve_information", "submit_final_result", "web_search"]
    )
    check("b. finance tools/list == 5 typed routes", fin_names == expected_fin, f"{fin_names}")
    by_name = {t.name: t for t in fin_tools}
    php = by_name["parse_html_page"].inputSchema
    check(
        "b. finance parse_html_page harvested schema has field names + required",
        sorted(php.get("properties", {})) == ["key", "url"] and sorted(php.get("required", [])) == ["key", "url"],
        json.dumps(php)[:200],
    )
    sfs = by_name["sec_filing_search"].inputSchema
    check(
        "b. finance sec_filing_search schema fields",
        sorted(sfs.get("properties", {})) == ["end_date", "form_types", "start_date", "ticker"],
        f"props={sorted(sfs.get('properties', {}))}",
    )
    check(
        "b. finance descriptions from docstrings",
        (by_name["sec_filing_search"].description or "").startswith("Search for SEC filings by ticker symbol."),
        repr((by_name["sec_filing_search"].description or "")[:80]),
    )

    wp_tools = await mcp_list_tools(mcp_url("workplace"), tokens["workplace"])
    wp_names = sorted(t.name for t in wp_tools)
    check("b. workplace tools/list has 27 tools (inventory override)", len(wp_names) == 27, f"{len(wp_names)} tools")
    wp_by_name = {t.name: t for t in wp_tools}
    send_schema = wp_by_name.get("email_send_email")
    check(
        "b. workplace email_send_email schema from get_tools()['schemas']",
        send_schema is not None
        and sorted(send_schema.inputSchema.get("properties", {})) == ["body", "recipient", "subject"],
        json.dumps(send_schema.inputSchema if send_schema else {})[:160],
    )

    if "aviary" in tokens:
        av_tools = await mcp_list_tools(mcp_url("aviary"), tokens["aviary"])
        av_names = sorted(t.name for t in av_tools)
        check("b. aviary tools/list == [close, step]", av_names == ["close", "step"], f"{av_names}")
        step_schema = {t.name: t for t in av_tools}["step"].inputSchema
        check(
            "b. aviary step schema exposes env_id + action",
            sorted(step_schema.get("properties", {})) == ["action", "env_id"],
            f"props={sorted(step_schema.get('properties', {}))}",
        )
        print("\naviary step schema:", json.dumps(step_schema)[:400])

    print("\nfinance tools/list:", json.dumps([t.model_dump(exclude_none=True) for t in fin_tools], indent=1)[:1200])
    print("\nworkplace 27 names:", wp_names)


async def check_a_state(fin_server, fin_client: aiohttp.ClientSession, fin_token: str, fin_sid: str) -> None:
    section("CHECK a (R2): verbatim handlers over MCP; MCP + HTTP door share the SAME per-session state")

    # 1. MCP writes state: parse_html_page stores the page under key 'mcp_doc'.
    result = await mcp_call(mcp_url("finance"), fin_token, "parse_html_page", {"url": PAGE_URL, "key": "mcp_doc"})
    text = result_text(result)
    print("MCP parse_html_page ->", text.strip()[:300])
    check(
        "a. MCP call ran the verbatim Request-taking handler (session via minted cookie)",
        not result.isError and "SUCCESS" in text and "mcp_doc" in text,
        text.strip()[:120],
    )
    check(
        "a. structuredContent mirrors the HTTP JSON body",
        isinstance(result.structuredContent, dict) and "results" in result.structuredContent,
        str(result.structuredContent)[:120],
    )

    # 2. HTTP door (same session cookie) writes 'http_doc'; its response lists BOTH keys.
    status, body = await http_post_json(fin_client, PORTS["finance"], "/parse_html_page", {"url": PAGE_URL, "key": "http_doc"})
    print("HTTP parse_html_page ->", body["results"].strip()[:300])
    check(
        "a. HTTP-door call sees the key MCP stored (same storage dict)",
        status == 200 and "mcp_doc" in body["results"] and "http_doc" in body["results"],
        body["results"].strip().replace("\n", " | ")[:160],
    )

    # 3. MCP reads state back: retrieve_information's missing-key error enumerates session keys.
    result = await mcp_call(mcp_url("finance"), fin_token, "retrieve_information", {"prompt": "{{missing_key}}"})
    text = result_text(result)
    print("MCP retrieve_information ->", text.strip()[:300])
    check(
        "a. MCP call reads back HTTP-door mutations (available keys list)",
        "mcp_doc" in text and "http_doc" in text,
        text.strip()[:160],
    )

    # 4. Server-side ground truth: one storage dict, keyed by the token's session id.
    storage = fin_server._data_storage.get(fin_sid, {})
    check(
        "a. server _data_storage[token_sid] holds exactly both docs",
        sorted(storage) == ["http_doc", "mcp_doc"],
        f"_data_storage[{fin_sid[:8]}...] keys = {sorted(storage)}",
    )


async def check_c_parity(secrets: dict[str, str]) -> None:
    section("CHECK c (R4): HTTP door byte-parity vs unmodified main-style app")

    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector, cookie_jar=aiohttp.DummyCookieJar()) as client:
        # Same class name + config name on both instances => same SessionMiddleware secret => the
        # SAME minted cookie is valid on both, so both sides see identical requests, byte for byte.
        def cookies(server_key: str, sid: str) -> str:
            secret = secrets[server_key]
            return mint_session_cookie(secret, secret, sid)

        # ---------- finance ----------
        fin_cookie = cookies("finance", "parity-fin-1")
        seed = b"{}"
        a = await raw_post(client, PORTS["finance"], "/seed_session", seed, fin_cookie)
        b = await raw_post(client, PORTS["finance_plain"], "/seed_session", seed, fin_cookie)
        a_json, b_json = json.loads(a[1]), json.loads(b[1])
        mcp_meta = a_json.pop("mcp", None)
        print("seed_session exposed body:", json.dumps(json.loads(a[1].decode()))[:60], "... plus 'mcp':", json.dumps(mcp_meta)[:120])
        print("seed_session plain body:  ", b[1].decode())
        check(
            "c. KNOWN ADDITIVE DELTA: /seed_session gains only the 'mcp' key (rest identical)",
            mcp_meta is not None and "mcp" not in b_json and a_json == b_json and a[0] == b[0] == 200,
            f"delta keys: {{'mcp'}}; shared body: {json.dumps(a_json)}",
        )

        payload = json.dumps({"url": PAGE_URL, "key": "doc"}).encode()
        a = await raw_post(client, PORTS["finance"], "/parse_html_page", payload, fin_cookie)
        b = await raw_post(client, PORTS["finance_plain"], "/parse_html_page", payload, fin_cookie)
        parity_compare("finance parse_html_page happy path (200)", a, b)

        bad = json.dumps({"url": PAGE_URL}).encode()  # missing required 'key' -> 422
        a = await raw_post(client, PORTS["finance"], "/parse_html_page", bad, fin_cookie)
        b = await raw_post(client, PORTS["finance_plain"], "/parse_html_page", bad, fin_cookie)
        print("finance 422 body:", a[1].decode()[:160])
        parity_compare("finance parse_html_page malformed args (422)", a, b)

        payload = json.dumps({"prompt": "{{nokey}}"}).encode()
        a = await raw_post(client, PORTS["finance"], "/retrieve_information", payload, fin_cookie)
        b = await raw_post(client, PORTS["finance_plain"], "/retrieve_information", payload, fin_cookie)
        parity_compare("finance retrieve_information soft-error (200)", a, b)

        payload = json.dumps({"final_result": "42"}).encode()
        a = await raw_post(client, PORTS["finance"], "/submit_final_result", payload, fin_cookie)
        b = await raw_post(client, PORTS["finance_plain"], "/submit_final_result", payload, fin_cookie)
        parity_compare("finance submit_final_result (200)", a, b)

        payload = json.dumps({"anything": 1}).encode()
        a = await raw_post(client, PORTS["finance"], "/made_up_tool", payload, fin_cookie)
        b = await raw_post(client, PORTS["finance_plain"], "/made_up_tool", payload, fin_cookie)
        parity_compare("finance catch-all unknown tool (200 soft error)", a, b)

        # no-cookie request: both mint fresh sessions, identical bodies
        payload = json.dumps({"final_result": "no-cookie"}).encode()
        a = await raw_post(client, PORTS["finance"], "/submit_final_result", payload, None)
        b = await raw_post(client, PORTS["finance_plain"], "/submit_final_result", payload, None)
        parity_compare("finance request with NO cookie (fresh session on both)", a, b)

        # ---------- workplace ----------
        wp_cookie = cookies("workplace", "parity-wp-1")
        a = await raw_post(client, PORTS["workplace"], "/seed_session", seed, wp_cookie)
        b = await raw_post(client, PORTS["workplace_plain"], "/seed_session", seed, wp_cookie)
        a_json, b_json = json.loads(a[1]), json.loads(b[1])
        a_json.pop("mcp", None)
        check("c. workplace /seed_session additive-only delta", a_json == b_json and a[0] == b[0] == 200, json.dumps(a_json))

        payload = json.dumps({"recipient": "jane.doe@company.com", "subject": "Parity", "body": "hello"}).encode()
        a = await raw_post(client, PORTS["workplace"], "/email_send_email", payload, wp_cookie)
        b = await raw_post(client, PORTS["workplace_plain"], "/email_send_email", payload, wp_cookie)
        print("workplace happy body:", a[1].decode()[:120])
        parity_compare("workplace dispatcher email_send_email happy (200)", a, b)

        payload = json.dumps({"recipient": "jane.doe@company.com"}).encode()  # missing args
        a = await raw_post(client, PORTS["workplace"], "/email_send_email", payload, wp_cookie)
        b = await raw_post(client, PORTS["workplace_plain"], "/email_send_email", payload, wp_cookie)
        print("workplace 200-soft-error body:", a[1].decode()[:200])
        parity_compare("workplace 200-soft-error (bad args)", a, b)

        unseeded = cookies("workplace", "parity-wp-unseeded")
        payload = json.dumps({"recipient": "x@y.z", "subject": "s", "body": "b"}).encode()
        a = await raw_post(client, PORTS["workplace"], "/email_send_email", payload, unseeded)
        b = await raw_post(client, PORTS["workplace_plain"], "/email_send_email", payload, unseeded)
        print("workplace unseeded 400 body:", a[1].decode())
        parity_compare("workplace unseeded session (400)", a, b)
        check("c. workplace unseeded is HTTP 400", a[0] == 400, f"status={a[0]}")

        payload = json.dumps({}).encode()
        a = await raw_post(client, PORTS["workplace"], "/unknown_tool_name", payload, wp_cookie)
        b = await raw_post(client, PORTS["workplace_plain"], "/unknown_tool_name", payload, wp_cookie)
        print("workplace unknown-tool body:", a[1].decode()[:160])
        parity_compare("workplace unknown tool via catch-all (200 soft error)", a, b)

        # ---------- aviary (only when fhaviary is installed) ----------
        if "aviary" in PORTS:
            av_cookie = cookies("aviary", "parity-av-1")
            seed_payload = json.dumps({"task_idx": 0}).encode()
            a = await raw_post(client, PORTS["aviary"], "/seed_session", seed_payload, av_cookie)
            b = await raw_post(client, PORTS["aviary_plain"], "/seed_session", seed_payload, av_cookie)
            a_json, b_json = json.loads(a[1]), json.loads(b[1])
            a_json.pop("mcp", None)
            env_a, env_b = a_json.pop("env_id"), b_json.pop("env_id")
            check(
                "c. aviary /seed_session parity modulo env_id uuid (nondeterministic on main too) + 'mcp'",
                a_json == b_json and a[0] == b[0] == 200,
                "obs+tools identical; env_id uuids differ by construction",
            )

            step = lambda env_id: json.dumps(
                {"env_id": env_id, "action": [{"type": "function_call", "call_id": "c1", "name": "cast_float", "arguments": json.dumps({"x": "3.14"})}]}
            ).encode()
            a = await raw_post(client, PORTS["aviary"], "/step", step(env_a), av_cookie)
            b = await raw_post(client, PORTS["aviary_plain"], "/step", step(env_b), av_cookie)
            print("aviary step body:", a[1].decode()[:200])
            parity_compare("aviary /step happy path (200)", a, b)

            payload = json.dumps({"env_id": "no-such-env"}).encode()
            a = await raw_post(client, PORTS["aviary"], "/close", payload, av_cookie)
            b = await raw_post(client, PORTS["aviary_plain"], "/close", payload, av_cookie)
            print("aviary close-unknown body:", a[1].decode())
            parity_compare("aviary /close unknown env (200, success=false)", a, b)

            a = await raw_post(client, PORTS["aviary"], "/close", json.dumps({"env_id": env_a}).encode(), av_cookie)
            b = await raw_post(client, PORTS["aviary_plain"], "/close", json.dumps({"env_id": env_b}).encode(), av_cookie)
            parity_compare("aviary /close happy path (200)", a, b)

        # ---------- openapi surface ----------
        async with client.get(f"http://127.0.0.1:{PORTS['finance']}/openapi.json") as r:
            exposed_oapi = await r.json()
        async with client.get(f"http://127.0.0.1:{PORTS['finance_plain']}/openapi.json") as r:
            plain_oapi = await r.json()
        path_delta = set(exposed_oapi["paths"]) ^ set(plain_oapi["paths"])
        tool_paths_equal = all(
            exposed_oapi["paths"][p] == plain_oapi["paths"][p]
            for p in plain_oapi["paths"]
            if p != "/seed_session"
        )
        check(
            "c. openapi: path set unchanged (/mcp hidden); every tool route's spec byte-identical",
            path_delta == set() and tool_paths_equal,
            f"path delta={path_delta}; seed_session response schema differs (documented delta): "
            f"{exposed_oapi['paths']['/seed_session']['post']['responses']['200']['content'] != plain_oapi['paths']['/seed_session']['post']['responses']['200']['content']}",
        )


async def check_d_errors(tokens: dict[str, str], wp_secret: str) -> None:
    section("CHECK d: error mapping over MCP — what does the model see?")

    serializer = URLSafeSerializer(wp_secret, salt=TOKEN_SALT)

    # 1. Unseeded session (valid token, but seed_session never ran for this sid) -> handler's 400
    stale_token = serializer.dumps("never-seeded-sid")
    result = await mcp_call(mcp_url("workplace"), stale_token, "email_send_email", {"recipient": "a@b.c", "subject": "s", "body": "b"})
    text = result_text(result)
    print("unseeded-400 over MCP -> isError:", result.isError, "| text:", text)
    check(
        "d. unseeded-session 400 -> isError with status + handler detail",
        result.isError and "HTTP 400" in text and "Session not initialized" in text,
        text[:160],
    )

    # 2. Dispatcher unknown tool over MCP -> clean isError, never reaches the catch-all
    result = await mcp_call(mcp_url("workplace"), tokens["workplace"], "email_explode", {})
    text = result_text(result)
    print("unknown-tool over MCP -> isError:", result.isError, "| text:", text[:200])
    check("d. unknown tool -> isError 'Unknown tool'", result.isError and "Unknown tool" in text, text[:120])

    # 3. Malformed args -> the HTTP door's own 422 body, verbatim
    result = await mcp_call(mcp_url("finance"), tokens["finance"], "parse_html_page", {"url": PAGE_URL})
    text = result_text(result)
    print("malformed-args over MCP -> isError:", result.isError, "| text:", text[:220])
    check(
        "d. malformed args -> isError with FastAPI 422 body",
        result.isError and "HTTP 422" in text and "Field required" in text,
        text[:160],
    )

    # 4. Missing token
    result = await mcp_call(mcp_url("workplace"), None, "email_send_email", {"recipient": "a@b.c", "subject": "s", "body": "b"})
    text = result_text(result)
    print("missing-token over MCP -> isError:", result.isError, "| text:", text)
    check("d. missing token -> isError", result.isError and TOKEN_HEADER in text, text[:120])

    # 5. Forged/invalid token
    result = await mcp_call(mcp_url("workplace"), "forged.token.value", "email_send_email", {"recipient": "a@b.c", "subject": "s", "body": "b"})
    text = result_text(result)
    print("invalid-token over MCP -> isError:", result.isError, "| text:", text)
    check("d. invalid token -> isError", result.isError and "Invalid" in text, text[:120])


async def check_e_aviary_hazard(av_server) -> None:
    section("CHECK e: the aviary hazard — MCP step with a DIFFERENT rollout's env_id")

    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as c1, aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as c2:
        _, seed1 = await http_post_json(c1, PORTS["aviary"], "/seed_session", {"task_idx": 1})
        _, seed2 = await http_post_json(c2, PORTS["aviary"], "/seed_session", {"task_idx": 2})
        env1, tok1 = seed1["env_id"], seed1["mcp"]["headers"][TOKEN_HEADER]
        env2 = seed2["env_id"]
        print(f"rollout 1: env_id={env1}  |  rollout 2: env_id={env2}")

        reward_before = dict(av_server.env_id_to_total_reward)
        action = [{"type": "function_call", "call_id": "x1", "name": "print_story", "arguments": json.dumps({"story": "five word story right here"})}]
        # Session 1's token, session 2's env_id:
        result = await mcp_call(mcp_url("aviary"), tok1, "step", {"env_id": env2, "action": action})
        text = result_text(result)
        print("cross-rollout step -> isError:", result.isError, "| body:", text[:200])
        reward_after = dict(av_server.env_id_to_total_reward)
        interfered = not result.isError and reward_after.get(env2, 0.0) > reward_before.get(env2, 0.0)
        check(
            "e. HAZARD CONFIRMED: session 1's token can step session 2's env (env registry is env_id-keyed, not session-keyed)",
            interfered,
            f"env2 total_reward {reward_before.get(env2, 0.0)} -> {reward_after.get(env2, 0.0)}; "
            f"identical behavior to the HTTP door on main (env_id is the only key)",
        )


async def check_f_concurrency(wp_server) -> None:
    section("CHECK f: two interleaved sessions, no state cross-talk (dispatcher, over MCP)")

    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as cA, aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True)
    ) as cB:
        _, seedA = await http_post_json(cA, PORTS["workplace"], "/seed_session", {})
        _, seedB = await http_post_json(cB, PORTS["workplace"], "/seed_session", {})
        tokA = seedA["mcp"]["headers"][TOKEN_HEADER]
        tokB = seedB["mcp"]["headers"][TOKEN_HEADER]

        url = mcp_url("workplace")
        sends = []
        for i in range(6):
            tok, tag = (tokA, "AAA") if i % 2 == 0 else (tokB, "BBB")
            sends.append(
                mcp_call(url, tok, "email_send_email",
                         {"recipient": "jane.doe@company.com", "subject": f"subject-{tag}-{i}", "body": "x"})
            )
        send_results = await asyncio.gather(*sends)
        check("f. 6 interleaved MCP sends all succeeded", all(not r.isError for r in send_results),
              "; ".join(result_text(r)[:40] for r in send_results[:2]))

        searchA, searchB = await asyncio.gather(
            mcp_call(url, tokA, "email_search_emails", {"query": "subject-"}),
            mcp_call(url, tokB, "email_search_emails", {"query": "subject-"}),
        )
        tA, tB = result_text(searchA), result_text(searchB)
        okA = "subject-AAA" in tA and "subject-BBB" not in tA
        okB = "subject-BBB" in tB and "subject-AAA" not in tB
        print("session A sees:", re.findall(r"subject-\w+-\d+", tA))
        print("session B sees:", re.findall(r"subject-\w+-\d+", tB))
        check("f. session A sees only A's emails; B only B's (no cross-talk)", okA and okB,
              f"A={re.findall(r'subject-[A-Z]+', tA)[:4]} B={re.findall(r'subject-[A-Z]+', tB)[:4]}")
        check(
            "f. two distinct tool envs live server-side",
            len(wp_server.session_id_to_tool_env) >= 2,
            f"{len(wp_server.session_id_to_tool_env)} sessions in session_id_to_tool_env",
        )


async def check_g_allowed_tools(wp_secret: str) -> None:
    section("CHECK g (bonus): allowed_tools claim inside the signed token")

    serializer = URLSafeSerializer(wp_secret, salt=TOKEN_SALT)
    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as c:
        _, seed = await http_post_json(c, PORTS["workplace"], "/seed_session", {})
        full_token = seed["mcp"]["headers"][TOKEN_HEADER]
        sid = serializer.loads(full_token)
    restricted = serializer.dumps({"sid": sid, "tools": ["email_send_email"]})

    tools = await mcp_list_tools(mcp_url("workplace"), restricted)
    check("g. restricted token: tools/list shows only the allowed tool", [t.name for t in tools] == ["email_send_email"],
          f"{[t.name for t in tools]}")
    result = await mcp_call(mcp_url("workplace"), restricted, "calendar_create_event",
                            {"event_name": "x", "participant_email": "a@b.c", "event_start": "2026-01-01 10:00:00", "duration": "30"})
    check("g. restricted token: disallowed call -> isError", result.isError and "not allowed" in result_text(result),
          result_text(result)[:100])
    result = await mcp_call(mcp_url("workplace"), restricted, "email_send_email",
                            {"recipient": "jane.doe@company.com", "subject": "ok", "body": "ok"})
    check("g. restricted token: allowed call still works", not result.isError, result_text(result)[:80])


# ---------------------------------------------------------------- main


async def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    print("Building servers from BYTE-IDENTICAL origin/main handler code (see proofs/diff_handlers.sh)...")

    fin_server, fin_app = spike_servers.build_finance(expose=True)
    fin_plain_server, fin_plain_app = spike_servers.build_finance(expose=False)
    wp_server, wp_app = spike_servers.build_workplace(expose=True)
    wp_plain_server, wp_plain_app = spike_servers.build_workplace(expose=False)

    app_by_key = [
        ("finance", fin_app), ("finance_plain", fin_plain_app),
        ("workplace", wp_app), ("workplace_plain", wp_plain_app),
    ]
    av_server = None
    if spike_servers.AVIARY_AVAILABLE:
        av_server, av_app = spike_servers.build_aviary(expose=True)
        av_plain_server, av_plain_app = spike_servers.build_aviary(expose=False)
        app_by_key += [("aviary", av_app), ("aviary_plain", av_plain_app)]
    else:
        for k in ("aviary", "aviary_plain"):
            PORTS.pop(k, None)
        print("NOTE: fhaviary not installed — skipping the aviary plumbing-hazard case "
              "(finance + workplace cover the typed + dispatcher paradigms).")

    start_page_server()
    servers = []
    for key, app in app_by_key:
        servers.append(await start_uvicorn(app, PORTS[key]))
    print(f"{len(servers)} uvicorn servers up on ports {sorted(PORTS.values())}; page server on {PAGE_PORT}")

    # One seeded session per server, reused throughout (the rollout pattern).
    secrets = {
        "finance": fin_server.get_session_middleware_key(),
        "workplace": wp_server.get_session_middleware_key(),
    }
    secret_precondition = (
        secrets["finance"] == fin_plain_server.get_session_middleware_key()
        and secrets["workplace"] == wp_plain_server.get_session_middleware_key()
    )
    if av_server is not None:
        secrets["aviary"] = av_server.get_session_middleware_key()
        secret_precondition = secret_precondition and secrets["aviary"] == av_plain_server.get_session_middleware_key()
    check(
        "pre. exposed and plain instances share session secrets (parity precondition)",
        secret_precondition,
        str(secrets),
    )

    fin_client = aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True))
    tokens: dict[str, str] = {}
    _, fin_seed = await http_post_json(fin_client, PORTS["finance"], "/seed_session", {})
    tokens["finance"] = fin_seed["mcp"]["headers"][TOKEN_HEADER]
    fin_sid = URLSafeSerializer(secrets["finance"], salt=TOKEN_SALT).loads(tokens["finance"])

    wp_client = aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True))
    _, wp_seed = await http_post_json(wp_client, PORTS["workplace"], "/seed_session", {})
    tokens["workplace"] = wp_seed["mcp"]["headers"][TOKEN_HEADER]

    av_client = None
    if av_server is not None:
        av_client = aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True))
        _, av_seed = await http_post_json(av_client, PORTS["aviary"], "/seed_session", {"task_idx": 0})
        tokens["aviary"] = av_seed["mcp"]["headers"][TOKEN_HEADER]

    wp_secret = secrets["workplace"]

    try:
        await check_b_tools_list(tokens)
        await check_a_state(fin_server, fin_client, tokens["finance"], fin_sid)
        await check_c_parity(secrets)
        await check_d_errors(tokens, wp_secret)
        if av_server is not None:
            await check_e_aviary_hazard(av_server)
        await check_f_concurrency(wp_server)
        await check_g_allowed_tools(wp_secret)
    finally:
        await fin_client.close()
        await wp_client.close()
        if av_client is not None:
            await av_client.close()
        for srv in (fin_server, fin_plain_server):  # finance's own shared aiohttp session (main behavior)
            if srv._session is not None and not srv._session.closed:
                await srv._session.close()
        for s in servers:
            s.should_exit = True
        await asyncio.sleep(0.3)

    section("SUMMARY")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, _ in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    (OUT / "summary.json").write_text(json.dumps([{"check": n, "pass": ok, "detail": d} for n, ok, d in RESULTS], indent=1))
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(main()))
