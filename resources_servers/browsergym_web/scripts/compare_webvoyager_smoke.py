# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare one deterministic WebVoyager interaction across Selenium and BrowserGym.

This is an environment smoke test, not a benchmark score. It calls the original
WebVoyager Selenium helpers or the Gym BrowserGym backend and writes a common,
secret-free event trace for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK = {
    "benchmark": "webvoyager",
    "task_id": "ArXiv--13",
    "intent": "How many articles on ArXiv with 'SimCSE' in the title?",
    "start_url": "https://arxiv.org/",
    "query": "SimCSE",
}
VIEWPORT = {"width": 1024, "height": 768}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _source_tree_sha256(root: Path, paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        candidate = root / relative
        files = sorted(candidate.rglob("*.py")) if candidate.is_dir() else [candidate]
        for path in files:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


class Trace:
    def __init__(self, out_dir: Path, runtime: str) -> None:
        self.out_dir = out_dir
        self.runtime = runtime
        self.started = time.monotonic()
        self.sequence = 0
        self.events_path = out_dir / "events.jsonl"
        for child in ("screenshots", "observations"):
            (out_dir / child).mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, *, step: int, data: dict[str, Any]) -> None:
        self.sequence += 1
        record = {
            "sequence": self.sequence,
            "timestamp": _utc_now(),
            "elapsed_ms": round((time.monotonic() - self.started) * 1000),
            "runtime": self.runtime,
            "event": event,
            "task_id": TASK["task_id"],
            "step": step,
            "data": data,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def screenshot_ref(self, path: Path) -> dict[str, Any]:
        return {
            "path": str(path.relative_to(self.out_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in (
        "browsergym-core",
        "browsergym-webarena",
        "browsergym-visualwebarena",
        "gymnasium",
        "playwright",
        "selenium",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _manifest(args: argparse.Namespace, runtime: str) -> dict[str, Any]:
    gym_root = args.gym_root.resolve()
    source_paths = [
        "nemo_gym/web",
        "resources_servers/browsergym_web",
        "responses_api_agents/web_agent",
    ]
    manifest = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "kind": "deterministic_environment_smoke",
        "runtime": runtime,
        "task": TASK,
        "viewport": VIEWPORT,
        "headless": True,
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "packages": _package_versions(),
        "gym": {
            "root": str(gym_root),
            "commit": _git_value(gym_root, "rev-parse", "HEAD"),
            "branch": _git_value(gym_root, "branch", "--show-current"),
            "status": _git_value(gym_root, "status", "--short"),
            "web_source_tree_sha256": _source_tree_sha256(gym_root, source_paths),
        },
    }
    if args.webvoyager_root:
        webvoyager_root = args.webvoyager_root.resolve()
        manifest["webvoyager"] = {
            "root": str(webvoyager_root),
            "commit": _git_value(webvoyager_root, "rev-parse", "HEAD"),
            "status": _git_value(webvoyager_root, "status", "--short"),
            "source_tree_sha256": _source_tree_sha256(
                webvoyager_root,
                ["run.py", "utils.py", "utils_webarena.py", "prompts.py"],
            ),
        }
    if args.chromium_binary:
        manifest["chromium_binary"] = str(args.chromium_binary.resolve())
    return manifest


def _element_description(element: Any, index: int) -> dict[str, Any]:
    def attribute(name: str) -> str:
        return str(element.get_attribute(name) or "")

    return {
        "index": index,
        "tag": str(element.tag_name),
        "type": attribute("type"),
        "name": attribute("name"),
        "placeholder": attribute("placeholder"),
        "aria_label": attribute("aria-label"),
        "title": attribute("title"),
        "text": str(element.text or "")[:300],
    }


def _select_selenium_search_link(elements: list[Any]) -> tuple[int, list[dict[str, Any]]]:
    candidates = [_element_description(element, index) for index, element in enumerate(elements)]
    ranked = []
    for candidate in candidates:
        if candidate["tag"].lower() != "a":
            continue
        searchable = " ".join(str(candidate[key]) for key in ("aria_label", "title", "text")).lower()
        score = int(searchable.strip() == "search") * 20 + int("search" in searchable) * 10
        ranked.append((score, candidate["index"]))
    ranked = [item for item in ranked if item[0] > 0]
    if not ranked:
        raise RuntimeError("original WebVoyager SOM did not expose the ArXiv Search link")
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][1], candidates


def _select_selenium_search_input(elements: list[Any]) -> tuple[int, list[dict[str, Any]]]:
    candidates = [_element_description(element, index) for index, element in enumerate(elements)]
    ranked = []
    for candidate in candidates:
        if candidate["tag"].lower() not in {"input", "textarea"}:
            continue
        if candidate["type"] not in {"", "email", "search", "tel", "text"}:
            continue
        searchable = " ".join(
            str(candidate[key]) for key in ("type", "name", "placeholder", "aria_label", "title", "text")
        ).lower()
        score = int("query" in searchable) * 20 + int("search" in searchable) * 10
        ranked.append((score, candidate["index"]))
    if not ranked:
        raise RuntimeError("original WebVoyager SOM did not expose a search query input")
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][1], candidates


def _result_excerpt(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [line[:500] for line in lines if "simcse" in line.lower() or "result" in line.lower()][:30]


def run_selenium(args: argparse.Namespace, trace: Trace) -> dict[str, Any]:
    if args.webvoyager_root is None or args.chromium_binary is None:
        raise ValueError("selenium-original requires --webvoyager-root and --chromium-binary")
    sys.path.insert(0, str(args.webvoyager_root.resolve()))
    from run import exec_action_click, exec_action_type  # type: ignore[import-not-found]
    from selenium import webdriver
    from utils import get_web_element_rect  # type: ignore[import-not-found]

    options = webdriver.ChromeOptions()
    options.binary_location = str(args.chromium_binary.resolve())
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--force-device-scale-factor=1")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    )

    trace.emit("reset", step=0, data={"start_url": TASK["start_url"]})
    driver = webdriver.Chrome(options=options)
    try:
        driver.set_window_size(VIEWPORT["width"], VIEWPORT["height"])
        driver.get(TASK["start_url"])
        time.sleep(args.initial_wait)
        before_url = driver.current_url
        rects, elements, som_text = get_web_element_rect(driver, fix_color=True)
        selected_link, candidates = _select_selenium_search_link(elements)
        screenshot0 = trace.out_dir / "screenshots" / "step-0000.png"
        if not driver.save_screenshot(str(screenshot0)):
            raise RuntimeError("Selenium failed to save the initial screenshot")
        observation0 = {
            "url": before_url,
            "title": driver.title,
            "interactive_element_count": len(elements),
            "selected_search_link": selected_link,
            "elements": candidates,
            "som_text": som_text,
        }
        _write_json(trace.out_dir / "observations" / "step-0000.json", observation0)
        trace.emit(
            "observation",
            step=0,
            data={**observation0, "screenshot": trace.screenshot_ref(screenshot0)},
        )
        for marker in rects:
            driver.execute_script("arguments[0].remove()", marker)

        trace.emit(
            "action_parse",
            step=0,
            data={
                "source": "fixed_smoke_action",
                "raw_model_output": None,
                "benchmark_action": f"Click [{selected_link}]",
                "runtime_action": "WebVoyager.run.exec_action_click",
            },
        )
        action_started = time.monotonic()
        exec_action_click((str(selected_link),), elements[selected_link], driver)
        action0_ms = round((time.monotonic() - action_started) * 1000)
        search_url = driver.current_url
        trace.emit(
            "action_execute",
            step=0,
            data={
                "execution_ok": True,
                "duration_ms": action0_ms,
                "url_before": before_url,
                "url_after": search_url,
            },
        )

        rects, elements, som_text = get_web_element_rect(driver, fix_color=True)
        selected_input, candidates = _select_selenium_search_input(elements)
        screenshot1 = trace.out_dir / "screenshots" / "step-0001.png"
        if not driver.save_screenshot(str(screenshot1)):
            raise RuntimeError("Selenium failed to save the search-page screenshot")
        observation1 = {
            "url": search_url,
            "title": driver.title,
            "interactive_element_count": len(elements),
            "selected_search_input": selected_input,
            "elements": candidates,
            "som_text": som_text,
        }
        _write_json(trace.out_dir / "observations" / "step-0001.json", observation1)
        trace.emit(
            "observation",
            step=1,
            data={**observation1, "screenshot": trace.screenshot_ref(screenshot1)},
        )
        for marker in rects:
            driver.execute_script("arguments[0].remove()", marker)

        trace.emit(
            "action_parse",
            step=1,
            data={
                "source": "fixed_smoke_action",
                "raw_model_output": None,
                "benchmark_action": f"Type [{selected_input}]; [{TASK['query']}]",
                "runtime_action": "WebVoyager.run.exec_action_type",
            },
        )
        action_started = time.monotonic()
        warning = exec_action_type(
            {"number": str(selected_input), "content": TASK["query"]},
            elements[selected_input],
            driver,
        )
        action1_ms = round((time.monotonic() - action_started) * 1000)
        after_url = driver.current_url
        screenshot2 = trace.out_dir / "screenshots" / "step-0002.png"
        if not driver.save_screenshot(str(screenshot2)):
            raise RuntimeError("Selenium failed to save the final screenshot")
        body_text = driver.find_element("tag name", "body").text
        observation2 = {
            "url": after_url,
            "title": driver.title,
            "body_excerpt": _result_excerpt(body_text),
        }
        _write_json(trace.out_dir / "observations" / "step-0002.json", observation2)
        trace.emit(
            "action_execute",
            step=1,
            data={
                "execution_ok": True,
                "warning": warning,
                "duration_ms": action1_ms,
                "url_before": search_url,
                "url_after": after_url,
            },
        )
        trace.emit(
            "observation",
            step=2,
            data={**observation2, "screenshot": trace.screenshot_ref(screenshot2)},
        )
        return {
            "execution_ok": True,
            "initial_url": before_url,
            "final_url": after_url,
            "final_title": driver.title,
            "result_excerpt": observation2["body_excerpt"],
            "initial_interactive_element_count": observation0["interactive_element_count"],
            "selected_search_link": selected_link,
            "selected_search_input": selected_input,
        }
    finally:
        driver.quit()
        trace.emit("close", step=2, data={})


def _select_browsergym_search_link(axtree_text: str) -> tuple[str, list[str]]:
    candidates = [line for line in axtree_text.splitlines() if "search" in line.lower()]
    ranked = []
    for index, line in enumerate(candidates):
        match = re.search(r"\[([^\]]+)\]\s", line)
        if not match:
            continue
        normalized = line.lower()
        score = int("link" in normalized) * 10 + int("search" in normalized)
        ranked.append((score, index, match.group(1), line))
    ranked = [item for item in ranked if item[0] >= 10]
    if not ranked:
        raise RuntimeError("BrowserGym AXTree did not expose the ArXiv Search link with a bid")
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][2], candidates


def _select_browsergym_search_input(axtree_text: str) -> tuple[str, list[str]]:
    candidates = [line for line in axtree_text.splitlines() if "search" in line.lower() or "query" in line.lower()]
    ranked = []
    for index, line in enumerate(candidates):
        match = re.search(r"\[([^\]]+)\]\s", line)
        if not match:
            continue
        normalized = line.lower()
        score = int("textbox" in normalized) * 10 + int("search" in normalized) + int("query" in normalized)
        ranked.append((score, index, match.group(1), line))
    ranked = [item for item in ranked if item[0] >= 10]
    if not ranked:
        raise RuntimeError("BrowserGym AXTree did not expose a search query input with a bid")
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][2], candidates


def _copy_artifact(uri: str, target: Path) -> None:
    source = Path(uri.removeprefix("file://"))
    target.write_bytes(source.read_bytes())


def run_browsergym(args: argparse.Namespace, trace: Trace) -> dict[str, Any]:
    sys.path.insert(0, str(args.gym_root.resolve()))
    from nemo_gym.web.actions import parse_model_action
    from nemo_gym.web.models import WebActionProfile, WebBenchmark, WebObservationProfile, WebTask
    from resources_servers.browsergym_web.artifacts import WebArtifactStore
    from resources_servers.browsergym_web.backend import BrowserGymBackend
    from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig

    config = BrowserGymWebResourcesServerConfig(
        name="browsergym_web",
        host="localhost",
        port=8000,
        entrypoint="app.py",
        domain="agent",
        artifact_dir=str(trace.out_dir / "native-artifacts"),
        inline_screenshots=False,
        headless=True,
        pre_observation_delay=args.observation_delay,
        tags_to_mark="standard_html",
    )
    task = WebTask(
        benchmark=WebBenchmark.WEBVOYAGER,
        task_id=TASK["task_id"],
        intent=TASK["intent"],
        start_urls=[TASK["start_url"]],
        sites=["ArXiv"],
        observation_profile=WebObservationProfile.SOM,
        action_profile=WebActionProfile.WEBVOYAGER_LEGACY,
        task_kwargs={"env_kwargs": {"viewport": VIEWPORT}},
        original_metadata={"source": "official WebVoyager task ArXiv--13"},
    )
    backend = BrowserGymBackend(config, "arxiv13", WebArtifactStore(config.artifact_dir, inline_screenshots=False))
    trace.emit("reset", step=0, data={"start_url": TASK["start_url"]})
    try:
        observation0, seed_info = backend.reset(task)
        selected_link_bid, candidates = _select_browsergym_search_link(observation0.axtree_text)
        screenshot0 = trace.out_dir / "screenshots" / "step-0000.png"
        if observation0.screenshot is None or observation0.screenshot.artifact is None:
            raise RuntimeError("BrowserGym did not produce an initial screenshot artifact")
        _copy_artifact(observation0.screenshot.artifact.uri, screenshot0)
        observation0_json = {
            "url": observation0.url,
            "tabs": [tab.model_dump(mode="json") for tab in observation0.tabs],
            "selected_search_link_bid": selected_link_bid,
            "search_candidates": candidates,
            "axtree_text": observation0.axtree_text,
            "element_map_count": len(observation0.element_map),
            "last_action_error": observation0.last_action_error,
            "seed_info": seed_info,
        }
        _write_json(trace.out_dir / "observations" / "step-0000.json", observation0_json)
        trace.emit(
            "observation",
            step=0,
            data={
                "url": observation0.url,
                "tabs": observation0_json["tabs"],
                "selected_search_link_bid": selected_link_bid,
                "search_candidates": candidates,
                "axtree_line_count": len(observation0.axtree_text.splitlines()),
                "element_map_count": len(observation0.element_map),
                "screenshot": trace.screenshot_ref(screenshot0),
            },
        )

        action = parse_model_action(f"Click [{selected_link_bid}]", task.action_profile)
        trace.emit(
            "action_parse",
            step=0,
            data={
                "source": "fixed_smoke_action",
                "raw_model_output": None,
                "benchmark_action": f"Click [{selected_link_bid}]",
                "runtime_action": action.model_dump(mode="json"),
            },
        )
        action_started = time.monotonic()
        step0_result = backend.step(action)
        action0_ms = round((time.monotonic() - action_started) * 1000)
        observation1 = step0_result.observation
        trace.emit(
            "action_execute",
            step=0,
            data={
                "execution_ok": step0_result.execution_ok,
                "duration_ms": action0_ms,
                "url_before": observation0.url,
                "url_after": observation1.url,
                "last_action_error": observation1.last_action_error,
                "benchmark_reward": step0_result.benchmark_reward,
                "terminated": step0_result.terminated,
                "truncated": step0_result.truncated,
            },
        )

        selected_input_bid, candidates = _select_browsergym_search_input(observation1.axtree_text)
        screenshot1 = trace.out_dir / "screenshots" / "step-0001.png"
        if observation1.screenshot is None or observation1.screenshot.artifact is None:
            raise RuntimeError("BrowserGym did not produce a search-page screenshot artifact")
        _copy_artifact(observation1.screenshot.artifact.uri, screenshot1)
        observation1_json = {
            "url": observation1.url,
            "tabs": [tab.model_dump(mode="json") for tab in observation1.tabs],
            "axtree_text": observation1.axtree_text,
            "selected_search_input_bid": selected_input_bid,
            "search_candidates": candidates,
            "element_map_count": len(observation1.element_map),
            "last_action": observation1.last_action,
            "last_action_error": observation1.last_action_error,
            "benchmark_reward": step0_result.benchmark_reward,
            "terminated": step0_result.terminated,
            "truncated": step0_result.truncated,
            "info": step0_result.info,
        }
        _write_json(trace.out_dir / "observations" / "step-0001.json", observation1_json)
        trace.emit(
            "observation",
            step=1,
            data={
                "url": observation1.url,
                "tabs": observation1_json["tabs"],
                "selected_search_input_bid": selected_input_bid,
                "search_candidates": candidates,
                "axtree_line_count": len(observation1.axtree_text.splitlines()),
                "element_map_count": len(observation1.element_map),
                "screenshot": trace.screenshot_ref(screenshot1),
            },
        )

        action = parse_model_action(f"Type [{selected_input_bid}]; [{TASK['query']}]", task.action_profile)
        trace.emit(
            "action_parse",
            step=1,
            data={
                "source": "fixed_smoke_action",
                "raw_model_output": None,
                "benchmark_action": f"Type [{selected_input_bid}]; [{TASK['query']}]",
                "runtime_action": action.model_dump(mode="json"),
            },
        )
        action_started = time.monotonic()
        step1_result = backend.step(action)
        action1_ms = round((time.monotonic() - action_started) * 1000)
        observation2 = step1_result.observation
        screenshot2 = trace.out_dir / "screenshots" / "step-0002.png"
        if observation2.screenshot is None or observation2.screenshot.artifact is None:
            raise RuntimeError("BrowserGym did not produce a final screenshot artifact")
        _copy_artifact(observation2.screenshot.artifact.uri, screenshot2)
        observation2_json = {
            "url": observation2.url,
            "tabs": [tab.model_dump(mode="json") for tab in observation2.tabs],
            "axtree_text": observation2.axtree_text,
            "body_excerpt": _result_excerpt(observation2.axtree_text),
            "element_map_count": len(observation2.element_map),
            "last_action": observation2.last_action,
            "last_action_error": observation2.last_action_error,
            "benchmark_reward": step1_result.benchmark_reward,
            "terminated": step1_result.terminated,
            "truncated": step1_result.truncated,
            "info": step1_result.info,
        }
        _write_json(trace.out_dir / "observations" / "step-0002.json", observation2_json)
        trace.emit(
            "action_execute",
            step=1,
            data={
                "execution_ok": step1_result.execution_ok,
                "duration_ms": action1_ms,
                "url_before": observation1.url,
                "url_after": observation2.url,
                "last_action_error": observation2.last_action_error,
                "benchmark_reward": step1_result.benchmark_reward,
                "terminated": step1_result.terminated,
                "truncated": step1_result.truncated,
            },
        )
        trace.emit(
            "observation",
            step=2,
            data={
                "url": observation2.url,
                "tabs": observation2_json["tabs"],
                "body_excerpt": observation2_json["body_excerpt"],
                "axtree_line_count": len(observation2.axtree_text.splitlines()),
                "element_map_count": len(observation2.element_map),
                "screenshot": trace.screenshot_ref(screenshot2),
            },
        )
        return {
            "execution_ok": step0_result.execution_ok and step1_result.execution_ok,
            "initial_url": observation0.url,
            "final_url": observation2.url,
            "result_excerpt": observation2_json["body_excerpt"],
            "axtree_line_count_initial": len(observation0.axtree_text.splitlines()),
            "element_map_count_initial": len(observation0.element_map),
            "selected_search_link_bid": selected_link_bid,
            "selected_search_input_bid": selected_input_bid,
            "last_action_error": observation2.last_action_error,
        }
    finally:
        backend.close()
        trace.emit("close", step=2, data={})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("selenium-original", "browsergym-gym"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gym-root", type=Path, required=True)
    parser.add_argument("--webvoyager-root", type=Path)
    parser.add_argument("--chromium-binary", type=Path)
    parser.add_argument("--initial-wait", type=float, default=5.0)
    parser.add_argument("--observation-delay", type=float, default=3.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trace = Trace(args.out_dir, args.runtime)
    _write_json(args.out_dir / "manifest.json", _manifest(args, args.runtime))
    started = time.monotonic()
    try:
        if args.runtime == "selenium-original":
            result = run_selenium(args, trace)
        else:
            result = run_browsergym(args, trace)
        summary = {
            "status": "passed" if result.get("execution_ok") else "failed",
            "runtime": args.runtime,
            "task_id": TASK["task_id"],
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            **result,
        }
        _write_json(args.out_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.get("execution_ok") else 1
    except Exception as exc:  # noqa: BLE001 - preserve a structured failure artifact for comparison.
        traceback_text = traceback.format_exc()
        trace.emit(
            "error",
            step=0,
            data={"type": type(exc).__name__, "message": str(exc), "traceback": traceback_text},
        )
        summary = {
            "status": "error",
            "runtime": args.runtime,
            "task_id": TASK["task_id"],
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback_text,
        }
        _write_json(args.out_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
