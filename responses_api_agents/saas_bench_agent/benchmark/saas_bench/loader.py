# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/saas_bench/loader.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
"""Load tasks from the task directory and build the agent prompt.

Task directory layout (any task_dir containing description.md + meta.json counts;
the loader walks recursively, so flat or nested layouts both work):
  tasks_root/
    [<modality>/]<DOMAIN>/
      <task_id>/
        description.md   ← task description (Task Requirements + Steps + Login Credentials)
        meta.json        ← task_id, category_id, meta_data.sites, etc.
        verify.py        ← verification script
"""

import json
import re
from pathlib import Path


def load_tasks(tasks_root: str) -> list[dict]:
    """Recursively scan tasks_root and return every dir that has both description.md and meta.json.

    Each task dict:
      task_id        : str
      category_id    : str
      description_md : str  (full text of description.md)
      meta           : dict (parsed meta.json)
      verify_py_path : str  (absolute path to verify.py)
    """
    root = Path(tasks_root)
    tasks = []
    for meta_file in sorted(root.rglob("meta.json")):
        task_dir = meta_file.parent
        desc_file = task_dir / "description.md"
        if not desc_file.exists():
            continue
        verify_file = task_dir / "verify.py"
        try:
            meta = json.loads(meta_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[loader] skipping {task_dir.name}: failed to parse meta.json ({e})", flush=True)
            continue
        tasks.append({
            "task_id": meta.get("task_id", task_dir.name),
            "category_id": meta.get("category_id", task_dir.parent.name),
            "description_md": desc_file.read_text(),
            "meta": meta,
            "verify_py_path": str(verify_file) if verify_file.exists() else None,
        })
    return tasks


_STEP_LINE_RE = re.compile(r'^\s*\d+\.\s+(.*\S)\s*$')


def _extract_steps(description_md: str) -> list[str]:
    """Extract ordered list items under the **Steps:** section from description.md."""
    # Find the Steps section heading (tolerating various bold/heading styles)
    m = re.search(r'\*\*\s*Steps?\s*:\s*\*\*', description_md, re.IGNORECASE)
    if not m:
        m = re.search(r'^#+\s*Steps?\s*$', description_md, re.IGNORECASE | re.MULTILINE)
    if not m:
        return []
    tail = description_md[m.end():]
    # Stop at the next ** bold heading or top-level heading
    stop = re.search(r'\n\s*(\*\*[^*]+\*\*|#+\s+\S)', tail)
    if stop:
        tail = tail[: stop.start()]
    steps: list[str] = []
    for line in tail.splitlines():
        sm = _STEP_LINE_RE.match(line)
        if sm:
            steps.append(sm.group(1).strip())
    return steps


def _build_todo_md(task_id: str, steps: list[str]) -> str:
    """Generate a pre-filled todo.md."""
    lines = [
        f"# Task: {task_id}",
        "",
        "## Plan (you may adjust as needed)",
        "",
    ]
    if steps:
        for s in steps:
            lines.append(f"- [ ] {s}")
    else:
        lines.append("- [ ] (No structured steps detected; build your own plan from <user_request>.)")
    lines += [
        "",
        "## Notes",
        "- Mark items [x] when done, [-] when skipped.",
        "- Add new items if you discover sub-steps.",
        "- Do not block on missing required form fields — fill reasonable defaults and continue.",
        "",
    ]
    return "\n".join(lines)


def _build_url_block(port_map: dict[str, int], hostname: str) -> str:
    """Generate the Application Access URLs section (with strong-constraint wording)."""
    if not port_map:
        return ""
    lines = [
        "## Application Access URLs",
        "",
        "⚠️ CRITICAL — USE THESE EXACT URLs. Do NOT construct URLs from app names,",
        "brand documentation, or default port numbers (those are WRONG here).",
        "After landing on an app, navigate within it via UI clicks only.",
        "",
    ]
    for app, port in sorted(port_map.items()):
        lines.append(f"- {app}: http://{hostname}:{port}")
    lines.append("")
    return "\n".join(lines)


def build_prompt(
    task: dict,
    port_map: dict[str, int] | None = None,
    hostname: str = "localhost",
    tasks_root: str | None = None,
) -> tuple[str, str, list[str]]:
    """Build the agent's task prompt, the pre-filled todo.md, and the list of absolute paths to multimodal input files.

    Returns:
        (full_prompt, todo_md, input_files)
        input_files: list of absolute paths, passed to Agent(available_file_paths=...)
    """
    description = task["description_md"]
    steps = _extract_steps(description)

    parts: list[str] = []
    url_block = _build_url_block(port_map or {}, hostname)
    if url_block:
        parts.append(url_block)
    parts.append(description)

    full_prompt = "\n".join(parts).strip() + "\n"
    todo_md = _build_todo_md(task["task_id"], steps)

    # Resolve multimodal_input file paths to absolute paths.
    # Relative paths in meta are anchored at the repo root (=os.cwd, run.sh has already cd'd).
    input_files: list[str] = []
    for item in task.get("meta", {}).get("multimodal_input", []):
        rel = item.get("file", "")
        if not rel:
            continue
        p = Path(rel)
        if p.is_absolute():
            abs_path = str(p)
        else:
            abs_path = str(Path.cwd() / rel)
        if Path(abs_path).exists():
            input_files.append(abs_path)
        else:
            print(f"[loader] multimodal file not found: {abs_path}", flush=True)

    return full_prompt, todo_md, input_files
