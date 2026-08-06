# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate deterministic, hostable JSON and HTML environment catalogs."""

from __future__ import annotations

import argparse
import copy
import html
import json
import os
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


# Make ``nemo_gym`` importable when run directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemo_gym.environment_catalog import EnvironmentCatalog, discover_environment_catalog
from nemo_gym.repository_io import atomic_write_text


_DEFERRED_ISSUE_CODES = frozenset({"migration-draft", "ambiguous-legacy-resource"})


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _repo_relative_path(value: object, repo_root: Path) -> object:
    if not isinstance(value, str) or not value:
        return value
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def catalog_payload(catalog: EnvironmentCatalog, repo_root: Path) -> dict[str, Any]:
    """Return the CLI JSON record shape with clone-specific paths normalized."""

    payload = copy.deepcopy(catalog.to_json_dict())
    for entry in payload["entries"]:
        entry["config_path"] = _repo_relative_path(entry.get("config_path"), repo_root)
        entry["manifest_path"] = _repo_relative_path(entry.get("manifest_path"), repo_root)
    for issue in payload["issues"]:
        original_path = issue.get("path")
        normalized_path = _repo_relative_path(original_path, repo_root)
        message = issue.get("message")
        if isinstance(message, str) and isinstance(original_path, str) and isinstance(normalized_path, str):
            issue["message"] = message.replace(original_path, normalized_path)
        issue["path"] = normalized_path
    return payload


def render_catalog_json(payload: Mapping[str, Any]) -> str:
    """Serialize the shared catalog payload reproducibly."""

    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _display(value: object) -> str:
    if value is None or value == []:
        return "Not declared"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "Not declared"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _cell(value: object) -> str:
    return html.escape(_display(value), quote=True)


def render_catalog_html(payload: Mapping[str, Any], *, title: str = "NeMo Gym Environment Catalog") -> str:
    """Render every JSON catalog record into accessible, script-free HTML."""

    coverage = payload.get("coverage", {})
    entries = payload.get("entries", [])
    issues = payload.get("issues", [])
    rows: list[str] = []
    details: list[str] = []
    for entry in entries:
        entry_id = f"entry-{entry.get('kind', 'unknown')}-{entry['name']}"
        rows.append(
            "          <tr>\n"
            f'            <th scope="row"><a href="#{html.escape(str(entry_id), quote=True)}">'
            f"{_cell(entry['name'])}</a></th>\n"
            f"            <td>{_cell(entry.get('kind'))}</td>\n"
            f"            <td>{_cell(entry.get('status'))}</td>\n"
            f"            <td>{_cell(entry.get('domain'))}</td>\n"
            f"            <td>{_cell(entry.get('modality'))}</td>\n"
            f"            <td>{_cell(entry.get('licensing'))}</td>\n"
            f"            <td>{_cell(entry.get('integration_profile'))}</td>\n"
            f"            <td>{_cell(entry.get('description'))}</td>\n"
            "          </tr>"
        )
        facts = []
        for key in (
            "version",
            "lifecycle",
            "authors",
            "determinism",
            "required_capabilities",
            "source",
            "config_path",
            "manifest_path",
        ):
            label = key.replace("_", " ").title()
            facts.append(f"          <dt>{html.escape(label)}</dt><dd>{_cell(entry.get(key))}</dd>")
        details.append(
            f'      <article id="{html.escape(str(entry_id), quote=True)}">\n'
            f"        <h3>{_cell(entry['name'])}</h3>\n"
            f"        <p>{_cell(entry.get('description'))}</p>\n"
            "        <dl>\n" + "\n".join(facts) + "\n        </dl>\n"
            "      </article>"
        )

    issue_section = ""
    if issues:
        explicit_issues = [issue for issue in issues if issue.get("code") not in _DEFERRED_ISSUE_CODES]
        deferred_counts = Counter(
            str(issue.get("code")) for issue in issues if issue.get("code") in _DEFERRED_ISSUE_CODES
        )
        items = [
            f"          <li><strong>{_cell(issue.get('code'))}</strong> "
            f"<code>{_cell(issue.get('path'))}</code>: {_cell(issue.get('message'))}</li>"
            for issue in explicit_issues
        ]
        deferred_labels = {
            "migration-draft": "generated migration draft",
            "ambiguous-legacy-resource": "ambiguous legacy component",
        }
        deferred_parts = [
            f"{count} {deferred_labels[code]}{'s' if count != 1 else ''}"
            for code, count in sorted(deferred_counts.items())
        ]
        if deferred_parts:
            items.append(
                "          <li><strong>Migration diagnostics:</strong> "
                + html.escape(", ".join(deferred_parts))
                + ' omitted from this page; full records are available in <a href="catalog.json">catalog.json</a>.</li>'
            )
        issue_section = (
            '\n      <section aria-labelledby="catalog-issues">\n'
            '        <h2 id="catalog-issues">Catalog diagnostics</h2>\n'
            "        <ul>\n"
            f"{'\n'.join(items)}\n"
            "        </ul>\n"
            "      </section>"
        )

    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="color-scheme" content="light dark">
    <title>{safe_title}</title>
    <style>
      :root {{ font-family: system-ui, sans-serif; line-height: 1.5; }}
      body {{ margin: 0 auto; max-width: 90rem; padding: 1rem; }}
      .skip-link {{ left: -10000px; position: absolute; }}
      .skip-link:focus {{ left: 1rem; top: 1rem; }}
      .table-scroll {{ overflow-x: auto; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border: 1px solid currentColor; padding: .5rem; text-align: left; vertical-align: top; }}
      thead th {{ background: Canvas; position: sticky; top: 0; }}
      article {{ border-top: 1px solid currentColor; margin-top: 1rem; }}
      dl {{ display: grid; grid-template-columns: minmax(10rem, 1fr) 3fr; gap: .25rem 1rem; }}
      dt {{ font-weight: 700; }}
      dd {{ margin: 0; overflow-wrap: anywhere; }}
    </style>
  </head>
  <body>
    <a class="skip-link" href="#catalog">Skip to catalog</a>
    <header>
      <h1>{safe_title}</h1>
      <p>
        {int(coverage.get("with_manifest", 0))} of {int(coverage.get("total", 0))} runnable units have manifests
        ({float(coverage.get("percent", 0.0)):.1f}% coverage); {int(coverage.get("without_manifest", 0))}
        remain labelled no-manifest.
      </p>
    </header>
    <main id="catalog">
      <section aria-labelledby="catalog-table-title">
        <h2 id="catalog-table-title">Catalog entries ({len(entries)})</h2>
        <div class="table-scroll" role="region" aria-label="Environment catalog" tabindex="0">
          <table>
            <caption>Published and migration-era runnable Gym units</caption>
            <thead>
              <tr>
                <th scope="col">Name</th><th scope="col">Kind</th><th scope="col">Status</th>
                <th scope="col">Domain</th><th scope="col">Modality</th><th scope="col">License</th>
                <th scope="col">Profile</th><th scope="col">Description</th>
              </tr>
            </thead>
            <tbody>
{chr(10).join(rows)}
            </tbody>
          </table>
        </div>
      </section>
      <section aria-labelledby="entry-details-title">
        <h2 id="entry-details-title">Entry details</h2>
{chr(10).join(details)}
      </section>{issue_section}
    </main>
  </body>
</html>
"""


def write_catalog_artifacts(
    catalog: EnvironmentCatalog,
    *,
    repo_root: Path,
    json_output: Path,
    html_output: Path,
    title: str = "NeMo Gym Environment Catalog",
    check: bool = False,
) -> bool:
    """Write both views from one payload, or check that existing files match.

    Returns ``True`` when both destinations already contain the expected bytes.
    In normal write mode the return value therefore reports whether the call was
    a no-op.
    """

    payload = catalog_payload(catalog, repo_root)
    expected = {
        json_output: render_catalog_json(payload),
        html_output: render_catalog_html(payload, title=title),
    }
    unchanged = all(path.is_file() and path.read_text(encoding="utf-8") == value for path, value in expected.items())
    if not check:
        for path, value in expected.items():
            if not path.is_file() or path.read_text(encoding="utf-8") != value:
                atomic_write_text(path, value, create_parent=True)
    return unchanged


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate static HTML and the equivalent normalized `gym list catalog --json` payload."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path, required=True)
    parser.add_argument("--title", default="NeMo Gym Environment Catalog")
    parser.add_argument("--check", action="store_true", help="Fail instead of writing when artifacts are stale.")
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Return a non-zero status if manifest or publication-lock integrity issues were reported.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    json_output = args.json_output if args.json_output.is_absolute() else repo_root / args.json_output
    html_output = args.html_output if args.html_output.is_absolute() else repo_root / args.html_output
    with _working_directory(repo_root):
        catalog = discover_environment_catalog()
    unchanged = write_catalog_artifacts(
        catalog,
        repo_root=repo_root,
        json_output=json_output,
        html_output=html_output,
        title=args.title,
        check=args.check,
    )
    if args.check and not unchanged:
        print("Environment catalog artifacts are stale; regenerate them.", file=sys.stderr)
        return 1
    blocking_codes = {"invalid-manifest", "invalid-version-lock"}
    blocking_issues = [issue for issue in catalog.issues if issue.code in blocking_codes]
    if args.fail_on_issues and blocking_issues:
        print(f"Environment catalog contains {len(blocking_issues)} blocking integrity issue(s).", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "entries": len(catalog.entries),
                "html": str(html_output),
                "json": str(json_output),
                "unchanged": unchanged,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
