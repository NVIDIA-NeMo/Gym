# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Sphinx directive to render environments from docs/data/environments.yaml.

Groups by server name, uses grid layout, schema-style expandable blocks.
Usage in MyST or RST:

    ```{environments}
    ```
"""

from collections import defaultdict
from html import escape
from pathlib import Path

import yaml
from docutils import nodes
from docutils.parsers.rst import Directive


GITHUB_REPO = "https://github.com/NVIDIA-NeMo/Gym"
BRANCH = "main"


class EnvironmentsDirective(Directive):
    """Render environments grouped by server, in a grid layout."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec = {}

    def run(self):
        env = self.state.document.settings.env
        srcdir = Path(env.srcdir)
        yaml_path = srcdir / "data" / "environments.yaml"

        if not yaml_path.exists():
            return [
                nodes.warning(
                    "",
                    nodes.paragraph(
                        text=f"environments.yaml not found at {yaml_path}. "
                        "Run: python scripts/generate_environments_yaml.py",
                    ),
                )
            ]

        with yaml_path.open() as f:
            data = yaml.safe_load(f)

        example = data.get("example", [])
        training = data.get("training", [])

        result = []

        if example:
            result.append(
                self._make_section(
                    "Example Environment Patterns",
                    example,
                    is_example=True,
                )
            )

        if training:
            result.append(
                self._make_section(
                    "Environments for Training & Evaluation",
                    training,
                    is_example=False,
                )
            )

        return result

    def _make_section(self, title: str, items: list, *, is_example: bool) -> nodes.section:
        """Build a section with grid layout."""
        section = nodes.section(ids=[nodes.make_id(title)])
        section += nodes.title("", title)

        html = self._render_example_grid(items) if is_example else self._render_training_grid(items)
        section += nodes.raw("", html, format="html")
        return section

    def _link(self, path: str, text: str) -> str:
        """Build GitHub link to repo file."""
        url = f"{GITHUB_REPO}/blob/{BRANCH}/{path}"
        return f'<a href="{escape(url)}">{escape(text)}</a>'

    def _render_example_grid(self, items: list) -> str:
        """Render example envs as a grid of cards."""
        cards = []
        for item in items:
            config_link = self._link(item["config_path"], item["config_filename"])
            readme_link = self._link(item["readme_path"], "README")
            cards.append(
                f"""
<div class="env-card env-card-example">
  <div class="env-card-header">
    <strong>{escape(item["name"])}</strong>
    <span class="type-badge type-object">example</span>
  </div>
  <div class="env-card-body">
    <p class="env-demonstrates">{escape(item.get("demonstrates", ""))}</p>
    <div class="env-card-links">
      {config_link} · {readme_link}
    </div>
  </div>
</div>"""
            )
        return f'<div class="env-grid env-grid-example">\n{"".join(cards)}\n</div>'

    def _group_by_server(self, items: list) -> dict[str, list]:
        """Group training items by server_name."""
        groups: dict[str, list] = defaultdict(list)
        for item in items:
            groups[item["server_name"]].append(item)
        return dict(groups)

    def _config_summary(self, item: dict) -> str:
        """Compact one-line summary for a config."""
        parts = []
        if item.get("domain"):
            parts.append(f'<span class="env-meta-item">{escape(item["domain"])}</span>')
        if item.get("train"):
            parts.append('<span class="env-pill env-pill-train">train</span>')
        if item.get("validation"):
            parts.append('<span class="env-pill env-pill-val">validation</span>')
        if item.get("license"):
            parts.append(f'<span class="env-meta-item">{escape(item["license"])}</span>')
        if not parts:
            return '<span class="env-meta-item">—</span>'
        return "".join(parts)

    def _config_full_details(self, item: dict) -> str:
        """Full expandable details for a config."""
        parts = []
        parts.append(
            f'<div class="env-config-detail">'
            f'<span class="env-detail-label">config</span> '
            f"{self._link(item['config_path'], item['config_filename'])}"
            f"</div>"
        )
        parts.append(
            f'<div class="env-config-detail">'
            f'<span class="env-detail-label">readme</span> '
            f"{self._link(item['readme_path'], 'README')}"
            f"</div>"
        )
        if item.get("domain"):
            parts.append(
                f'<div class="env-config-detail">'
                f'<span class="env-detail-label">domain</span> {escape(item["domain"])}'
                f"</div>"
            )
        if item.get("description"):
            parts.append(
                f'<div class="env-config-detail">'
                f'<span class="env-detail-label">description</span> {escape(item["description"])}'
                f"</div>"
            )
        if item.get("value"):
            parts.append(
                f'<div class="env-config-detail">'
                f'<span class="env-detail-label">value</span> {escape(item["value"])}'
                f"</div>"
            )
        if item.get("dataset"):
            ds = item["dataset"]
            parts.append(
                f'<div class="env-config-detail">'
                f'<span class="env-detail-label">dataset</span> '
                f'<a href="{escape(ds["url"])}">{escape(ds["name"])}</a>'
                f"</div>"
            )
        return "".join(parts)

    def _render_training_grid(self, items: list) -> str:
        """Render training envs grouped by server, in a grid."""
        groups = self._group_by_server(items)

        cards = []
        for server_name, configs in sorted(groups.items()):
            display_name = configs[0]["name"]
            domains = sorted({c.get("domain") for c in configs if c.get("domain")})
            domain_badges = " ".join(f'<span class="env-domain-badge">{escape(d)}</span>' for d in domains)
            config_count = len(configs)

            config_rows = []
            for cfg in sorted(configs, key=lambda c: c.get("config_filename", "")):
                summary = self._config_summary(cfg)
                details = self._config_full_details(cfg)
                config_link = self._link(cfg["config_path"], cfg["config_filename"])
                config_rows.append(
                    f'<details class="env-config-row">'
                    f'<summary class="env-config-summary">'
                    f'<div class="env-config-summary-inner">'
                    f'<span class="env-config-name">{config_link}</span>'
                    f'<span class="env-config-meta">{summary}</span>'
                    f"</div>"
                    f"</summary>"
                    f'<div class="env-config-details">{details}</div>'
                    f"</details>"
                )

            cards.append(
                f"""
<div class="env-card env-card-server">
  <div class="env-card-header">
    <strong>{escape(display_name)}</strong>
    <span class="env-config-count">{config_count} config{"" if config_count == 1 else "s"}</span>
  </div>
  <div class="env-card-domains">{domain_badges}</div>
  <div class="env-config-list">{"".join(config_rows)}</div>
</div>"""
            )

        return f'<div class="env-grid env-grid-server">\n{"".join(cards)}\n</div>'


def setup(app):
    """Register the environments directive and CSS."""
    app.add_directive("environments", EnvironmentsDirective)
    app.add_css_file("environments-spec.css")
    return {
        "version": "0.3",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
