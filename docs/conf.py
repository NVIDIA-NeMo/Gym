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

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "NeMo Gym"
copyright = "2025, NVIDIA Corporation"
author = "NVIDIA Corporation"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # For our markdown docs
    "autodoc2",  # Generates API docs
    "sphinx.ext.viewcode",  # For adding a link to view source code in docs
    "sphinx.ext.doctest",  # Allows testing in docstrings
    "sphinx.ext.napoleon",  # For google style docstrings
    "sphinx_copybutton",  # For copy button in code blocks
    "sphinx_design",  # For grid layouts and card components
    "sphinx_reredirects",  # For URL redirects when pages move
    "sphinxcontrib.mermaid",  # For mermaid diagrams
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "rfc"]

# -- Options for MyST Parser (Markdown) --------------------------------------
# MyST Parser settings
myst_enable_extensions = [
    "dollarmath",  # Enables dollar math for inline math
    "amsmath",  # Enables LaTeX math for display mode
    "colon_fence",  # Enables code blocks using ::: delimiters instead of ```
    "deflist",  # Supports definition lists with term: definition format
    "fieldlist",  # Enables field lists for metadata like :author: Name
    "tasklist",  # Adds support for GitHub-style task lists with [ ] and [x]
    "substitution",  # Enables variable substitutions like {{product_name}}
]
myst_heading_anchors = 5  # Generates anchor links for headings up to level 5

# MyST substitutions - variables that can be used in markdown files
myst_substitutions = {
    "product_name": "NeMo Gym",
}

# -- Options for Autodoc2 ---------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

autodoc2_packages = [
    "../nemo_gym",  # Path to your package relative to conf.py
]
autodoc2_render_plugin = "myst"  # Use MyST for rendering docstrings
autodoc2_output_dir = "apidocs"  # Output directory for autodoc2 (relative to docs/)
# This is a workaround that uses the parser located in autodoc2_docstrings_parser.py to allow autodoc2 to
# render google style docstrings.
# Related Issue: https://github.com/sphinx-extensions2/sphinx-autodoc2/issues/33
autodoc2_docstring_parser_regexes = [
    (r".*", "docs.autodoc2_docstrings_parser"),
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nvidia_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA-NeMo/Gym",
            "icon": "fa-brands fa-github",
        }
    ],
    "switcher": {
        "json_url": "../versions1.json",
        "version_match": release,
    },
    "extra_head": {
        """
    <script src="https://assets.adobedtm.com/5d4962a43b79/c1061d2c5e7b/launch-191c2462b890.min.js" ></script>
    """
    },
    "extra_footer": {
        """
    <script type="text/javascript">if (typeof _satellite !== "undefined") {_satellite.pageBottom();}</script>
    """
    },
}
html_extra_path = ["project.json", "versions1.json"]

# -- Options for sphinx-reredirects ------------------------------------------
# https://documatt.com/sphinx-reredirects/
#
# Use this to create redirects when pages are moved or renamed.
# Format: "old/path.html": "target-relative-to-source.html"
#
# IMPORTANT: Target paths are RELATIVE to the source file's directory!
#   - Same directory: "dir/old.html": "new.html"
#   - Cross directory: "old-dir/page.html": "../new-dir/page.html"
#   - External redirect: "old-page.html": "https://example.com/new-page"
#
# The .html extension is required for source paths.

redirects = {
    # Get Started section renames (same directory)
    "get-started/setup-installation.html": "detailed-setup.html",
    # RL Framework Integration moved from training/ to contribute/
    # Source is in training/rl-framework-integration/, target is in contribute/rl-framework-integration/
    "training/rl-framework-integration/index.html": "../../contribute/rl-framework-integration/index.html",
    "training/rl-framework-integration/gym-integration-footprint-and-form-factor.html": "../../contribute/rl-framework-integration/gym-integration-footprint-and-form-factor.html",
    "training/rl-framework-integration/gym-rl-framework-integration-success-criteria.html": "../../contribute/rl-framework-integration/gym-rl-framework-integration-success-criteria.html",
    "training/rl-framework-integration/generation-backend-and-openai-compatible-http-server.html": "../../contribute/rl-framework-integration/generation-backend-and-openai-compatible-http-server.html",
    "training/rl-framework-integration/openai-compatible-http-server-on-policy-correction.html": "../../contribute/rl-framework-integration/openai-compatible-http-server-on-policy-correction.html",
    # Alternate naming conventions for the same pages
    "training/rl-framework-integration/integration-footprint.html": "../../contribute/rl-framework-integration/gym-integration-footprint-and-form-factor.html",
    "training/rl-framework-integration/on-policy-corrections.html": "../../contribute/rl-framework-integration/openai-compatible-http-server-on-policy-correction.html",
    # About/Concepts section renames (same directory)
    "about/concepts/core-abstractions.html": "core-components.html",
    "about/concepts/configuration-system.html": "configuration.html",
    # Top-level page moves (from root to reference/)
    "how-to-faq.html": "reference/faq.html",
    # Infrastructure engineering notes moved to subdirectory
    "infrastructure/swe-rl-case-study.html": "engineering-notes/swe-rl-case-study.html",
    "infrastructure/aiohttp-vs-httpx.html": "engineering-notes/aiohttp-vs-httpx.html",
}
