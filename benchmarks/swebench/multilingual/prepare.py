# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prepare SWE Bench Multilingual benchmark data for NeMo Gym."""

import json
from pathlib import Path

from datasets import load_dataset

from nemo_gym.global_config import get_hf_token


BENCHMARK_DIR = Path(__file__).parent.parent
DATA_DIR = BENCHMARK_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FPATH = DATA_DIR / "swebench_multilingual_benchmark.jsonl"

# Copied from `swebench.harness.constants.MAP_REPO_TO_EXT`
MAP_REPO_TO_EXT = {
    "redis/redis": "c",
    "jqlang/jq": "c",
    "nlohmann/json": "c",
    "micropython/micropython": "c",
    "valkey-io/valkey": "c",
    "fmtlib/fmt": "c",
    "caddyserver/caddy": "go",
    "hashicorp/terraform": "go",
    "prometheus/prometheus": "go",
    "gohugoio/hugo": "go",
    "gin-gonic/gin": "go",
    "google/gson": "java",
    "apache/druid": "java",
    "javaparser/javaparser": "java",
    "projectlombok/lombok": "java",
    "apache/lucene": "java",
    "reactivex/rxjava": "java",
    "Automattic/wp-calypso": "js",
    "chartjs/Chart.js": "js",
    "markedjs/marked": "js",
    "processing/p5.js": "js",
    "diegomura/react-pdf": "js",
    "babel/babel": "js",
    "vuejs/core": "js",
    "facebook/docusaurus": "js",
    "immutable-js/immutable-js": "js",
    "mrdoob/three.js": "js",
    "preactjs/preact": "js",
    "axios/axios": "js",
    "phpoffice/phpspreadsheet": "php",
    "laravel/framework": "php",
    "php-cs-fixer/php-cs-fixer": "php",
    "briannesbitt/carbon": "php",
    "astropy/astropy": "py",
    "dbt-labs/dbt-core": "py",
    "django/django": "py",
    "matplotlib/matplotlib": "py",
    "marshmallow-code/marshmallow": "py",
    "mwaskom/seaborn": "py",
    "pallets/flask": "py",
    "psf/requests": "py",
    "pvlib/pvlib-python": "py",
    "pydata/xarray": "py",
    "pydicom/pydicom": "py",
    "pylint-dev/astroid": "py",
    "pylint-dev/pylint": "py",
    "pytest-dev/pytest": "py",
    "pyvista/pyvista": "py",
    "scikit-learn/scikit-learn": "py",
    "sphinx-doc/sphinx": "py",
    "sqlfluff/sqlfluff": "py",
    "swe-bench/humaneval": "py",
    "sympy/sympy": "py",
    "jekyll/jekyll": "rb",
    "fluent/fluentd": "rb",
    "fastlane/fastlane": "rb",
    "jordansissel/fpm": "rb",
    "faker-ruby/faker": "rb",
    "rubocop/rubocop": "rb",
    "burntsushi/ripgrep": "rs",
    "sharkdp/bat": "rs",
    "astral-sh/ruff": "rs",
    "tokio-rs/tokio": "rs",
    "uutils/coreutils": "rs",
    "nushell/nushell": "rs",
    "tokio-rs/axum": "rs",
}


def prepare():
    ds = load_dataset("SWE-bench/SWE-bench_Multilingual", split="test", token=get_hf_token())

    prompt_template = Path("benchmarks/swebench/minimax_prompt.txt").read_text()

    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for row in ds:
            prompt = (
                prompt_template.replace(
                    "{{ workspace_path }}",
                    "/testbed",
                )
                .replace(
                    "{{ instance.problem_statement }}",
                    row["problem_statement"],
                )
                .replace(
                    "{{ instance.repo_language ~ ' ' if instance.repo_language else '' }}",
                    MAP_REPO_TO_EXT[row["repo"]] + " ",
                )
            )

            row = row | {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                },
                "subset": "multilingual",
                "split": "test",
                # SWE Bench Multilingual doesn't have/use this
                "environment_setup_commit": "",
                "difficulty": "",
            }
            # Minor tweaks on the formatting to match that of SWE Bench
            row["FAIL_TO_PASS"] = json.dumps(row["FAIL_TO_PASS"])
            row["PASS_TO_PASS"] = json.dumps(row["PASS_TO_PASS"])

            fout.write(json.dumps(row) + "\n")

    print(f"Wrote {len(ds)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
