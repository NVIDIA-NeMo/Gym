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

import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

from benchmarks.swebench.pro import prepare as prepare_module
from benchmarks.swebench.pro.prepare import UPSTREAM_COMMIT, enrich_row, fetch_image_digest, prepare


def make_upstream(root: Path, instance_id: str) -> None:
    assets = {
        f"run_scripts/{instance_id}/run_script.sh": "#!/bin/bash\n",
        f"run_scripts/{instance_id}/parser.py": "print('parser')\n",
        f"dockerfiles/base_dockerfile/{instance_id}/Dockerfile": "FROM base\n",
        f"dockerfiles/instance_dockerfile/{instance_id}/Dockerfile": "FROM instance\n",
    }
    for relative_path, contents in assets.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")


def dataset_row(instance_id: str = "instance_example", dockerhub_tag: str = "example-tag") -> dict:
    return {
        "repo": "example/repo",
        "instance_id": instance_id,
        "base_commit": "abc123",
        "patch": "patch",
        "test_patch": "",
        "problem_statement": "Fix it",
        "fail_to_pass": '["new_test"]',
        "pass_to_pass": '["old_test"]',
        "before_repo_set_cmd": "",
        "selected_test_files_to_run": '["tests"]',
        "dockerhub_tag": dockerhub_tag,
    }


def test_enrich_row_embeds_pinned_evaluator_assets(tmp_path) -> None:
    make_upstream(tmp_path, "instance_example")

    row = enrich_row(dataset_row(), tmp_path, "sha256:digest")

    assert row["run_script"] == "#!/bin/bash\n"
    assert row["parser_script"] == "print('parser')\n"
    assert row["base_dockerfile"] == "FROM base\n"
    assert row["image_digest"] == "sha256:digest"
    assert row["evaluator_commit"] == UPSTREAM_COMMIT
    assert row["responses_create_params"]["input"][0]["content"] == "Fix it"


def test_prepare_writes_self_contained_jsonl(tmp_path) -> None:
    make_upstream(tmp_path, "instance_example")
    output = tmp_path / "output.jsonl"

    result = prepare(
        dataset=[dataset_row()],
        upstream_root=tmp_path,
        output_fpath=output,
        image_digest_resolver=lambda _: "sha256:digest",
    )

    assert result == output
    prepared = json.loads(output.read_text(encoding="utf-8"))
    assert prepared["instance_id"] == "instance_example"
    assert prepared["parser_script"] == "print('parser')\n"


def test_fetch_image_digest_retries_docker_hub_rate_limit(monkeypatch) -> None:
    rate_limit = HTTPError("https://hub.docker.com", 429, "Too Many Requests", {"Retry-After": "0"}, None)
    response = BytesIO(b'{"digest": "sha256:digest"}')
    urlopen = iter([rate_limit, response])

    def open_response(*args, **kwargs):
        result = next(urlopen)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(prepare_module, "sleep", lambda _: None)
    monkeypatch.setattr(prepare_module.urllib.request, "urlopen", open_response)

    assert fetch_image_digest("example-tag") == "sha256:digest"


def test_prepare_recovers_digests_from_partial_output(tmp_path) -> None:
    make_upstream(tmp_path, "instance_example")
    make_upstream(tmp_path, "instance_second")
    output = tmp_path / "output.jsonl"
    output.write_text(
        json.dumps({"dockerhub_tag": "cached-tag", "image_digest": "sha256:cached"}) + "\n",
        encoding="utf-8",
    )
    resolved_tags = []

    prepare(
        dataset=[
            dataset_row(dockerhub_tag="cached-tag"),
            dataset_row("instance_second", "new-tag"),
        ],
        upstream_root=tmp_path,
        output_fpath=output,
        image_digest_resolver=lambda tag: resolved_tags.append(tag) or "sha256:new",
    )

    prepared = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert resolved_tags == ["new-tag"]
    assert [row["image_digest"] for row in prepared] == ["sha256:cached", "sha256:new"]
