# SPDX-FileCopyrightText: Copyright (c) 2026 Blobfish AI. All rights reserved.
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
"""Emit a NeMo Gym contribution in NVIDIA's current repository layout.

Two halves, mirroring how NVIDIA-NeMo/Gym is organised today:

* ``resources_servers/blobfish_enterprise_ops/``  the server code (``app.py``, the vendored
  ``blobfish_nemo_gym`` package with NVIDIA's SPDX header, tests, requirements, README).
* ``environments/blobfish_<suite>/``  one manifest-backed catalog entry per suite:
  ``manifest.yaml``, ``config.yaml`` (resources server instance + ``simple_agent``), ``README.md``,
  and ``data/example.jsonl``. Train and validation splits are not committed; the config points at
  a Hugging Face dataset and the same files are written to ``<out_dir>/hf_upload/<suite>/`` for
  publishing.

Task packages themselves are never copied. ``suite_dir`` resolves from an environment variable
with a default under ``data/suites/``, and the README documents the one-line Harbor download.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from .dataset import build_rows, split_rows, write_jsonl
from .suite import Suite, package_digest


__all__ = ["DATA_LICENSE_LABEL", "DEFAULT_HF_REPO", "SERVER_NAME", "attributions_entry", "emit_contribution", "slug"]

SERVER_NAME = "blobfish_enterprise_ops"
DATA_LICENSE_LABEL = "Creative Commons Attribution 4.0 International"  # NeMo Gym's enum spelling
CODE_LICENSE_LABEL = "Apache-2.0"
DEFAULT_HF_REPO = "johnnychien/blobfish-nemo-gym-enterprise-ops"
VENDORED_MODULES = (
    "__init__.py",
    "suite.py",
    "profiles.py",
    "dataset.py",
    "server.py",
    "fixture.py",
    "cli.py",
    "emit.py",
    "__main__.py",
)

SPDX_HEADER = """# SPDX-FileCopyrightText: Copyright (c) 2026 Blobfish AI. All rights reserved.
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


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


HARBOR_DATASETS = {"erpbench_100": "erpbench-100-suite"}  # published under a different slug than the suite


def _harbor_dataset(suite_slug: str) -> str:
    return HARBOR_DATASETS.get(suite_slug, suite_slug.replace("_", "-"))


_WORLD_KIND = {
    "runtime": "SQLite-backed business systems with transactional writes and a full call trace",
    "image_framework": "SQLite-backed multi-system world (ERP, books, docs, email, filings, sheets) with a call trace",
    "image_documents": "provider documents and objects (matter management, mail, drive, chat) with an append-only trace",
    "image_package": "a filesystem of documents plus CRM, marketing and call-intelligence records with a mutation trace",
    "image_companion": "SQLite-backed engineering systems (tickets, on-call, deploys, monitoring) with a call trace",
}


def _family_description(default_suite: Suite) -> str:
    return (
        "Blobfish enterprise-operations suites: stateful tasks in isolated business systems, graded by "
        "deterministic state-and-trace verifiers with no LLM judge. This default instance serves "
        f"{default_suite.name}; each suite has its own entry under environments/blobfish_*."
    )


def _description(suite: Suite) -> str:
    kind = _WORLD_KIND.get(suite.tasks[0].profile, "stateful business systems with a call trace")
    return (
        f"{suite.name}: {len(suite)} stateful enterprise-operations tasks over {len(suite.tools)} tools; "
        f"{kind}; graded by a deterministic state-and-trace verifier with no LLM judge."
    )


def _value(suite: Suite) -> str:
    return (
        "Improve long-horizon, multi-system tool use in back-office work: investigate before writing, "
        "keep unrelated records unchanged, read writes back, and submit exact answers."
    )


def _config_yaml(
    suite: Suite,
    env_name: str,
    suite_slug: str,
    env_var: str,
    hf_repo: str,
    *,
    description: str | None = None,
    data_dir: str | None = None,
) -> str:
    longest = max((len(task.oracle_steps or ()) for task in suite.tasks), default=0)
    max_steps = max(120, 2 * longest)
    data_dir = data_dir or f"environments/{env_name}/data"
    return f"""{env_name}:
  resources_servers:
    {SERVER_NAME}:
      entrypoint: app.py
      domain: agent
      verified: false
      description: >-
        {description or _description(suite)}
      value: >-
        {_value(suite)}
      # Task packages come from the Harbor Hub (one command, no source repos):
      #   harbor datasets download blobfishai/{_harbor_dataset(suite_slug)} -o data/suites/{_harbor_dataset(suite_slug)}
      # They are executable code run in-process; suite_manifest pins every package to the reviewed
      # SHA-256 and the server refuses anything else.
      suite_dir: ${{oc.env:{env_var},data/suites/{_harbor_dataset(suite_slug)}/{_harbor_dataset(suite_slug)}}}
      suite_manifest: {data_dir}/suite_manifest.json
      # fraction (criterion fraction), strict (all criteria), or gated (fraction x earned share of the
      # suite's outcome families: a wrong final write then loses most of the reward). Use gated for RL.
      reward_mode: fraction
      max_open_episodes: 4096
      episode_ttl_seconds: 7200
{env_name}_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      # Longest oracle plan in this suite: {longest} tool calls; the cap ends runaway loops (simple_agent has none by default).
      max_steps: {max_steps}
      resources_server:
        type: resources_servers
        name: {env_name}
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        jsonl_fpath: {data_dir}/train.jsonl
        source:
          type: huggingface
          repo_id: {hf_repo}
          artifact_fpath: {suite_slug}/train.jsonl
        license: {DATA_LICENSE_LABEL}
      - name: validation
        type: validation
        jsonl_fpath: {data_dir}/validation.jsonl
        source:
          type: huggingface
          repo_id: {hf_repo}
          artifact_fpath: {suite_slug}/validation.jsonl
        license: {DATA_LICENSE_LABEL}
      - name: example
        type: example
        jsonl_fpath: {data_dir}/example.jsonl
        license: {DATA_LICENSE_LABEL}
"""


def _manifest_yaml(suite: Suite, env_name: str) -> str:
    # NeMo Gym mirrors resources_server/agent_server (the implementations, not the instance names)
    # and the dataset list from config.yaml; validation reads every listed file, so the HF-backed
    # train and validation splits must be downloaded before `gym env validate`, as for NVIDIA's own
    # HF-backed environments.
    data = f"environments/{env_name}/data"
    return f"""name: {env_name}
version: 1.0.0
kind: environment
integration_profile: custom-gym-verifier
domain: agent
description: >-
  {_description(suite)}
modality: text
licensing: {CODE_LICENSE_LABEL} AND CC-BY-4.0
authors:
- blobfishai
reward:
  range:
  - 0.0
  - 1.0
  higher_is_better: true
determinism: seeded
resources_server: {SERVER_NAME}
agent_server: simple_agent
datasets:
- name: train
  type: train
  jsonl_fpath: {data}/train.jsonl
  num_repeats: 1
- name: validation
  type: validation
  jsonl_fpath: {data}/validation.jsonl
  num_repeats: 1
- name: example
  type: example
  jsonl_fpath: {data}/example.jsonl
  num_repeats: 1
model_server: policy_model
session_model: episode
state: per_session
lifecycle: active
"""


def _profile_section(suite: Suite, calibration_table: str | None) -> str:
    """The reward-profile rows for this suite only, or an empty string."""
    if not calibration_table:
        return ""
    lines = calibration_table.strip().splitlines()
    header, rows = lines[:2], [line for line in lines[2:] if line.startswith(f"| {suite.name} |")]
    if not rows:
        return ""
    return (
        "\n## Reward profile\n\nValidation split, two repeats, temperature 0, step-capped `simple_agent`.\n\n"
        + "\n".join(header + rows)
        + "\n"
    )


def _env_readme(
    suite: Suite,
    env_name: str,
    suite_slug: str,
    env_var: str,
    hf_repo: str,
    example_count: int,
    calibration_table: str | None = None,
) -> str:
    harbor = _harbor_dataset(suite_slug)
    tools = ", ".join(tool.public_name for tool in suite.tools[:6])
    return f"""# Description

{_description(suite)}

Each task drops the policy into an isolated company: {len(suite.tools)} tools such as `{tools}`,
one assignment written the way a manager would write it, and world state that persists across the
episode. Grading is deterministic: the package's own verifier scores the final state and the call
trace (outcome, required investigation before writes, readbacks, containment of unrelated records).
Reward is the criterion fraction (`reward_mode: gated` scales it by the earned share of the
suite's outcome families); `strict_pass` requires every criterion and is rare by design, so the
dense fraction is the training signal.
{_profile_section(suite, calibration_table)}
# Example usage

Create an `env.yaml` in the Gym root with `policy_base_url`, `policy_model_name`, `policy_api_key`,
and `hf_token` (the train and validation splits resolve from Hugging Face).

## Getting the task packages

```bash
harbor datasets download blobfishai/{harbor} -o data/suites/{harbor}
export {env_var}=$PWD/data/suites/{harbor}/{harbor}   # optional; this is the config default
```

Packages are executable code run in-process by the server; `data/suite_manifest.json` pins each
one to its reviewed SHA-256 and the server refuses to start on anything else.

## Running servers

```bash
gym env start --environment {env_name} --model-type openai_model
```

## Collecting rollouts

```bash
gym eval run --no-serve --agent {env_name}_simple_agent \
    --input environments/{env_name}/data/example.jsonl \
    --output results/{env_name}_example_rollouts.jsonl --num-repeats 1
```

`data/example.jsonl` carries {example_count} validation rows; `data/example_rollouts.jsonl` is one
DeepSeek rollout per example row.

# Licensing

Tasks are synthetic, generated by Blobfish AI's factory from real enterprise structure and qualified
with oracle replay and adversarial negative controls; no customer data. Models assisted with some
task text; verifiers are hand-written deterministic code. Task data: {DATA_LICENSE_LABEL} (Harbor
Hub `blobfishai/{harbor}` and Hugging Face `{hf_repo}`). Server code: Apache-2.0, Copyright (c) 2026
Blobfish AI, vendored per the repository's third-party code rules (see the server's VENDORING.md).
"""


def _server_readme(suites: list[tuple[Suite, str, str]], calibration_table: str | None = None) -> str:
    del (
        suites,
        calibration_table,
    )  # the server README describes the server only; suites live in their environment entries
    return f"""# Description

A resources server that hosts a published Blobfish enterprise-operations suite in-process: per-rollout
worlds built by the task package's own runtime, `POST /<tool>` dispatch, and `/verify` through the
package's own deterministic verifier (no LLM judge). Each suite is a separate environment entry under
`environments/blobfish_*` that points this server at its task packages through `suite_dir`; the default
instance in `configs/{SERVER_NAME}.yaml` serves the first contributed suite.

Task packages are executable code and are pinned by SHA-256 (`suite_manifest.json` next to each
environment); the server refuses packages that differ from the reviewed ones.

Reward modes (`reward_mode` in the server config): `fraction` (criterion fraction, the default),
`strict` (all criteria), and `gated` (fraction times the earned share of the suite's outcome
families, measured per suite by perturbing each oracle write, so process without outcomes is not
rewarded). `strict_pass`, `criterion_fraction` and `outcome_fraction` are always reported.

# Example usage

See the environment README (`environments/blobfish_*/README.md`) for the download, run and rollout
commands.

## Tests

```bash
pytest resources_servers/{SERVER_NAME}/tests
gym env test <environment>   # onboarding verifier cases: full, zero, malformed, determinism
```

`tests/fixtures/mini_suite` is a self-contained task package, so the seed, call and verify path and
the verifier fixture (`tests/verifier_cases.jsonl`) run in CI without any download; a downloaded
suite is exercised when its `BLOBFISH_<SUITE>_DIR` is set.

# Licensing

Server code: Apache-2.0, Copyright (c) 2026 Blobfish AI (vendored; see VENDORING.md). Task data:
{DATA_LICENSE_LABEL}.
"""


def _vendoring_md(suites: list[tuple[Suite, str, str]]) -> str:
    names = ", ".join(suite.name for suite, _, _ in suites)
    return f"""# Vendored: Blobfish AI NeMo Gym adapter

This directory is a **vendored copy** of Blobfish AI's `blobfish-nemo-gym-adapter`
(`packages/nemo-gym-adapter` in https://github.com/blobfishai/blobfishai), contributed by
Blobfish AI to NeMo Gym.

- **Upstream copyright:** Copyright (c) 2026 Blobfish AI. All rights reserved.
- **License:** Apache License, Version 2.0 (same as NeMo Gym).
- **Task data served by this adapter:** the Blobfish suites on the Harbor Hub ({names}),
  Creative Commons Attribution 4.0 International; prompt splits on Hugging Face under the
  same licence. The task packages are downloaded at run time and are not vendored here.

## License-header policy

Every file under `blobfish_nemo_gym/` keeps the Blobfish AI notice verbatim. NVIDIA
modifications, if any, add the standard NVIDIA modifications block below it, following the
repository's vendoring convention. This attribution is also recorded in the repository-level
`ATTRIBUTIONS.md` under "Vendored Components".
"""


def attributions_entry(suites: list[tuple[Suite, str, str]]) -> str:
    """The ATTRIBUTIONS.md paragraph NVIDIA's vendoring rule asks for (pasted into the PR)."""
    names = ", ".join(suite.name for suite, _, _ in suites)
    return (
        "### blobfish-nemo-gym-adapter (resources_servers/blobfish_enterprise_ops)\n\n"
        "- Upstream: https://github.com/blobfishai/blobfishai (packages/nemo-gym-adapter)\n"
        "- Copyright (c) 2026 Blobfish AI. All rights reserved.\n"
        "- License: Apache-2.0. Vendored and contributed by Blobfish AI; original notices preserved.\n"
        f"- Data served (not vendored): Blobfish task suites on the Harbor Hub ({names}), CC-BY-4.0.\n"
    )


def _app_py() -> str:
    return (
        SPDX_HEADER
        + '''"""NeMo Gym entrypoint for Blobfish enterprise-operations suites."""

from pathlib import Path

from resources_servers.blobfish_enterprise_ops.blobfish_nemo_gym.fixture import verifier_fixture
from resources_servers.blobfish_enterprise_ops.blobfish_nemo_gym.server import BlobfishSuiteResourcesServer


# `gym env test <environment>` scores the cases in tests/verifier_cases.jsonl on the self-contained
# fixture suite (tests/fixtures/mini_suite) through the same code path as /verify.
VERIFIER_FIXTURE = verifier_fixture(Path(__file__).resolve().parent / "tests")

if __name__ == "__main__":
    BlobfishSuiteResourcesServer.run_webserver()
'''
    )


def _test_py(suites: list[tuple[Suite, str, str]]) -> str:
    cases = "\n".join(
        f'    ("{env_name}", "BLOBFISH_{suite_slug.upper()}_DIR"),' for _, env_name, suite_slug in suites
    )
    return (
        SPDX_HEADER
        + f"""import asyncio
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.blobfish_enterprise_ops.blobfish_nemo_gym.dataset import build_row
from resources_servers.blobfish_enterprise_ops.blobfish_nemo_gym.server import (
    BlobfishSuiteResourcesServer,
    BlobfishSuiteResourcesServerConfig,
)
from resources_servers.blobfish_enterprise_ops.blobfish_nemo_gym.suite import Suite, public_tool_name


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_suite"
SUITES = [
{cases}
]


def _server(suite_dir):
    config = BlobfishSuiteResourcesServerConfig(
        host="0.0.0.0", port=8080, entrypoint="", name="blobfish_test", suite_dir=str(suite_dir)
    )
    return BlobfishSuiteResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _response():
    return {{
        "id": "resp_test",
        "created_at": time.time(),
        "model": "test",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }}


class TestApp:
    def test_seed_call_verify_on_the_fixture(self):
        suite = Suite.load(FIXTURE)
        task = suite.get("mini-001")
        server = _server(FIXTURE)
        row = build_row(task, suite.name)
        with TestClient(server.setup_webserver()) as client:
            assert client.post("/seed_session", json=row).status_code == 200
            for step in task.oracle_steps:
                result = client.post(f"/{{public_tool_name(step['tool'])}}", json=step.get("arguments") or {{}})
                assert result.status_code == 200 and "error" not in result.json(), result.text
            body = client.post("/verify", json=row | {{"response": _response()}}).json()
            assert body["reward"] == pytest.approx(1.0) and body["strict_pass"] == 1.0
        with TestClient(server.setup_webserver()) as client:  # no work: dense reward stays low, strict fails
            assert client.post("/seed_session", json=row).status_code == 200
            body = client.post("/verify", json=row | {{"response": _response()}}).json()
            assert body["strict_pass"] == 0.0 and body["reward"] < 1.0

    def test_onboarding_verifier_fixture(self):
        exercise = pytest.importorskip("nemo_gym.verifier_fixture").exercise_verifier_fixture
        from resources_servers.blobfish_enterprise_ops.app import VERIFIER_FIXTURE

        results = asyncio.run(
            exercise(VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="seeded")
        )
        assert [result.kind for result in results] == ["full_reward", "zero_reward", "malformed", "determinism"]

    @pytest.mark.parametrize("env_name,env_var", SUITES)
    def test_downloaded_suite_loads(self, env_name, env_var):
        suite_dir = os.environ.get(env_var)
        if not suite_dir or not Path(suite_dir).exists():
            pytest.skip(f"set {{env_var}} to a downloaded suite for {{env_name}}")
        server = _server(suite_dir)
        assert len(server.suite) > 0
        assert server.setup_webserver() is not None
"""
    )


def _vendor_module(source: Path, target: Path) -> None:
    text = source.read_text(encoding="utf-8")
    if text.startswith("#!"):
        first, _, rest = text.partition("\n")
        text = first + "\n" + SPDX_HEADER + rest
    else:
        text = SPDX_HEADER + text
    target.write_text(text, encoding="utf-8")


def _rollouts_for(rollouts_dir: Path | None, suite_slug: str, task_ids: list[str]) -> list[dict[str, Any]]:
    """One recorded rollout per example task id, from the first rollouts file for the suite."""
    if rollouts_dir is None:
        return []
    for path in sorted(Path(rollouts_dir).glob(f"{suite_slug}__*.jsonl")):
        if path.name.endswith(("_failures.jsonl", "_materialized_inputs.jsonl")):
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        picked: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row.get("task_id") in task_ids and row["task_id"] not in picked:
                picked[row["task_id"]] = row
        if len(picked) == len(task_ids):
            return [picked[t] for t in task_ids]
    return []


def emit_contribution(
    out_dir: Path,
    suite_dirs: Iterable[Path],
    *,
    example_count: int = 5,
    hf_repo: str = DEFAULT_HF_REPO,
    rollouts_dir: Path | None = None,
    calibration_readme: Path | None = None,
) -> list[Path]:
    out_dir = Path(out_dir)
    server_dir = out_dir / "resources_servers" / SERVER_NAME
    written: list[Path] = []

    package_src = Path(__file__).resolve().parent
    vendored = server_dir / "blobfish_nemo_gym"
    vendored.mkdir(parents=True, exist_ok=True)
    for name in VENDORED_MODULES:
        target = vendored / name
        _vendor_module(package_src / name, target)
        written.append(target)
    (server_dir / "tests").mkdir(exist_ok=True)
    (server_dir / "configs").mkdir(exist_ok=True)
    (server_dir / "data").mkdir(exist_ok=True)
    fixture_src = package_src.parent / "tests" / "fixtures" / "mini_suite"
    if fixture_src.exists():
        fixture_dst = server_dir / "tests" / "fixtures" / "mini_suite"
        if fixture_dst.exists():
            shutil.rmtree(fixture_dst)
        shutil.copytree(fixture_src, fixture_dst, ignore=shutil.ignore_patterns("__pycache__"))
        written.extend(p for p in fixture_dst.rglob("*") if p.is_file())
    cases_src = package_src.parent / "tests" / "verifier_cases.jsonl"
    if cases_src.exists():
        shutil.copyfile(cases_src, server_dir / "tests" / "verifier_cases.jsonl")
        written.append(server_dir / "tests" / "verifier_cases.jsonl")

    table = None
    if calibration_readme is not None and Path(calibration_readme).exists():
        text_ = Path(calibration_readme).read_text(encoding="utf-8")
        if "| Suite" in text_:
            table = text_[text_.index("| Suite") :]
            table = table[: table.index("\n\n")] if "\n\n" in table else table
    suites: list[tuple[Suite, str, str]] = []
    for suite_dir in suite_dirs:
        suite = Suite.load(Path(suite_dir))
        suite_slug = slug(suite.name)
        env_name = f"blobfish_{suite_slug}"
        env_var = f"BLOBFISH_{suite_slug.upper()}_DIR"
        suites.append((suite, env_name, suite_slug))

        env_dir = out_dir / "environments" / env_name
        data_dir = env_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for name, text in (
            ("__init__.py", ""),
            ("config.yaml", _config_yaml(suite, env_name, suite_slug, env_var, hf_repo)),
            ("manifest.yaml", _manifest_yaml(suite, env_name)),
            ("README.md", _env_readme(suite, env_name, suite_slug, env_var, hf_repo, example_count, table)),
        ):
            path = env_dir / name
            path.write_text(text, encoding="utf-8")
            written.append(path)

        rows = build_rows(
            suite, source_url=f"https://hub.harborframework.com/datasets/blobfishai/{_harbor_dataset(suite_slug)}"
        )
        train, validation = split_rows(rows)
        examples = validation[:example_count]  # examples come from the validation split so rollouts exist
        write_jsonl(examples, data_dir / "example.jsonl")
        written.append(data_dir / "example.jsonl")
        example_rollouts = _rollouts_for(rollouts_dir, suite_slug, [row["task_id"] for row in examples])
        if example_rollouts:
            write_jsonl(example_rollouts, data_dir / "example_rollouts.jsonl")
            written.append(data_dir / "example_rollouts.jsonl")
        manifest_path = data_dir / "suite_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "suite": suite.name,
                    "harbor_dataset": f"blobfishai/{_harbor_dataset(suite_slug)}",
                    "packages": {task.root.name: package_digest(task.root) for task in suite.tasks},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(manifest_path)
        gitignore = data_dir / ".gitignore"
        gitignore.write_text("train.jsonl\nvalidation.jsonl\n", encoding="utf-8")
        written.append(gitignore)
        if len(suites) == 1:  # the first suite also backs the server's own default config and data
            default = _config_yaml(
                suite,
                SERVER_NAME,
                suite_slug,
                env_var,
                hf_repo,
                description=_family_description(suite),
                data_dir=f"resources_servers/{SERVER_NAME}/data",
            )
            (server_dir / "configs" / f"{SERVER_NAME}.yaml").write_text(default, encoding="utf-8")
            written.append(server_dir / "configs" / f"{SERVER_NAME}.yaml")
            for name in ("example.jsonl", "example_rollouts.jsonl", "suite_manifest.json"):
                if (data_dir / name).exists():
                    shutil.copyfile(data_dir / name, server_dir / "data" / name)
                    written.append(server_dir / "data" / name)

        upload_dir = out_dir / "hf_upload" / suite_slug
        for name, subset in (
            ("train.jsonl", train),
            ("validation.jsonl", validation),
            ("example.jsonl", rows[:example_count]),
        ):
            write_jsonl(subset, upload_dir / name)
            written.append(upload_dir / name)
        manifest = upload_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "suite": suite.name,
                    "environment": env_name,
                    "tasks": len(rows),
                    "train": len(train),
                    "validation": len(validation),
                    "tools": len(suite.tools),
                    "license": DATA_LICENSE_LABEL,
                    "source": f"https://hub.harborframework.com/datasets/blobfishai/{_harbor_dataset(suite_slug)}",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(manifest)

    for name, text in (
        ("app.py", _app_py()),
        ("requirements.txt", "-e nemo-gym[dev] @ ../../\n"),
        ("README.md", _server_readme(suites, table)),
        ("VENDORING.md", _vendoring_md(suites)),
        ("ATTRIBUTIONS_ENTRY.md", attributions_entry(suites)),
    ):
        path = server_dir / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    test_path = server_dir / "tests" / "test_app.py"
    test_path.write_text(_test_py(suites), encoding="utf-8")
    written.append(test_path)
    return written
