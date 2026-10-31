# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import importlib.util
import json
import uuid
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("reward_diff", REPO_ROOT / "scripts" / "ci" / "reward_diff.py")
reward_diff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reward_diff)

_APP = """
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)


class FakeConfig(BaseResourcesServerConfig):
    REVERIFY_MODE = ReverifyMode.{mode}
    target: str = "unset"


class FakeServer(SimpleResourcesServer):
    config: FakeConfig

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        text = body.response.output[-1].content[0].text
        return BaseVerifyResponse(**body.model_dump(), reward=float(text == self.config.target))
"""


def _rollout(text: str, stored: float) -> dict:
    """A saved rollout in the format an older Gym wrote, including the null Responses parameters."""
    source = REPO_ROOT / "resources_servers" / "example_single_tool_call" / "data" / "example_rollouts.jsonl"
    row = copy.deepcopy(json.loads(source.read_text().splitlines()[0]))
    row["response"]["output"][-1]["content"][0]["text"] = text
    row["reward"] = stored
    return row


def _server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mode: str = "STATELESS") -> Path:
    name = f"fake_{uuid.uuid4().hex[:8]}"
    server_dir = tmp_path / "resources_servers" / name
    (server_dir / "configs").mkdir(parents=True)
    (server_dir / "app.py").write_text(_APP.format(mode=mode))
    config = {name: {"resources_servers": {name: {"entrypoint": "app.py", "target": "yes"}}}}
    (server_dir / "configs" / f"{name}.yaml").write_text(json.dumps(config))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return Path("resources_servers") / name


def test_score_uses_the_server_config_and_reports_unscorable_rows(tmp_path: Path, monkeypatch) -> None:
    server = _server(tmp_path, monkeypatch)
    rollouts = tmp_path / "rollouts.jsonl"
    rows = [_rollout("yes", 1.0), _rollout("no", 1.0), {"reward": 0.0}]
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = reward_diff.score(server, rollouts)

    assert result["reverify_mode"] == "stateless"
    # The configured target decides the reward, so the config block was applied.
    assert [row.get("reward") for row in result["rows"]] == [1.0, 0.0, None]
    assert [row["stored"] for row in result["rows"]] == [1.0, 1.0, 0.0]
    assert "ValidationError" in result["rows"][2]["error"]


def test_score_skips_verifiers_that_read_session_state(tmp_path: Path, monkeypatch) -> None:
    server = _server(tmp_path, monkeypatch, mode="UNSUPPORTED")
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps(_rollout("yes", 1.0)) + "\n")

    assert "session state" in reward_diff.score(server, rollouts)["skipped"]


def _scores(*rows: tuple[float | None, float | None]) -> dict:
    return {
        "reverify_mode": "stateless",
        "rows": [
            {"row": index, "stored": stored, **({"reward": reward} if reward is not None else {"error": "E: boom"})}
            for index, (stored, reward) in enumerate(rows)
        ],
    }


def test_compare_lists_the_rollouts_whose_reward_changed() -> None:
    base = _scores((0.0, 0.0), (1.0, 1.0), (1.0, 1.0))
    head = _scores((0.0, 1.0), (1.0, 1.0), (1.0, 0.0))

    report = reward_diff.compare("resources_servers/x", "data/example_rollouts.jsonl", base, head)

    assert "**2 of 3 rollouts changed reward.**" in report
    assert "| 0 | 0.0 | 1.0 |" in report and "| 2 | 1.0 | 0.0 |" in report
    assert "| mean reward | 0.667 | 0.667 | 0.667 |" in report
    assert "WARNING" not in report


def test_compare_warns_when_the_base_cannot_reproduce_stored_rewards() -> None:
    scores = _scores((1.0, 0.0), (1.0, 0.0))

    report = reward_diff.compare("resources_servers/x", "data/example_rollouts.jsonl", scores, scores)

    assert "Base scores differ from the stored rewards on 2 of 2 rollouts" in report
    assert "No reward changes." in report


def test_compare_reports_a_new_server_and_unscorable_verifiers() -> None:
    new_server = reward_diff.compare("resources_servers/x", "r.jsonl", None, _scores((1.0, 1.0)))
    assert "No base comparison: the server is new in this PR." in new_server
    assert "changed reward" not in new_server

    judge = reward_diff.compare("resources_servers/x", "r.jsonl", _scores((1.0, None)), _scores((1.0, None)))
    assert "Could not score any rollout offline" in judge
    assert "first error: `E: boom`" in judge
    assert "No reward changes." not in judge
