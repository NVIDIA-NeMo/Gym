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
"""Play one game with the pinned official ARC Prize harness (arc-agi-3-benchmarking), in the harness's own venv.

The Gym agent starts this script once per game (``python -I worker.py --settings settings.json``, cwd = the game's
directory). The harness plays exactly as upstream's own ``Swarm`` does: an ONLINE ``arc_agi.Arcade`` on the
official game server (the resources server's, the authority) makes the game on the scorecard that the agent opened for
this rollout, and upstream's unmodified ``BenchmarkingAgent`` plays it. Its model calls go to the agent's route.

Model calls: every call carries the game's step and level when it was sent and its role in request headers
(``CALL_HEADERS``; the agent's route records them in the game's call log): step = actions answered + 1, level = levels
completed + 1, role ``gameplay`` / ``main``, or ``helper`` / ``harness-summary`` inside upstream's compaction summary.

Stops: upstream's own rules (per-level action budgets and the whole game's cap, told apart in the report's
``action_budget``: "level" or "total"; the RESET after GAME_OVER), its ``MAX_RUNTIME_SECONDS`` capped
at the time left in the run's play window, a ``SIGALRM`` at the window's end (it interrupts a model call in flight),
and ``SIGTERM`` (the agent's cancellation). In every case the worker writes ``result.json``: its own report of the
game (guid, levels, final state, the actions the server answered), which the verifier checks against the scorecard.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any


STOP_SIGNALS = {signal.SIGALRM, signal.SIGTERM}
# The request headers that tag a model call (lower case: the agent's route reads them case-insensitively).
CALL_HEADERS = {
    "step": "x-arc3-step",
    "level": "x-arc3-level",
    "role": "x-arc3-call-role",
    "history_id": "x-arc3-history-id",
}
SUMMARY_HISTORY_ID = "harness-summary"


class Stopped(BaseException):
    """The run window ended or the agent cancelled the game. A BaseException, so that upstream's retry handlers let
    it through."""


def prepare_model_config(entry: dict[str, Any], directory: Path, api_key: str) -> dict[str, Any]:
    """Validate ``entry`` with upstream's own loader (a one-entry model_configs.yaml in this game's directory)."""
    import yaml
    from benchmarking import model_config

    path = directory / "model_configs.yaml"
    path.write_text(yaml.safe_dump([entry], sort_keys=False))
    model_config.MODEL_CONFIG_PATH = path
    validated = model_config.get_model_config(entry["id"])
    os.environ[validated["client"]["api_key_env"]] = api_key
    return validated


def make_exact_version(arcade_cls: Any) -> None:
    """Make ``arc_agi.Arcade.make("<game>-<version>")`` in ONLINE mode play exactly that version (arc_agi asks the
    server for the metadata of the game name only and plays whichever version comes back). A copy of
    ``arc_agi_3_agent.harness_runtime.make_exact_version`` (this script runs in the harness's own venv, without Gym).
    """
    original = arcade_cls._create_remote_wrapper
    if getattr(original, "exact_version", False):
        return

    def create_remote_wrapper(self: Any, game_id: str, version: str | None, *args: Any, **kwargs: Any) -> Any:
        if version:
            game_id, version = f"{game_id}-{version}", None
        return original(self, game_id, version, *args, **kwargs)

    create_remote_wrapper.exact_version = True
    arcade_cls._create_remote_wrapper = create_remote_wrapper


def set_reset_timeout(wrapper_cls: Any, seconds: float) -> None:
    """Make arc_agi's ``RemoteEnvironmentWrapper.reset`` wait ``seconds`` for the server (its own timeout is 10 s; the
    engine imports a game on its first RESET). A copy of ``arc_agi_3_agent.harness_runtime.set_reset_timeout``.
    """
    original = getattr(wrapper_cls.reset, "original_reset", wrapper_cls.reset)

    def reset(self: Any, *args: Any, **kwargs: Any) -> Any:
        session = self._session
        post = session.post

        def post_with_reset_timeout(*post_args: Any, **post_kwargs: Any) -> Any:
            post_kwargs["timeout"] = seconds
            return post(*post_args, **post_kwargs)

        session.post = post_with_reset_timeout
        try:
            return original(self, *args, **kwargs)
        finally:
            del session.post

    reset.original_reset = original
    wrapper_cls.reset = reset


def tag_model_calls(agent: Any, report: dict[str, Any]) -> Callable[..., None]:
    """Tag the harness's model calls with ``CALL_HEADERS``; returns the function that re-tags them from ``report`` (call
    it after every action). The headers go on the harness's OpenAI client (``_custom_headers``, sent with every request
    of the pinned openai version); upstream's compaction summary is wrapped to tag its calls as helpers."""
    headers = getattr(agent._client, "_custom_headers", None)
    if not isinstance(headers, dict):
        raise RuntimeError("the harness's model client has no _custom_headers; its model calls cannot be tagged")

    def tag(role: str = "gameplay", history_id: str = "main") -> None:
        headers.update(
            {
                CALL_HEADERS["step"]: str(int(report["actions"]) + 1),
                CALL_HEADERS["level"]: str(int(report["levels_completed"]) + 1),
                CALL_HEADERS["role"]: role,
                CALL_HEADERS["history_id"]: history_id,
            }
        )

    compactor = getattr(agent, "_summary_compactor", None)
    if compactor is not None:
        compact = compactor.compact

        def tagged_compact(*args: Any, **kwargs: Any) -> Any:
            tag("helper", SUMMARY_HISTORY_ID)
            try:
                return compact(*args, **kwargs)
            finally:
                tag()

        compactor.compact = tagged_compact
    tag()
    return tag


def action_budget_fired(agent: Any) -> str:
    """Which of upstream's action budgets ended the game (its ExitReason ACTION_BUDGET covers both): "level" when the
    level the game is on used its own allowance (``_level_action_budgets``), else "total" (``MAX_ACTIONS_HARD_CAP``,
    checked first, or the game's ``MAX_ACTIONS``). The checks are those of upstream's ``BenchmarkingAgent.is_done`` at
    the pinned commit, on the agent's final state."""
    cap = getattr(agent, "_max_actions_hard_cap", None)
    if cap is not None and agent.action_counter >= cap:
        return "total"
    budgets = getattr(agent, "_level_action_budgets", None) or []
    level = int(agent.frames[-1].levels_completed) if getattr(agent, "frames", None) else None
    if level is not None and level < len(budgets) and getattr(agent, "_level_action_counter", 0) >= budgets[level]:
        return "level"
    return "total"


def play(settings: dict[str, Any], directory: Path, report: dict[str, Any]) -> None:
    import arc_agi
    import arc_agi.remote_wrapper
    from benchmarking.agent import BenchmarkingAgent
    from benchmarking.base import ExitReason

    entry = prepare_model_config(settings["benchmark_config"], directory, settings["model_api_key"])
    make_exact_version(arc_agi.Arcade)
    set_reset_timeout(arc_agi.remote_wrapper.RemoteEnvironmentWrapper, float(settings.get("reset_timeout_s", 120.0)))
    os.environ["RECORDINGS_DIR"] = str(directory / "recordings")
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.ONLINE,
        arc_base_url=settings["game_server_url"],
        arc_api_key=settings["game_api_key"],
        logger=logging.getLogger("arc_agi"),
    )
    env = arcade.make(settings["game_id"], scorecard_id=settings["card_id"])
    if env is None:
        raise RuntimeError(f"the game server did not make {settings['game_id']}")
    report["guid"] = env.observation_space.guid
    step = env.step
    retag: list = []  # the call tagger, once the harness's client exists

    def counted_step(*args: Any, **kwargs: Any) -> Any:
        # A stop (SIGALRM / SIGTERM) waits until the step is answered and counted, so the count is exact.
        signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
        try:
            frame = step(*args, **kwargs)
            report["actions"] += 1  # an action the server answered
            report["levels_completed"] = int(frame.levels_completed)
            report["state"] = frame.state.name
            for tag in retag:
                tag()
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)
        return frame

    env.step = counted_step
    report.update(actions=0, levels_completed=0, state=env.observation_space.state.name)
    agent = BenchmarkingAgent(
        card_id=settings["card_id"],
        game_id=settings["game_id"],
        agent_name="arc_agi_3_official_agent",
        ROOT_URL=settings["game_server_url"],
        record=True,
        arc_env=env,
        config=entry["id"],
    )
    retag.append(tag_model_calls(agent, report))
    seconds_left = max(0.0, float(settings["deadline_epoch"]) - time.time())
    agent.MAX_RUNTIME_SECONDS = min(float(agent.MAX_RUNTIME_SECONDS), seconds_left)
    # One HTTP attempt per model call (upstream's own retry loop decides), bounded by the window.
    agent._client.timeout = max(1.0, seconds_left)
    agent._client.max_retries = 0

    def window_end(*_: Any) -> None:
        raise Stopped("the run's play window ended")

    signal.signal(signal.SIGALRM, window_end)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds_left))
    try:
        agent.main()
        report["exit_reason"] = agent.exit_reason.value
        if agent.exit_reason == ExitReason.ACTION_BUDGET:
            report["action_budget"] = action_budget_fired(agent)
    except Stopped as exc:
        report["exit_reason"] = ExitReason.TIME_BUDGET.value
        report["stopped"] = str(exc)
    except Exception as exc:
        report["exit_reason"] = agent.exit_reason.value
        report["error"] = f"{agent.exit_reason.value}: {type(exc).__name__}: {exc}"
        logging.error("the harness stopped with an error:\n%s", traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            agent.cleanup()
            agent._client.close()
        except Exception:
            logging.error("harness cleanup failed:\n%s", traceback.format_exc())
        report["run_dir"] = str(directory / agent.run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    sys.path.insert(0, settings["harness_src"])
    directory = Path.cwd()
    report: dict[str, Any] = {"exit_reason": None, "error": None}

    def cancelled(*_: Any) -> None:
        raise Stopped("the agent cancelled the game")

    signal.signal(signal.SIGTERM, cancelled)
    try:
        play(settings, directory, report)
    except BaseException as exc:  # the harness or the game could not be set up, or a stop came before play
        logging.error("worker failed:\n%s", traceback.format_exc())
        if not isinstance(exc, Stopped):
            report["error"] = report.get("error") or f"{type(exc).__name__}: {exc}"
        else:
            report["stopped"] = str(exc)
    finally:
        tmp = directory / "result.json.tmp"
        tmp.write_text(json.dumps(report))
        os.replace(tmp, directory / "result.json")


if __name__ == "__main__":
    main()
