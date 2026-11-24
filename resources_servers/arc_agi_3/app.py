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
"""ARC-AGI-3 verifier: RHAE of one played game attempt, from the official game server's scorecard.

The authority is the official ``arc_agi`` game server (``game_server.py``), which this server starts: every harness plays
its game through it, one scorecard per rollout. At ``/verify`` this server closes the rollout's scorecard (if its
harness has not), takes ``reward = score / 100`` from it, and fails loudly (``ValueError``) when the agent's own report
(guid, levels, won or not, action count) differs from the card. RHAE@T comes from the server's own per-action
timestamps (its recordings) on the run clock.

The agent's report of the game travels in the response it verifies: the activation reply of an ARC-AGI-3 agent session
carries it under ``ATTEMPT_REPORT_KEY`` (see ``ArcAgi3VerifyRequest``).
"""

import asyncio
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.core import MODEL_REQUEST_ERROR
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.arc_agi_3.game_server import GameServer
from resources_servers.arc_agi_3.rhae import (
    TimingError,
    level_clear_steps,
    recorded_play,
    rhae_at,
    utc_clock_offset_seconds,
)
from resources_servers.arc_agi_3.setup_arc_agi import ensure_flask


LOG = logging.getLogger(__name__)
# The field of an ARC-AGI-3 agent's activation reply (the response verified) that holds its report of the game: the
# ArcAgi3VerifyRequest fields below (game_record, scorecard_id, ...) and the rollout's harness fields (harness,
# harness_revision, sampling, agent_call_log, ...), which the verify response keeps.
ATTEMPT_REPORT_KEY = "arc_agi_3_attempt"


class ArcAgi3ResourcesServerConfig(BaseResourcesServerConfig):
    # Directory of the ARC-AGI-3 environment files: <game>/<version>/{<game>.py, metadata.json}. A game is
    # <game>-<version>; one directory can hold several versions of a game (e.g. synthetic games sg0005-aee9edbb and
    # sg0005-2b3d7198).
    environments_dir: str
    # Directories put first on the game server's PYTHONPATH: runtimes that some games import (e.g. the param25 games'
    # bundled upstream `arcgames`). Relative paths are relative to the Gym repository root.
    python_path: list[str] = Field(default_factory=list)
    # RHAE is also reported at these hours of play (the kit's RHAE@T).
    score_hours: list[float] = Field(default_factory=lambda: [1.0, 2.0])
    # The number of games (dataset rows) of the evaluation, e.g. 25 for public25. When set, the aggregate RHAE is
    # reported only if every expected game has a valid result; when unset, it is a statistic of the received rows.
    expected_games: int | None = None
    # The official game server: its files (closed cards, recordings, log) go to <game_server_dir>/<start time>/.
    game_server_dir: str = "results/arc_agi_3_game_server"
    # null = a free port.
    game_server_port: int | None = None
    # An open scorecard idle this long is closed by the server (its default, 15 min, is shorter than a long model
    # turn); it must exceed the agents' play window.
    scorecard_timeout_min: int = 240


class ArcAgi3RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    # A create_dataset.py row: game_id, full_game_id, version, files_sha256, baseline_actions, ...
    verifier_metadata: dict[str, Any]


class ArcAgi3VerifyRequest(ArcAgi3RunRequest, BaseVerifyRequest):
    # These fields come from the response's ATTEMPT_REPORT_KEY (with_attempt_report).
    # The agent's own report of the attempt: a taaf GameRun.to_json_dict() (Kaggle kit harness) or the same fields with an
    # "actions" count instead of a history (official harness). None when the game could not start.
    game_record: dict[str, Any] | None = None
    # The rollout's scorecard on the game server, its game run's guid and the key that opened the card.
    scorecard_id: str | None = None
    game_guid: str | None = None
    scorecard_api_key: str | None = None
    # Why the game could not start or did not finish normally (agent side).
    agent_error: str | None = None
    # Set when no model call of the game got an answer (the agent's call log): the game was played without the model,
    # so its score does not measure it. The result is invalid and the rollout masked (model_request_error).
    model_error: str | None = None
    # The attempt was played but the agent could not save its run snapshot (the record is still scored).
    persistence_error: str | None = None
    # The run's clock origin as a UTC ISO time (taken before the first game started): RHAE@T counts clears before
    # origin + T.
    benchmark_start_time: str | None = None


class ArcAgi3VerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    game_id: str
    rhae: float
    levels_completed: int
    number_of_levels: int
    actions: int
    # Terminal state from the scorecard: "won" or "not_won" ("not_started" without a game).
    state: str
    # The agent's own end state (gave_up, cancelled, crashed, won, not_started).
    agent_state: str
    scorecard_id: str | None = None
    # Runs (plays) of the game on the card; a full reset starts another run.
    scorecard_runs: int | None = None
    # False when the server's action times cannot support RHAE@T; the rhae_<h>h fields are then None.
    timing_valid: bool
    timing_error: str | None = None
    # Seconds from the run's clock origin to this game's first server event; None (and rhae_<h>h None) without the
    # origin.
    clock_offset_seconds: float | None = None
    # Why the rhae_<h>h fields are None although the timing is valid (no run clock origin in the request).
    clock_error: str | None = None
    # True when the attempt counts as a complete, verified result: played to the end of its window (not cancelled),
    # server timing valid, no agent, model or persistence error. A crashed game keeps its score and is counted.
    result_valid: bool
    agent_error: str | None = None
    model_error: str | None = None
    persistence_error: str | None = None
    # From the game server's recording of the run (None when it cannot be read): the actions taken on each level
    # (counted on the level the game was on before the action; the clearing action counts on the level it clears), and
    # the game-wide action number (1-based) at which each cleared level was first cleared.
    actions_per_level: list[int] | None = None
    level_clear_steps: list[int] | None = None


def with_attempt_report(body: ArcAgi3VerifyRequest) -> ArcAgi3VerifyRequest:
    """The request with the agent's report (the response's ``ATTEMPT_REPORT_KEY``) as its own fields, and the response
    without it. The report cannot replace the task's own fields."""
    response = body.response.model_dump(mode="json") if body.response is not None else {}
    report = response.pop(ATTEMPT_REPORT_KEY, None)
    if report is None:
        return body
    task_fields = ("verifier_metadata", "responses_create_params", "response")
    report = {k: v for k, v in report.items() if k not in task_fields}
    return ArcAgi3VerifyRequest.model_validate(body.model_dump(mode="json") | report | {"response": response})


def hour_key(hours: float) -> str:
    return f"rhae_{hours:g}h"


def repo_path(path: str) -> str:
    """A config path; relative paths are relative to the Gym repository root (servers run in their own dir)."""
    return str(Path(path) if Path(path).is_absolute() else PARENT_DIR / path)


def game_files_sha256(game_dir: Path) -> dict[str, str]:
    """sha256 of a game version's files except metadata.json (create_dataset.py ``files_sha256``)."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(game_dir.iterdir())
        if p.is_file() and p.name != "metadata.json"
    }


def load_environment_metadata(environments_dir: str) -> dict[str, dict[str, Any]]:
    """metadata.json of every game under ``environments_dir``, keyed by full game id (``ls20-9607627b``), with the
    loaded files' version directory and sha256 (``_version``, ``_files_sha256``)."""
    root = Path(environments_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"ARC-AGI-3 environments_dir does not exist: {root}")
    games: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(root.glob("*/*/metadata.json")):
        meta = json.loads(meta_path.read_text())
        if meta["game_id"] in games:
            raise ValueError(f"Two copies of game {meta['game_id']} under {root}; keep exactly one")
        meta["_version"] = meta_path.parent.name
        meta["_files_sha256"] = game_files_sha256(meta_path.parent)
        games[meta["game_id"]] = meta
    if not games:
        raise FileNotFoundError(f"No ARC-AGI-3 games (*/*/metadata.json) under {root}")
    return games


def check_pins(verifier_metadata: dict[str, Any], loaded: dict[str, Any]) -> None:
    """Raise unless the dataset row's pinned game (full id, version, file sha256) is exactly the loaded game."""
    game_id = verifier_metadata.get("game_id")
    for key, want, have in (
        ("full_game_id", verifier_metadata.get("full_game_id"), loaded["game_id"]),
        ("version", verifier_metadata.get("version"), loaded["_version"]),
        ("files_sha256", verifier_metadata.get("files_sha256"), loaded["_files_sha256"]),
    ):
        if want is None:
            raise ValueError(f"{game_id}: the dataset row has no {key} pin (rebuild it with create_dataset.py)")
        if want != have:
            raise ValueError(f"{game_id}: dataset pin {key}={want!r} does not match the loaded game ({have!r})")


def card_game(card: dict[str, Any], full_game_id: str, card_id: str) -> Any:
    """The card's only game, as the SDK's ``EnvironmentScoreList`` (score of the best run, summed actions)."""
    from arc_agi.scorecard import EnvironmentScoreList

    games = card.get("environments") or []
    ids = [game.get("id") for game in games]
    if ids != [full_game_id]:
        raise ValueError(f"scorecard {card_id} holds the games {ids}, not exactly {full_game_id}")
    return EnvironmentScoreList.model_validate(games[0])


def state_name(state: Any) -> str:
    return str(getattr(state, "name", state))


def cross_check(score: Any, record: dict[str, Any], guid: str | None, card_id: str, game_id: str) -> None:
    """Raise ``ValueError`` unless the agent's report agrees with the authoritative card."""
    runs = score.runs or []
    reported_actions = int(record["actions"]) if "actions" in record else len(record.get("history") or [])
    problems = []
    if not runs:
        problems.append("the card has no run of the game")
    else:
        guids = sorted({run.guid for run in runs})
        if guids != [guid]:
            problems.append(f"run guids {guids} != the agent's {guid!r}")
        last = runs[-1]
        if int(last.levels_completed) != int(record.get("levels_completed") or 0):
            problems.append(f"levels {last.levels_completed} != the agent's {record.get('levels_completed')}")
        if (state_name(last.state) == "WIN") != (record.get("state") == "won"):
            problems.append(f"state {state_name(last.state)} != the agent's {record.get('state')!r}")
    if int(score.actions) != reported_actions:
        problems.append(f"actions {score.actions} != the agent's {reported_actions}")
    if problems:
        raise ValueError(
            f"{game_id}: the authoritative scorecard {card_id} does not match the agent's report: "
            + "; ".join(problems)
        )


class ArcAgi3ResourcesServer(SimpleResourcesServer):
    config: ArcAgi3ResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        ensure_flask()
        if self.config.num_workers not in (None, 1):
            # Each worker process would start its own game server; a verify could reach one without the rollout's card.
            raise ValueError("the arc_agi_3 resources server runs one game server; num_workers must be 1")
        self._environments_dir = repo_path(self.config.environments_dir)
        self._games = load_environment_metadata(self._environments_dir)
        out = Path(repo_path(self.config.game_server_dir)) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._game_server = GameServer(
            self._environments_dir,
            out,
            self.config.game_server_port,
            self.config.scorecard_timeout_min,
            python_path=[repo_path(p) for p in self.config.python_path],
        )
        return super().model_post_init(context)

    @property
    def game_server(self) -> GameServer:
        return self._game_server

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.get("/game_server")(self.game_server_info)
        inner_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app_: FastAPI):
            await asyncio.to_thread(self.game_server.start, len(self._games))
            try:
                async with inner_lifespan(app_) as state:
                    yield state
            finally:
                await asyncio.to_thread(self.game_server.stop)

        app.router.lifespan_context = lifespan
        return app

    async def game_server_info(self) -> dict[str, Any]:
        """Where the agents play: the official game server's URL."""
        if not self.game_server.running:
            raise HTTPException(status_code=503, detail="the ARC-AGI-3 game server is not running")
        return {"url": self.game_server.url, "games": len(self._games), "out_dir": str(self.game_server.out_dir)}

    def loaded_game(self, full_game_id: str) -> dict[str, Any]:
        meta = self._games.get(full_game_id)
        if meta is None:
            raise KeyError(f"Game {full_game_id} is not in {self._environments_dir}")
        return meta

    async def verify(self, body: ArcAgi3VerifyRequest) -> ArcAgi3VerifyResponse:
        body = with_attempt_report(body)
        meta = body.verifier_metadata
        game_id = str(meta["game_id"])
        if not meta.get("full_game_id"):
            raise ValueError(f"{game_id}: the dataset row has no full_game_id pin (rebuild it with create_dataset.py)")
        loaded = self.loaded_game(str(meta["full_game_id"]))
        check_pins(meta, loaded)
        full_game_id = loaded["game_id"]
        baseline = [int(x) for x in loaded["baseline_actions"]]
        number_of_levels = len(baseline)
        hours: dict[str, float | None] = {hour_key(h): None for h in self.config.score_hours}
        common = dict(
            **body.model_dump(exclude={"game_record", "scorecard_api_key", "scorecard_id"}),
            game_id=game_id,
            number_of_levels=number_of_levels,
            scorecard_id=body.scorecard_id,
        )
        if body.model_error:  # the model never answered: the score is no measurement of it
            common.update(mask_sample=True, failure_kind=MODEL_REQUEST_ERROR, failure_reason=body.model_error)
        record = body.game_record
        if record is None:
            if body.scorecard_id and body.scorecard_api_key:  # a card opened for a game that never started
                await asyncio.to_thread(self.game_server.close_card, body.scorecard_id, body.scorecard_api_key)
            return ArcAgi3VerifyResponse(
                **common,
                reward=0.0,
                rhae=0.0,
                levels_completed=0,
                actions=0,
                state="not_started",
                agent_state="not_started",
                timing_valid=False,
                timing_error="no game record",
                result_valid=False,
                **hours,
            )

        if record.get("game_id") != full_game_id:
            raise ValueError(
                f"{game_id}: the record is for {record.get('game_id')!r}, not the pinned {full_game_id!r}"
            )
        record_baseline = record.get("base_actions_per_level")
        if record_baseline is not None and [int(x) for x in record_baseline] != baseline:
            raise ValueError(
                f"{game_id}: the record's baselines {record_baseline} differ from the server's {baseline}"
            )
        if int(record["number_of_levels"]) != number_of_levels:
            raise ValueError(
                f"{game_id}: the record has {record['number_of_levels']} levels, the server's game has {number_of_levels}"
            )
        if not (body.scorecard_id and body.scorecard_api_key and body.game_guid):
            raise ValueError(f"{game_id}: the request has no scorecard id, key or game guid; the game is unverifiable")

        card = await asyncio.to_thread(self.game_server.close_card, body.scorecard_id, body.scorecard_api_key)
        score = card_game(card, full_game_id, body.scorecard_id)
        cross_check(score, record, body.game_guid, body.scorecard_id, game_id)
        rhae = float(score.score)
        levels_completed = int(score.levels_completed)
        won = state_name(score.runs[-1].state) == "WIN"
        agent_state = str(record.get("state") or "")

        timing_error = None
        clock_error = None
        offset = None
        per_level: dict[str, list[int] | None] = {"actions_per_level": None, "level_clear_steps": None}
        try:
            events = self.game_server.recording(body.scorecard_id, full_game_id, body.game_guid)
            actions_per_level, times, recorded_levels, first_event = recorded_play(events, number_of_levels)
            if len(times) != int(score.actions) or recorded_levels != levels_completed:
                raise ValueError(
                    f"{game_id}: the server's recording ({len(times)} actions, {recorded_levels} levels) differs from "
                    f"its own scorecard ({score.actions} actions, {levels_completed} levels)"
                )
            per_level = {"actions_per_level": actions_per_level, "level_clear_steps": level_clear_steps(events)}
            if body.benchmark_start_time is None:
                # RHAE@T is on the run clock only (the kit's); a per-game clock would give other values at T.
                clock_error = "no benchmark_start_time (the run clock origin): RHAE@T not computed"
            else:
                offset = utc_clock_offset_seconds(body.benchmark_start_time, first_event)
                for h in self.config.score_hours:
                    hours[hour_key(h)] = rhae_at(
                        actions_per_level,
                        baseline,
                        number_of_levels,
                        recorded_levels,
                        times,
                        t_seconds=h * 3600.0,
                        offset_seconds=offset,
                    )
        except (TimingError, FileNotFoundError) as exc:
            timing_error = str(exc)
            hours = {hour_key(h): None for h in self.config.score_hours}
        result_valid = (
            agent_state != "cancelled"  # aborted before the play window ended
            and timing_error is None
            and body.agent_error is None
            and body.model_error is None
            and body.persistence_error is None
        )
        return ArcAgi3VerifyResponse(
            **common,
            reward=rhae / 100.0,
            rhae=rhae,
            levels_completed=levels_completed,
            actions=int(score.actions),
            state="won" if won else "not_won",
            agent_state=agent_state,
            scorecard_runs=len(score.runs),
            timing_valid=timing_error is None,
            timing_error=timing_error,
            clock_offset_seconds=offset,
            clock_error=clock_error,
            result_valid=result_valid,
            **per_level,
            **hours,
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """RHAE over the attempts received here, with counts of everything that is not a valid result.

        Gym's failed rollouts never reach this method, so the received rows can be fewer than the evaluation's games.
        A game is a pinned game id (``full_game_id``): ``rhae`` / ``rhae_<h>h`` are the mean over games of each game's
        mean over its rollouts, so repeats never weigh a game more. With ``expected_games`` set, they are reported only
        when that many distinct games are here, every row valid (``complete``); a duplicated game does not stand in for
        a missing one. Without it, they are reported when every received row is valid, and they are a statistic of the
        received games only (``score_over: received_games``; ``complete`` is None). The benchmark table comes from
        ``summarize_rhae.py --expected <dataset>``.
        """
        rows = [row for task in tasks for row in task]
        if not rows:
            return {}
        valid = [r for r in rows if r.get("result_valid")]
        all_valid = len(valid) == len(rows)
        # Games by their pinned id: repeats of a game (several rollouts, or the same row twice) are one game, and every
        # game weighs the same in the mean (each game's mean over its rollouts first).
        by_game: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            meta = r.get("verifier_metadata") or {}
            by_game.setdefault(str(meta.get("full_game_id") or r.get("game_id")), []).append(r)
        expected = self.config.expected_games
        complete = None if expected is None else all_valid and len(by_game) == expected
        scored = complete if expected is not None else all_valid

        def game_mean(key: str) -> float:
            return sum(sum(float(r[key]) for r in g) / len(g) for g in by_game.values()) / len(by_game)

        metrics: dict[str, Any] = {
            "received_games": len(by_game),
            "received_rows": len(rows),
            "expected_games": expected,
            "valid_results": len(valid),
            "all_received_valid": all_valid,
            "complete": complete,
            "score_over": "received_games" if expected is None else "expected_games",
            "not_started": sum(1 for r in rows if r.get("agent_state") == "not_started"),
            "crashed": sum(1 for r in rows if r.get("agent_state") == "crashed"),
            "cancelled": sum(1 for r in rows if r.get("agent_state") == "cancelled"),
            "timing_invalid": sum(1 for r in rows if not r.get("timing_valid")),
            "persistence_errors": sum(1 for r in rows if r.get("persistence_error")),
            "model_errors": sum(1 for r in rows if r.get("model_error")),
            "levels_completed": sum(int(r.get("levels_completed") or 0) for r in rows),
            "rhae": game_mean("rhae") if scored else None,
        }
        for h in self.config.score_hours:
            key = hour_key(h)
            timed = scored and all(r.get(key) is not None for r in rows)
            metrics[key] = game_mean(key) if timed else None
        return metrics

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """The shared aggregation, with the RHAE and completeness fields over every received row.

        The shared path calls ``compute_metrics`` with the measured rows only (masked rows removed), so a masked
        model-error attempt would vanish from ``received_rows`` and ``complete``. Those fields are recomputed from all
        rows here; the generic quality metrics and the coverage counters stay as the shared path computes them.
        """
        metrics = await super().aggregate_metrics(body)
        if body.verify_responses:
            metrics.agent_metrics.update(self.compute_metrics([body.verify_responses]))
            metrics.key_metrics.update(self.get_key_metrics(metrics.agent_metrics))
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "rhae",
            *(hour_key(h) for h in self.config.score_hours),
            "complete",
            "score_over",
            "received_games",
            "received_rows",
            "expected_games",
            "valid_results",
            "not_started",
            "crashed",
            "cancelled",
            "timing_invalid",
            "persistence_errors",
            "levels_completed",
        ]
        return {k: agent_metrics[k] for k in keys if k in agent_metrics}


if __name__ == "__main__":
    ArcAgi3ResourcesServer.run_webserver()
