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
"""One pivot window's harness, in its own process: the recorded harness plays the game, and its model is a pipe.

    <worker python> worker.py <socket fd>

The harness runs unmodified: its solver plays one game (``HarnessSolver._play_one`` on a local ``taaf`` game of the
pivot pool), and only its model endpoint is replaced (``tool_agent.requests``). Each model request the harness sends is
answered from one of two places:

- **Replay**, up to the pivot: the recorded reply. Before it is answered, the request must equal the recorded one
  (``request_sha256`` of the POST body, or equal once object addresses are masked) and the board and action count must
  equal the recorded ones. The harness thus re-runs every recorded Python cell and game action itself; its host state
  and its Python sandbox are its own.
- **Live**, from the pivot on: the request goes to the server (the policy's turn), and the policy's reply comes back.

Without a recording, every request is live from the game's start. With ``record`` (a path), each live request is also
written there with the state it was sent from, as a new recording (``record.py`` builds a pivot pool that way).

What a replay needs to reproduce the recording (configuration, not harness logic): the sandbox seed the recording ran
with; the run clock the recording printed (a virtual clock, set from the recorded tool results; live calls advance it
per call and per generated token); the sandbox memory limit and the retry guard's cell counter that the recorded tool
results show; the recorded names of the sandbox's temporary directories; and a sandbox timeout long enough that cells
that finished in the recording finish here.

Protocol on the socket: requests ``(op, args)``, replies ``(True, value)`` or ``(False, error dict)``. Ops: ``start``
(spec) -> the pivot request; ``reply`` (message, finish_reason, usage) -> the next request or the game's end; ``close``.
Standard library plus the harness runtime (arc_agi, arcengine, numpy, requests).
"""

import hashlib
import json
import os
import queue
import shutil
import sys
import tempfile
import textwrap
import threading
import traceback
import weakref
from multiprocessing.connection import Connection
from typing import Any

import oracle
import recording


# --------------------------------------------------------------------------------------------- sandbox determinism
# The recording's sandbox ran with a fixed hash seed and seeded RNGs. The same seeding is put in the harness's sandbox
# bootstrap (read at every sandbox start): a re-exec with PYTHONHASHSEED first, then the RNG seeds right after the
# bootstrap installs its import hook.
SEED_ANCHOR = 'runtime_globals["__builtins__"]["__import__"] = _safe_import'
HASH_SEED_PREFIX = r"""
def _gym_fixed_hash_seed(seed):
    import os, sys
    if not sys.flags.hash_randomization or os.environ.get("PYTHONHASHSEED") == str(seed):
        return
    try:
        with open("/proc/self/cmdline", "rb") as stream:
            argv = [a.decode("utf-8", "surrogateescape") for a in stream.read().split(b"\0")[:-1]]
    except OSError:
        return
    if "-c" not in argv:
        return
    code = argv[argv.index("-c") + 1]
    flags = ["-s", "-P"] + (["-S"] if sys.flags.no_site else [])
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONHASHSEED"] = str(seed)
    os.execve(sys.executable, [sys.executable, *flags, "-c", code], env)
_gym_fixed_hash_seed(__SEED__)
del _gym_fixed_hash_seed
"""
SEED_RNGS = r"""
def _gym_seed_rngs(seed=__SEED__):
    import random as _random
    _random.seed(seed)
    _seeded = []

    def _numpy(module):
        if _seeded or not hasattr(module, "random"):
            return
        _seeded.append(True)
        module.random.seed(seed)
        _default_rng, _count = module.random.default_rng, [0]

        def default_rng(seed_value=None):
            if seed_value is None:
                _count[0] += 1
                seed_value = module.random.SeedSequence([seed, _count[0]])
            return _default_rng(seed_value)
        module.random.default_rng = default_rng

    if "numpy" in sys.modules:
        _numpy(sys.modules["numpy"])
    _import = runtime_globals["__builtins__"]["__import__"]

    def _seeding_import(name, *args, **kwargs):
        module = _import(name, *args, **kwargs)
        if not _seeded and "numpy" in sys.modules and "numpy.random" in sys.modules:
            _numpy(sys.modules["numpy"])
        return module
    runtime_globals["__builtins__"]["__import__"] = _seeding_import
_gym_seed_rngs()
"""


def seeded_bootstrap(bootstrap: str, seed: int) -> str:
    lines = bootstrap.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.strip() == SEED_ANCHOR]
    if len(hits) != 1:
        raise RuntimeError("the harness's sandbox bootstrap has no single seed anchor: this harness is not supported")
    line = lines[hits[0]]
    indent = line[: len(line) - len(line.lstrip())]
    block = textwrap.indent(textwrap.dedent(SEED_RNGS.replace("__SEED__", str(int(seed)))), indent)
    lines.insert(hits[0] + 1, block)
    return HASH_SEED_PREFIX.replace("__SEED__", str(int(seed))) + "".join(lines)


# The solver's play budget when the recording shows no run clock (the clock then shows no time remaining).
NO_BUDGET_S = 1e9


class _NamedTempDir:
    def __init__(self, path: str):
        self.name = path
        self._finalizer = weakref.finalize(self, shutil.rmtree, path, True)

    def cleanup(self) -> None:
        self._finalizer()

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, *exc: Any) -> None:
        self.cleanup()


class SandboxTempfile:
    """``tempfile`` for the harness's sandbox module: during a replay, a sandbox's working directory takes the name the
    recording shows (``source``); otherwise it is ``tempfile`` itself."""

    def __init__(self) -> None:
        self.source = None

    def __getattr__(self, name: str) -> Any:
        return getattr(tempfile, name)

    def TemporaryDirectory(self, *args: Any, **kwargs: Any) -> Any:  # noqa: N802 (tempfile's name)
        name = (
            self.source() if self.source is not None and kwargs.get("prefix") == recording.SANDBOX_DIR_PREFIX else None
        )
        if name:
            path = os.path.join(tempfile.gettempdir(), name)
            try:
                os.mkdir(path, 0o700)
                return _NamedTempDir(path)
            except FileExistsError:
                pass
        return tempfile.TemporaryDirectory(*args, **kwargs)


class MemoryPool:
    """The harness's sandbox memory pool interface with a settable limit. On Kaggle the pool's total is split over the
    unfinished games, so a game's limit grows as others finish; here the limit is what the recording reported."""

    def __init__(self, limit_bytes: int, total_bytes: int):
        self.limit_bytes = int(limit_bytes)
        self.total_bytes = max(int(total_bytes), int(limit_bytes))
        self.remaining = 1

    def finish(self, attempt_index: int) -> None:
        return None


class VirtualClock:
    """The run clock the harness shows the model (``run_elapsed_seconds`` / ``time_remaining_seconds``)."""

    def __init__(self, call_s: float, token_s: float, budget: float | None, overrides: dict[int, tuple]):
        self.call_s, self.token_s, self.budget = call_s, token_s, budget
        self.elapsed = 0.0
        self.pinned: tuple | None = None
        # every pair returned, in order; a recorded pair to return at a given read (a replay that showed the clock a
        # call read in a later cell)
        self.log: list[tuple] = []
        self.overrides = overrides

    def payload(self) -> dict[str, float | None]:
        override = self.overrides.get(len(self.log))
        if override is not None:
            self.set(*override)
        if self.pinned is not None and self.pinned[0] == self.elapsed:
            pair = self.pinned
        else:
            pair = (self.elapsed, None if self.budget is None else max(0.0, self.budget - self.elapsed))
        self.log.append(pair)
        return {"run_elapsed_seconds": pair[0], "time_remaining_seconds": pair[1]}

    def advance(self, tokens: int) -> None:
        self.elapsed += self.call_s + self.token_s * max(0, int(tokens or 0))

    def set(self, elapsed: float, remaining: float | None) -> None:
        self.elapsed = float(elapsed)
        if remaining is not None:
            self.budget = float(elapsed) + float(remaining)
        self.pinned = (self.elapsed, None if remaining is None else float(remaining))


class Stop(BaseException):
    """Ends the harness's game thread (the window is closed). A BaseException: the harness's handlers let it through."""


class ReplayError(BaseException):
    """The harness left the recording (the pivot cannot be restored exactly)."""

    def __init__(self, info: dict[str, Any]):
        super().__init__(info.get("reason"))
        self.info = info


class Response:
    """The part of ``requests.Response`` the harness reads."""

    def __init__(self, payload: dict[str, Any]):
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class ModelPipe:
    """Stands in for ``requests`` in the harness's tool agent: a chat completion is answered by the window."""

    def __init__(self, window: "Window", requests_module: Any):
        self._window, self._requests = window, requests_module

    def post(self, url: str, *args: Any, json: dict[str, Any] | None = None, **kwargs: Any) -> Response:
        if not str(url).rstrip("/").endswith("/chat/completions") or json is None:
            raise self._requests.RequestException(f"the pivot window serves only chat completions, not {url}")
        message, finish_reason, usage = self._window.model_call(json)
        payload = {"choices": [{"index": 0, "message": message, "finish_reason": finish_reason}]}
        if usage:
            payload["usage"] = usage
        return Response(payload)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._requests, name)


class Window:
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        if spec.get("recording"):
            self.recorded = recording.parse(open(spec["recording"], "rb").read())
        else:  # no recording: every request is live
            self.recorded = {"header": {}, "actions": [], "calls": []}
        self.calls = self.recorded["calls"]
        self.upto = int(spec.get("upto") or 0)
        if self.calls and not 0 <= self.upto < len(self.calls):
            raise ValueError(f"pivot call index {self.upto} is not in the recording ({len(self.calls)} calls)")
        self.results = recording.tool_results(self.calls)
        # Thinking effort. While replaying, each request gets its recorded request's effort fields (a recording whose
        # effort was set outside the harness). Live, every thinking request carries the spec's ``reasoning_effort``,
        # else the pivot's recorded effort, as chat-template kwargs (the agent sends a request's chat_template_kwargs):
        # under the recording's effort kwargs, default ``reasoning_effort``. The recorded effort is the recorded
        # requests' own, else the pool's (``recorded_effort``: a collector that set it on the wire only, past the
        # harness's request).
        recorded = next(
            (
                recording.effort_of(c["payload"])
                for c in reversed(self.calls[: self.upto + 1])
                if recording.thinks(c["payload"])
            ),
            {},
        )
        recorded = {field.lstrip("@"): value for field, value in recorded.items()}
        if not recorded and spec.get("recorded_effort"):
            recorded = {"reasoning_effort": spec["recorded_effort"]}
        effort = spec.get("reasoning_effort")
        self.live_effort = {field: effort for field in recorded or ("reasoning_effort",)} if effort else recorded
        self.k = 0  # requests the harness sent
        self.replaying = bool(self.calls)
        self.writer = Writer(spec) if spec.get("record") else None
        self.masked = 0  # requests equal to the recorded ones once object addresses are masked
        self.engine_fields = 0  # requests equal to the recorded ones but for engine-only fields
        self.to_main: "queue.Queue" = queue.Queue()
        self.to_game: "queue.Queue" = queue.Queue()
        self.local = threading.local()
        self.steps: list[dict[str, Any]] = []
        self.analyzer = None
        self.thread: threading.Thread | None = None
        self.ended: str | None = None
        self.stopped = False  # the window is over (a replay error or close): every further model call raises Stop

    # ------------------------------------------------------------------ harness setup
    def setup(self) -> None:
        spec = self.spec
        os.environ.update({k: str(v) for k, v in spec["env"].items()})
        for entry in reversed(spec["harness_paths"]):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        import arc_agi
        import requests
        import taaf.game
        import taaf.game_api
        from inference.agent import python_tool_sandbox, tool_agent
        from inference.framework import solver as solver_module

        self.requests = requests
        python_tool_sandbox._SANDBOX_BOOTSTRAP = seeded_bootstrap(
            python_tool_sandbox._SANDBOX_BOOTSTRAP, int(spec["sandbox_seed"])
        )
        if spec.get("sandbox_python"):
            native = python_tool_sandbox._sandbox_command
            path = spec["sandbox_python"]
            python_tool_sandbox._sandbox_command = lambda: [path, *native()[1:]]
        self.tempfile = SandboxTempfile()
        python_tool_sandbox.tempfile = self.tempfile
        tool_agent.requests = ModelPipe(self, requests)

        # the request's role (gameplay or a helper: a summary, a turn compaction) and whether it continues a reply
        original = tool_agent.ToolAgent._chat_completion
        local = self.local

        def chat_completion(agent: Any, *args: Any, **kwargs: Any) -> Any:
            local.role = str(kwargs.get("gdn_role", "gameplay"))
            local.continuation = kwargs.get("continue_assistant") is not None or (
                kwargs.get("force_close_prefix") is not None
            )
            return original(agent, *args, **kwargs)

        tool_agent.ToolAgent._chat_completion = chat_completion

        clock = spec["clock"]
        overrides = {int(k): tuple(v) for k, v in (spec.get("clock_overrides") or {}).items()}
        self.clock = VirtualClock(float(clock["call_s"]), float(clock["token_s"]), clock.get("budget"), overrides)
        for text in self.results.values():  # the run's budget: elapsed + remaining of any recorded pair
            match = recording.CLOCK.search(text or "")
            if match and match.group(2) not in ("None", "null"):
                self.clock.budget = float(match.group(1)) + float(match.group(2))
                break
        # the harness's run clock and its limit are the virtual clock's (the play budget: the recording's)
        session_cls = solver_module._HarnessGameSession
        session_cls.timing_payload = lambda session: self.clock.payload()
        session_cls.runtime_limit_reached = lambda session: (
            self.clock.budget is not None and self.clock.elapsed >= self.clock.budget
        )

        names = recording.sandbox_dir_names(self.calls, self.results)
        self.dir_cursor = {"index": 0, "used": set()}

        def next_name() -> str | None:
            for index, name in names:
                if index >= self.dir_cursor["index"] and name not in self.dir_cursor["used"]:
                    self.dir_cursor["used"].add(name)
                    return name
            return None

        self.tempfile.source = next_name if names else None

        workdir = spec["workdir"]
        os.makedirs(workdir, exist_ok=True)
        solver = solver_module.HarnessSolver(
            label="arc3-kit-harness",
            model="local",
            analyzer_timeout=900.0,
            max_actions_per_game=None,
            # a finite play budget, as on Kaggle (the priority gate's tail fade needs one); read on the virtual clock
            max_runtime_s_per_game=self.clock.budget if self.clock.budget is not None else NO_BUDGET_S,
            concurrency=1,
            save_request_logs=False,
            start_local_server=False,
        )
        solver.job_dir = __import__("pathlib").Path(workdir)
        solver.minimal_diagnostics = True
        solver._setup()
        pool_gib = float(os.environ.get("ARC3_SANDBOX_MEMORY_POOL_GIB", "110") or 110)
        self.pool = MemoryPool(int(float(spec["memory_mib"]) * (1 << 20)), int(pool_gib * (1 << 30)))
        solver._sandbox_memory_pool = self.pool
        solver._stop_event.clear()
        # the warmup RESET before the first turn: the recording shows it as a first request after one action (a new
        # recording starts without it)
        solver._warmup_remaining = 1 if self.calls and int(self.calls[0].get("step") or 0) == 1 else 0
        make_analyzer = solver._make_analyzer

        def capture_analyzer(*args: Any, **kwargs: Any) -> Any:
            self.analyzer = make_analyzer(*args, **kwargs)
            self.harness_timeout = int(self.analyzer._python_timeout)
            return self.analyzer

        solver._make_analyzer = capture_analyzer
        self.solver = solver

        spec_game = taaf.game_api.ArcadeSpec(
            operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=spec["environments_dir"]
        )
        game = taaf.game_api.GameAPI(env_name=spec["game_id"], arcade_spec=spec_game)
        game.start_game(taaf.game.RunSession(record_intermediate_states=False))
        execute = game._execute_action

        def logged(action: Any) -> Any:
            state = execute(action)
            raw = state.raw
            self.steps.append(
                {
                    "name": action.id.name,
                    "data": dict(action.data),
                    "state": getattr(raw.state, "name", str(raw.state)),
                    "levels_completed": int(raw.levels_completed or 0),
                }
            )
            return state

        game._execute_action = logged
        self.game = game
        self.engine_game = game.env._game

    # ------------------------------------------------------------------ the harness's model
    def model_call(self, payload: dict[str, Any]) -> tuple:
        if self.stopped:
            raise Stop()
        k = self.k
        self.k += 1
        role = getattr(self.local, "role", "gameplay")
        continuation = bool(getattr(self.local, "continuation", False))
        if self.replaying:
            try:
                self._check(
                    k,
                    recording.with_effort(payload, recording.effort_of(self.calls[k]["payload"]))
                    if k < len(self.calls)
                    else payload,
                )
                if k < self.upto:
                    return self._recorded_reply(k)
            except ReplayError as exc:
                # reported here: the harness may catch what it raises through (a failed summary), the server must not
                # wait for it
                self.stopped = True
                self.to_main.put(("error", exc.info))
                raise Stop() from None
            self.replaying = False
            self.tempfile.source = None
            if self.analyzer is not None:
                self.analyzer._python_timeout = self.harness_timeout
        if recording.thinks(payload):
            payload = recording.with_effort(payload, self.live_effort)
        if self.writer is not None:
            self.writer.request(k, payload, role, continuation, self)
        self.to_main.put(("request", payload, role, continuation))
        item = self.to_game.get()
        if item[0] == "stop":
            self.stopped = True
            raise Stop()
        _, message, finish_reason, usage = item
        if self.writer is not None:
            self.writer.answered(finish_reason, usage)
        self.clock.advance(int((usage or {}).get("completion_tokens") or 0))
        return message, finish_reason, usage

    def _check(self, k: int, payload: dict[str, Any]) -> None:
        if k >= len(self.calls):
            raise ReplayError({"reason": "more requests than the recording", "request": k})
        want = self.calls[k]
        if recording.request_sha256(payload) != want.get("request_sha256"):
            # what the model sees must be equal; engine-only fields (a recording made on the Kaggle engine) are not
            got, wanted = recording.model_visible(payload), recording.model_visible(want["payload"])
            got_messages, want_messages = got.pop("messages", None) or [], wanted.pop("messages", None) or []
            params_equal = got == wanted
            if params_equal and got_messages == want_messages:
                self.engine_fields += 1
            elif params_equal and recording.masked_equal(got_messages, want_messages):
                self.masked += 1
            else:
                raise ReplayError(
                    {
                        "reason": "request differs from the recording",
                        "request": k,
                        "call": want.get("n"),
                        "params_equal": params_equal,
                        "diff": recording.first_difference(got_messages, want_messages),
                        "clock_overrides": self._clock_overrides(got_messages, want_messages),
                    }
                )
        if str(want.get("kind") or "policy") == "policy":
            frame = recording.frame_sha256(self.game.current_state.frame.data)
            step = len(self.game.game_run.history)
            if frame != want.get("frame_sha256") or step != int(want.get("step") or 0):
                raise ReplayError(
                    {
                        "reason": "board differs from the recording",
                        "request": k,
                        "step": step,
                        "want_step": want.get("step"),
                        "board_equal": frame == want.get("frame_sha256"),
                    }
                )

    def _clock_overrides(self, got: list[dict[str, Any]], want: list[dict[str, Any]]) -> dict[int, tuple]:
        """When the first differing tool result differs only in run-clock values: {clock read index: the recorded
        pair} for every read that returned a differing value (the next replay pins them)."""
        for g, w in zip(got, want):
            if g == w:
                continue
            gt, wt = str(g.get("content")), str(w.get("content"))
            gp, wp = recording.clock_pairs(gt), recording.clock_pairs(wt)
            if not gp or len(gp) != len(wp) or recording.without_clock(gt) != recording.without_clock(wt):
                return {}
            reads: dict[tuple, list[int]] = {}
            for i, (e, r) in enumerate(self.clock.log):
                reads.setdefault((repr(float(e)), "None" if r is None else repr(float(r))), []).append(i)
            out: dict[int, tuple] = {}
            for (ge, gr), (we, wr) in zip(gp, wp):
                if (ge, gr) == (we, wr):
                    continue
                indices = reads.get((ge, gr))
                if not indices:
                    return {}
                for i in indices:
                    out[i] = (float(we), None if wr in ("None", "null") else float(wr))
            return out
        return {}

    def _recorded_reply(self, k: int) -> tuple:
        recovered = recording.recorded_reply(self.calls, k)
        if recovered is None:
            raise ReplayError(
                {
                    "reason": "the recording has no reply to this request",
                    "request": k,
                    "kind": self.calls[k].get("kind"),
                }
            )
        reply, finish_reason = recovered
        ids = [str(tc.get("id", "")) for tc in reply.get("tool_calls") or []]
        texts = [self.results.get(i) for i in ids]
        # what the recording shows for this reply's tool calls, set before the harness runs them
        self.dir_cursor["index"] = k + 1
        limit = recording.memory_limit_bytes(texts)
        if limit:
            self.pool.limit_bytes = limit
            self.pool.total_bytes = max(self.pool.total_bytes, limit)
            old = getattr(self.analyzer, "_reported_sandbox_memory_limit", None)
            if old and f"{old / (1 << 20):.1f}" == f"{limit / (1 << 20):.1f}":
                self.analyzer._reported_sandbox_memory_limit = limit  # same rendering: not a change to report
        if self.analyzer is not None:
            timed_out = any(recording.TIMEOUT_TEXT in (t or "") for t in texts)
            self.analyzer._python_timeout = self.harness_timeout if timed_out else int(self.spec["replay_timeout_s"])
            guard = getattr(self.analyzer, "_python_retry_guard", None)
            match = recording.GUARD_CALL.search(texts[0] or "") if texts else None
            if match and guard is not None and hasattr(guard, "call"):
                guard.call = int(match.group(1)) - 1
        # the served response's usage and finish reason when the recording kept them (record.py keeps them); else the
        # finish reason the reply implies and its token count estimated from the text
        usage = self.calls[k].get("usage")
        finish_reason = self.calls[k].get("finish_reason") or finish_reason
        tokens = int(usage.get("completion_tokens") or 0) if usage else self._reply_tokens(k, reply)
        clock = next((m for m in (recording.CLOCK.search(t or "") for t in texts) if m), None)
        if clock:
            remaining = clock.group(2)
            self.clock.set(float(clock.group(1)), None if remaining in ("None", "null") else float(remaining))
        else:  # the clock moves by the reply's length (the text estimate when the recording has no usage)
            self.clock.advance(tokens if usage else recording.reply_tokens(reply))
        return dict(reply), finish_reason, dict(usage) if usage else {"completion_tokens": tokens}

    def _reply_tokens(self, k: int, reply: dict[str, Any]) -> int:
        """The generated tokens reported for recorded reply ``k`` (the recording keeps text only). The harness ends a
        turn once the turn's replies generated its yield budget (``LOCAL_ANALYZER_YIELD_TOKENS``), so the count is the
        text's estimate, clamped to cross the budget exactly where the recording yielded after this reply."""
        estimate = recording.reply_tokens(reply)
        budget = getattr(self.analyzer, "_yield_tokens", None)
        if budget is None:
            return estimate
        spent = int(getattr(self.analyzer, "_turn_generated_tokens", 0) or 0)
        if recording.yielded_after(self.calls, k):
            return max(estimate, budget - spent)
        return max(0, min(estimate, budget - 1 - spent))

    # ------------------------------------------------------------------ the game thread
    def _play(self) -> None:
        try:
            self.solver._play_one(self.game, 0, 0, None)
            run = self.game.game_run
            state = str(getattr(run, "state", None) or "ended")
            note = getattr(run, "solver_note", None)  # the harness's own note when it crashed
            self.to_main.put(("ended", f"{state}: {note}" if note else state))
        except Stop:
            pass
        except ReplayError as exc:
            self.to_main.put(("error", exc.info))
        except BaseException as exc:  # noqa: BLE001 -- reported to the server
            self.to_main.put(
                ("error", {"reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]})
            )

    def _boundary(self) -> dict[str, Any]:
        """Wait for the harness's next request (or its end) and describe the state it was sent from."""
        item = self.to_main.get()
        if item[0] == "error":
            raise ReplayError(item[1])
        out: dict[str, Any] = {"request": None, "role": None, "continuation": False, "ended": None}
        if item[0] == "request":
            out.update(request=item[1], role=item[2], continuation=item[3])
        else:
            self.ended = out["ended"] = str(item[1])
        out.update(
            action_count=len(self.game.game_run.history),
            level_index=int(self.engine_game.level_index),
            levels_completed=int(self.game.current_state.levels_completed),
            frame_sha256=recording.frame_sha256(self.game.current_state.frame.data),
        )
        return out

    def distance(self, level: int) -> int | None:
        return oracle.distance(self.engine_game, level)

    # ------------------------------------------------------------------ ops
    def start(self) -> dict[str, Any]:
        self.setup()
        self.thread = threading.Thread(target=self._play, name="harness-game", daemon=True)
        self.thread.start()
        out = self._boundary()
        if out["request"] is None:
            raise ReplayError({"reason": f"the game ended before the pivot ({out['ended']})", "request": self.k})
        out.update(
            masked=self.masked,
            engine_fields=self.engine_fields,
            replayed=self.upto,
            clock=self.clock.elapsed,
            reasoning_effort=self.live_effort,
        )
        return out

    def reply(self, message: dict[str, Any], finish_reason: str, usage: dict[str, Any] | None) -> dict[str, Any]:
        if self.ended is not None:
            raise RuntimeError(f"the game ended ({self.ended})")
        n0 = len(self.steps)
        self.to_game.put(("reply", message, finish_reason, usage))
        out = self._boundary()
        out["actions"] = self.steps[n0:]
        return out

    def close(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.to_game.put(("stop",))
            self.thread.join(timeout=30)
        if self.writer is not None:
            self.writer.close()


class Writer:
    """A new recording (``recording.SCHEMA``): a header, then before each request the actions taken since the previous
    one and the request (a ``call`` line: the payload as a delta of the previous request, the state it was sent from,
    and its reply's ``usage`` and ``finish_reason``). A call line is written once its reply came back (the last one,
    never answered, at close), so a replay gets each reply's exact token count."""

    def __init__(self, spec: dict[str, Any]):
        self.out = open(spec["record"], "w", encoding="utf-8")
        self.pending: dict[str, Any] | None = None
        self.actions = 0
        self.messages: list[dict[str, Any]] = []
        self.params: dict[str, Any] | None = None
        self.tools: set = set()
        self._line(
            {
                "t": "header",
                "schema": recording.SCHEMA,
                "attempt": spec.get("attempt"),
                "game_id": spec["game_id"],
                "harness_revision": spec.get("harness_revision"),
                "sandbox_seed": int(spec["sandbox_seed"]),
                "source": spec.get("source"),
            }
        )

    def _line(self, record: dict[str, Any]) -> None:
        self.out.write(json.dumps(record) + "\n")
        self.out.flush()

    def request(self, k: int, payload: dict[str, Any], role: str, continuation: bool, window: "Window") -> None:
        messages = list(payload.get("messages") or [])
        keep = 0
        while keep < min(len(messages), len(self.messages)) and messages[keep] == self.messages[keep]:
            keep += 1
        self.messages = messages
        params = {k2: (None if k2 in ("messages", "tools") else v) for k2, v in payload.items()}
        tools = payload.get("tools")
        tools_sha = hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest() if tools else None
        steps = window.steps
        self.pending = {
            "actions": [{"id": s["name"], "data": s["data"]} for s in steps[self.actions :]],
            "call": {
                "t": "call",
                "n": k,
                "kind": "continuation" if continuation else ("policy" if role == "gameplay" else "summary"),
                "level": int(window.engine_game.level_index) + 1,
                "levels_completed": int(window.game.current_state.levels_completed),
                "step": len(window.game.game_run.history),
                "frame_sha256": recording.frame_sha256(window.game.current_state.frame.data),
                "request_sha256": recording.request_sha256(payload),
                "messages_keep": keep,
                "messages_append": messages[keep:],
                "params": None if params == self.params else params,
                "tools_sha256": tools_sha,
                "tools": tools if tools_sha and tools_sha not in self.tools else None,
            },
        }
        self.params = params
        if tools_sha:
            self.tools.add(tools_sha)
        self.actions = len(steps)

    def answered(self, finish_reason: str | None, usage: dict[str, Any] | None) -> None:
        self._flush(finish_reason=finish_reason, usage=dict(usage) if usage else None)

    def _flush(self, **reply: Any) -> None:
        if self.pending is None:
            return
        if self.pending["actions"]:
            start = self.actions - len(self.pending["actions"])
            self._line({"t": "actions", "from": start, "rows": self.pending["actions"]})
        self._line({**self.pending["call"], **{k: v for k, v in reply.items() if v is not None}})
        self.pending = None

    def close(self) -> None:
        self._flush()
        self.out.close()


def main() -> None:
    conn = Connection(int(sys.argv[1]))
    window: Window | None = None
    while True:
        try:
            op, args = conn.recv()
        except (EOFError, OSError):
            break
        try:
            if op == "start":
                window = Window(args[0])
                value = window.start()
            elif op == "reply":
                value = window.reply(*args)
            elif op == "distance":
                value = window.distance(*args)
            elif op == "close":
                conn.send((True, None))
                break
            else:
                raise ValueError(f"unknown op {op!r}")
            conn.send((True, value))
        except ReplayError as exc:
            conn.send((False, dict(exc.info, kind="replay")))
        except BaseException as exc:  # noqa: BLE001 -- reported to the server
            conn.send(
                (
                    False,
                    {
                        "kind": "error",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-3000:],
                    },
                )
            )
    if window is not None:
        window.close()
    os._exit(0)  # the harness's daemon threads and sandbox children do not hold the process


if __name__ == "__main__":
    main()
