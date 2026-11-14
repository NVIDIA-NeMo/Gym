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

"""Wire-level tests for the prefix pass@K agent.

These cover the details that fail SILENTLY rather than loudly: an action that
does not parse, or an observation whose shape differs from the captured
harness, both leave a rollout that looks like a weak model instead of a broken
replay.
"""

import asyncio
import copy
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.sandbox.providers.base import SandboxExecResult


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGENT = _load("prefix_pass_k_agent_app", Path(__file__).parent.parent / "app.py")


class TestParseAction:
    def test_single_fence_returns_command(self):
        content = "THOUGHT: look around\n\n```mswea_bash_command\nls -la\n```"
        assert AGENT.parse_action(content) == "ls -la"

    def test_multiline_command_preserved(self):
        content = "```mswea_bash_command\ncd /testbed && python -c 'x=1\nprint(x)'\n```"
        assert AGENT.parse_action(content) == "cd /testbed && python -c 'x=1\nprint(x)'"

    def test_zero_fences_is_none(self):
        assert AGENT.parse_action("THOUGHT: I will think some more.") is None

    def test_two_fences_is_none(self):
        # The harness requires EXACTLY one action; two is a format error, not a
        # licence to pick the first.
        content = "```mswea_bash_command\nls\n```\nthen\n```mswea_bash_command\npwd\n```"
        assert AGENT.parse_action(content) is None

    def test_malformed_fence_is_none(self):
        # Observed in real traces: the model wrote ```mswea_bash_command] and
        # mini-swe-agent replied "Expected exactly 1 action, found 0". The
        # parser must AGREE with the harness, not be more lenient.
        content = "THOUGHT: go\n\n```mswea_bash_command]\ncd /testbed && grep -rn qop requests\n```"
        assert AGENT.parse_action(content) is None

    def test_empty_command_is_none(self):
        assert AGENT.parse_action("```mswea_bash_command\n\n```") is None


class TestRenderObservation:
    def test_short_output_inline(self):
        assert AGENT.render_observation(0, "hello\n") == "<returncode>0</returncode>\n<output>\nhello\n</output>"

    def test_nonzero_returncode_preserved(self):
        assert "<returncode>2</returncode>" in AGENT.render_observation(2, "boom")

    def test_empty_output_still_well_formed(self):
        assert AGENT.render_observation(0, "") == "<returncode>0</returncode>\n<output>\n</output>"

    def test_long_output_elided_with_head_and_tail(self):
        output = "A" * 4000 + "B" * 8000 + "C" * 4000  # 16000 chars
        rendered = AGENT.render_observation(0, output)
        assert "<warning>" in rendered
        assert "<elided_chars>\n6000 characters elided\n</elided_chars>" in rendered
        assert rendered.count("A") >= 4000  # head retained
        assert rendered.count("C") >= 4000  # tail retained
        assert "<output>" not in rendered  # elided form replaces the plain block

    def test_boundary_just_under_limit_is_inline(self):
        rendered = AGENT.render_observation(0, "x" * (AGENT.OBSERVATION_LIMIT - 1))
        assert "<warning>" not in rendered


class TestFormatErrorObservation:
    def test_reports_action_count(self):
        assert "found 0." in AGENT.format_error_observation(0)
        assert "found 2." in AGENT.format_error_observation(2)

    def test_matches_captured_harness_opening(self):
        # The captured traces open with exactly this text.
        assert AGENT.format_error_observation(0).startswith("Format error:\n\n<error>\nExpected exactly 1 action,")


class TestFenceAgreesWithDatasetBuilder:
    def test_agent_and_prepare_use_the_same_pattern(self):
        # The builder decides what counts as a replayable action and the agent
        # replays it. If these drift, the prefix stops matching the trajectory.
        prepare_src = (
            Path(__file__).parent.parent.parent.parent / "benchmarks" / "prefix_pass_k" / "prepare.py"
        ).read_text()
        assert f'ACTION_FENCE = re.compile(r"{AGENT.ACTION_FENCE.pattern}", re.DOTALL)' in prepare_src


class TestSubmitSentinel:
    def test_sentinel_matches_prompt_contract(self):
        # The task prompt instructs the model to submit by echoing this string.
        assert AGENT.SUBMIT_SENTINEL == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


# --- function-calling wire -----------------------------------------------------
# Fixtures are synthetic on purpose: the DeepSWE traces are evaluation-only and
# must not be copied into the repo.


class TestRenderToolObservation:
    """Expected strings are mini.yaml's observation_template rendered by Jinja on
    the same inputs. Its `tojson` is HTML-safe: ASCII-only, with <, >, & and '
    escaped, which plain json.dumps does not do."""

    def test_short_output_is_html_safe_ascii_json(self):
        assert AGENT.render_tool_observation(0, "a <b> & 'c' é\n") == (
            '{\n  "returncode": 0,\n  "output": "a \\u003cb\\u003e \\u0026 \\u0027c\\u0027 \\u00e9\\n"\n}'
        )

    def test_timeout_info_rides_on_the_last_field_line(self):
        info = "An error occurred while executing the command: Command 'sleep 40' timed out after 30 seconds"
        assert AGENT.render_tool_observation(-1, "partial\n", info) == (
            '{\n  "returncode": -1,\n  "output": "partial\\n", "exception_info": "An error occurred while '
            'executing the command: Command \\u0027sleep 40\\u0027 timed out after 30 seconds"\n}'
        )

    def test_long_output_elided_to_head_and_tail(self):
        rendered = AGENT.render_tool_observation(1, "x" * 6000 + "y" * 6000)
        assert json.loads(rendered) == {
            "returncode": 1,
            "output_head": "x" * 5000,
            "output_tail": "y" * 5000,
            "elided_chars": 2000,
            "warning": "Output too long.",
        }
        assert rendered.endswith('"warning": "Output too long."\n}')


class TestIsSubmission:
    def test_sentinel_alone_on_first_line_of_clean_exit(self):
        assert AGENT.is_submission(0, "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n")

    def test_leading_whitespace_tolerated(self):
        assert AGENT.is_submission(0, "\n  COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nrest")

    def test_nonzero_exit_is_not_a_submission(self):
        assert not AGENT.is_submission(1, "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n")

    def test_sentinel_further_down_is_not_a_submission(self):
        # e.g. a grep over a file that mentions it
        assert not AGENT.is_submission(0, "notes.md:3:\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n")


def _call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class TestToolCallCommands:
    def test_every_bash_call_in_order_with_its_id(self):
        calls = [_call("a", "bash", '{"command": "ls"}'), _call("b", "bash", '{"command": "pwd"}')]
        assert AGENT.tool_call_commands(calls) == [("a", "ls"), ("b", "pwd")]

    def test_unusable_calls_are_skipped(self):
        calls = [
            _call("a", "python", '{"command": "x"}'),
            _call("b", "bash", "{not json"),
            _call("c", "bash", '{"command": "  "}'),
            _call("d", "", '{"command": "ls"}'),  # blank placeholder left by some gateways
        ]
        assert AGENT.tool_call_commands(calls) == []


def _bare_agent(**config):
    """The agent without its web server: enough for the methods under test."""
    defaults = dict(
        wire="function_calling",
        exec_env={},
        interpreter=None,
        step_timeout=30,
        max_forwards=3,
        replay_through_target=False,
        debug=False,
        model_server=SimpleNamespace(name="policy_model"),
    )
    agent = AGENT.PrefixPassKAgent.__new__(AGENT.PrefixPassKAgent)
    object.__setattr__(agent, "config", SimpleNamespace(**(defaults | config)))
    object.__setattr__(agent, "_session_id_to_sandbox", {})
    object.__setattr__(agent, "_session_id_to_stats", {})
    return agent


def _result(stdout="", stderr=None, return_code=0, error_type=None):
    return SimpleNamespace(stdout=stdout, stderr=stderr, return_code=return_code, error_type=error_type)


class FakeSandbox:
    def __init__(self, *results):
        self.results = list(results)
        self.commands = []

    async def exec(self, command, timeout_s=None):
        self.commands.append(command)
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
class TestExecWrapper:
    """Run the wrapped command through a real bash, as the sandbox shell would."""

    def _run(self, agent, command):
        # PATH only. An inherited POSIXLY_CORRECT, or an exported SHELLOPTS with
        # `posix`, puts bash in POSIX mode, where a non-interactive shell never
        # reads BASH_ENV. CI passes its job's environment into the test container
        # (`srun --export=ALL`), and this test failed there exactly that way.
        return subprocess.run(
            ["bash", "-c", agent._wrap(command)], capture_output=True, text=True, env={"PATH": os.environ["PATH"]}
        )

    def test_interpreter_sources_bash_env_and_merges_stderr(self, tmp_path):
        bashrc = tmp_path / "bashrc"
        bashrc.write_text("export FROM_BASHRC=yes\n")
        agent = _bare_agent(exec_env={"BASH_ENV": str(bashrc), "PAGER": "cat"}, interpreter=["bash", "-c"])
        result = self._run(agent, "echo $FROM_BASHRC $PAGER; echo err >&2; echo done; exit 3")
        assert (result.stdout, result.stderr, result.returncode) == ("yes cat\nerr\ndone\n", "", 3)

    def test_quotes_heredocs_and_trailing_comments_survive(self):
        agent = _bare_agent(interpreter=["bash", "-c"])
        result = self._run(agent, "cat <<'EOF'\nit's \"quoted\" $HOME\nEOF\n# trailing comment")
        assert result.stdout == 'it\'s "quoted" $HOME\n'

    def test_without_interpreter_stderr_is_still_merged(self):
        agent = _bare_agent(exec_env={"LANG": "C.UTF-8"})
        result = self._run(agent, "echo $LANG; echo err >&2")
        assert (result.stdout, result.stderr) == ("C.UTF-8\nerr\n", "")


class TestRun:
    """What the sandbox returns is not what the harness showed; `_run` translates."""

    def test_sandbox_status_text_is_not_output(self):
        # OpenSandbox reports a nonzero exit as stderr="exit status 1"; the
        # harness never showed that under a grep that matched nothing.
        sandbox = FakeSandbox(_result(stdout="", stderr="exit status 1", return_code=1))
        assert asyncio.run(_bare_agent()._run(sandbox, "grep -r nope .")) == (1, "", None)

    def test_kill_at_the_timeout_reads_as_a_timeout(self):
        # OpenSandbox does not type timeouts: rc -1 and "signal: killed".
        sandbox = FakeSandbox(_result(stdout="partial\n", stderr="signal: killed", return_code=-1))
        assert asyncio.run(_bare_agent(step_timeout=0)._run(sandbox, "sleep 40")) == (
            -1,
            "partial\n",
            "An error occurred while executing the command: Command 'sleep 40' timed out after 0 seconds",
        )

    def test_early_kill_is_not_a_timeout(self):
        sandbox = FakeSandbox(_result(stderr="signal: killed", return_code=-1))
        assert asyncio.run(_bare_agent(step_timeout=3600)._run(sandbox, "kill -9 $$")) == (-1, "", None)

    @pytest.mark.parametrize("return_code", [-1, 124])
    def test_typed_timeout_quotes_the_raw_command(self, return_code: int) -> None:
        sandbox = FakeSandbox(_result(return_code=return_code, error_type="timeout"))
        _, _, info = asyncio.run(_bare_agent(exec_env={"PAGER": "cat"})._run(sandbox, "make test"))
        assert info.endswith("Command 'make test' timed out after 30 seconds")


class TestFunctionCallingReplay:
    def test_multi_call_turn_runs_every_call_and_gauges_fidelity(self):
        agent = _bare_agent()
        sandbox = FakeSandbox(_result(stdout="hi\n"))
        calls = [_call("a", "bash", '{"command": "echo hi"}'), _call("b", "bash", '{"command": "echo hi"}')]
        step = {
            "tool_calls": calls,
            "captured_observations": [AGENT.render_tool_observation(0, "hi\n"), "something else"],
        }
        messages, stats = [], {"prefix_observations_compared": 0, "prefix_observations_identical": 0}
        assert asyncio.run(agent._replay_step(sandbox, messages, step, stats)) is False
        assert len(sandbox.commands) == 2
        assert [m["role"] for m in messages] == ["assistant", "tool", "tool"]
        assert [m.get("tool_call_id") for m in messages[1:]] == ["a", "b"]
        assert stats == {"prefix_observations_compared": 2, "prefix_observations_identical": 1}

    def test_replay_override_runs_what_completed_and_shows_the_capture(self):
        # A captured timeout cut this command short; finishing it would change
        # the repository, so only the completed part runs.
        agent = _bare_agent()
        sandbox = FakeSandbox(_result(stdout="Saved working directory\n"))
        timeout_obs = '{\n  "returncode": -1,\n  "output": "Saved working directory\\n", "exception_info": "..."\n}'
        step = {
            "tool_calls": [_call("a", "bash", '{"command": "git stash && slow; git stash pop"}')],
            "captured_observations": [timeout_obs],
            "replay_commands": {"a": "git stash && slow"},
        }
        messages, stats = [], {"prefix_observations_compared": 0, "prefix_observations_identical": 0}
        asyncio.run(agent._replay_step(sandbox, messages, step, stats))
        assert len(sandbox.commands) == 1 and sandbox.commands[0].endswith("git stash && slow")
        assert messages[-1] == {"role": "tool", "tool_call_id": "a", "content": timeout_obs}
        assert stats["prefix_observations_compared"] == 0

    def test_unparseable_forward_is_resampled_from_the_same_transcript(self):
        # The reference bridge answers an unparseable generation with a 502 and
        # mini retries the identical request: the failed forward is spent, and
        # the next sample sees exactly the transcript the failed one saw.
        agent = _bare_agent()
        sandbox = FakeSandbox(_result(stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"))
        agent._session_id_to_sandbox["s"] = sandbox
        submit = _call("c", "bash", '{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}')
        replies = iter([({"content": '[{"name": "bash", "argu'}, "length"), ({"tool_calls": [submit]}, "tool_calls")])
        seen = []

        async def fake_generate_chat(request, messages, body):
            seen.append(copy.deepcopy(messages))
            return next(replies)

        object.__setattr__(agent, "_generate_chat", fake_generate_chat)
        request = SimpleNamespace(
            session={AGENT.SESSION_ID_KEY: "s"},
            state=SimpleNamespace(_ng_prefix_pass_k_row={"prefix_pass_k": {"target_turn": 1, "prefix": []}}),
        )
        body = SimpleNamespace(
            # As Responses input items arrive: chat completions rejects `type`/`phase`.
            input=[{"role": "user", "content": "task", "type": "message", "phase": None}],
            model=None,
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
        )
        asyncio.run(agent.responses(request, body))

        assert seen[0] == seen[1] == [{"role": "user", "content": "task"}]
        stats = agent._session_id_to_stats["s"]
        assert (stats["forwards"], stats["unparsed_forwards"], stats["truncated_forwards"]) == (2, 1, 1)
        assert stats["submitted"] is True
        # Both forwards are kept, the unparseable one included, so the rollout replays without the model.
        assert stats["candidate_turns"] == [
            {"content": '[{"name": "bash", "argu', "tool_calls": None, "finish_reason": "length"},
            {"content": None, "tool_calls": [submit], "finish_reason": "tool_calls"},
        ]

    def test_backticks_forwards_are_recorded(self):
        agent = _bare_agent(wire="backticks")
        agent._session_id_to_sandbox["s"] = FakeSandbox(_result(stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"))
        texts = iter(["no fence here", "```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"])

        async def fake_generate(request, messages, body):
            return next(texts), {}

        object.__setattr__(agent, "_generate", fake_generate)
        request = SimpleNamespace(
            session={AGENT.SESSION_ID_KEY: "s"},
            state=SimpleNamespace(_ng_prefix_pass_k_row={"prefix_pass_k": {"target_turn": 1, "prefix": []}}),
        )
        body = SimpleNamespace(
            input=[{"role": "user", "content": "task"}],
            model=None,
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
        )
        asyncio.run(agent.responses(request, body))

        stats = agent._session_id_to_stats["s"]
        assert stats["candidate_turns"] == [
            {"content": "no fence here"},
            {"content": "```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"},
        ]
        assert (stats["forwards"], stats["unparsed_forwards"], stats["submitted"]) == (2, 1, True)

    def test_context_overflow_ends_the_rollout_on_the_prefix(self, monkeypatch):
        # A transcript past the model's context fails identically on every
        # retry, so the rollout stops and is graded on the prefix's state.
        agent = _bare_agent()
        sandbox = FakeSandbox(_result())
        agent._session_id_to_sandbox["s"] = sandbox
        posts = []

        async def post(**kwargs):
            posts.append(kwargs)
            return SimpleNamespace()

        async def overflowing(response):
            error = AGENT.ClientResponseError(request_info=None, history=(), status=500)
            error.response_content = b"This model's maximum context length is 128000 tokens. However, ..."
            raise error

        object.__setattr__(agent, "server_client", SimpleNamespace(post=post))
        monkeypatch.setattr(AGENT, "raise_for_status", overflowing)
        request = SimpleNamespace(
            session={AGENT.SESSION_ID_KEY: "s"},
            cookies={},
            state=SimpleNamespace(_ng_prefix_pass_k_row={"prefix_pass_k": {"target_turn": 1, "prefix": []}}),
        )
        body = SimpleNamespace(
            input=[{"role": "user", "content": "task"}],
            model=None,
            temperature=None,
            top_p=None,
            max_output_tokens=None,
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
        )
        asyncio.run(agent.responses(request, body))

        stats = agent._session_id_to_stats["s"]
        assert len(posts) == 1 and sandbox.commands == []
        assert (stats["forwards"], stats["context_overflow"], stats["unparsed_forwards"]) == (1, True, 0)

    def test_other_model_errors_still_fail_the_rollout(self, monkeypatch):
        # An outage must surface, not be scored as a quiet prefix-only 0.
        agent = _bare_agent()
        agent._session_id_to_sandbox["s"] = FakeSandbox(_result())

        async def post(**kwargs):
            return SimpleNamespace()

        async def failing(response):
            error = AGENT.ClientResponseError(request_info=None, history=(), status=500)
            error.response_content = b"503 no active backend is ready for new requests"
            raise error

        object.__setattr__(agent, "server_client", SimpleNamespace(post=post))
        monkeypatch.setattr(AGENT, "raise_for_status", failing)
        request = SimpleNamespace(
            session={AGENT.SESSION_ID_KEY: "s"},
            cookies={},
            state=SimpleNamespace(_ng_prefix_pass_k_row={"prefix_pass_k": {"target_turn": 1, "prefix": []}}),
        )
        body = SimpleNamespace(
            input=[{"role": "user", "content": "task"}],
            model=None,
            temperature=None,
            top_p=None,
            max_output_tokens=None,
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
        )
        with pytest.raises(AGENT.ClientResponseError):
            asyncio.run(agent.responses(request, body))


class TestGitState:
    def test_counts_staged_untracked_and_head_movement(self):
        output = "abc123\nM  staged.py\n M unstaged.py\nA  added.py\n?? new.py\n?? patch.txt\n"
        assert AGENT.git_state(output, "base000") == {"head_moved": True, "staged_files": 2, "untracked_files": 2}

    def test_clean_tree_at_base(self):
        assert AGENT.git_state("base000\n", "base000") == {
            "head_moved": False,
            "staged_files": 0,
            "untracked_files": 0,
        }

    def test_unknown_base_never_reports_movement(self):
        assert AGENT.git_state("abc123\n", None)["head_moved"] is False


class RepoSandbox:
    """Runs commands in a real git repository, as the sandbox shell would."""

    ENV = {
        "PATH": os.environ["PATH"],
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }

    def __init__(self, cwd):
        self.cwd = cwd

    def run(self, command):
        return subprocess.run(["bash", "-c", command], cwd=self.cwd, capture_output=True, text=True, env=self.ENV)

    async def exec(self, command, timeout_s=None):
        done = self.run(command)
        return SimpleNamespace(stdout=done.stdout, stderr=done.stderr, return_code=done.returncode)


@pytest.mark.skipif(shutil.which("git") is None or shutil.which("bash") is None, reason="needs git and bash")
class TestUncommitWorktree:
    """A `git diff` collector must see what `git add -A && git diff --cached <base>` sees."""

    def _repo(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        repo = RepoSandbox(tmp_path)
        repo.run(
            "git init -q && printf 'a\\n' > kept.py && printf 'b\\n' > deleted.py && printf 'c\\n' > staged.py "
            "&& printf 'ignored.log\\n' > .gitignore && git add -A && git commit -qm base"
        )
        base = repo.run("git rev-parse HEAD").stdout.strip()
        # What rollouts did: commit an edit, stage another, delete a file, add a
        # file, and leave an ignored artifact behind.
        repo.run(
            "printf 'a2\\n' > kept.py && git commit -qam fix && printf 'c2\\n' > staged.py && git add staged.py "
            "&& rm deleted.py && printf 'new\\n' > new.py && printf 'x\\n' > ignored.log"
        )
        return repo, base

    def test_commits_staged_edits_deletions_and_new_files_reach_git_diff(self, tmp_path):
        repo, base = self._repo(tmp_path)
        # The reference view, taken on a copy of the index so the repository is untouched.
        probe = "GIT_INDEX_FILE=.git/probe-index"
        expected = set(
            repo.run(
                f"cp .git/index .git/probe-index && {probe} git add -A && {probe} git diff --cached --name-only {base}"
            ).stdout.split()
        )
        assert repo.run("git --no-pager diff --name-only").stdout.split() == ["deleted.py"]  # the collector, before

        asyncio.run(_bare_agent()._uncommit_worktree(repo, base))

        assert expected == {"deleted.py", "kept.py", "new.py", "staged.py"}
        assert set(repo.run("git --no-pager diff --name-only").stdout.split()) == expected
        assert "new file mode" in repo.run("git --no-pager diff -- new.py").stdout

    def test_unknown_base_leaves_the_repository_alone(self):
        sandbox = FakeSandbox(_result())
        asyncio.run(_bare_agent()._uncommit_worktree(sandbox, None))
        assert sandbox.commands == []

    def test_a_failing_reset_is_logged_not_raised(self):
        asyncio.run(_bare_agent()._uncommit_worktree(FakeSandbox(_result(stdout="fatal", return_code=128)), "abc"))

    def test_worktree_patch_is_the_whole_change_set_and_touches_nothing(self, tmp_path):
        repo, base = self._repo(tmp_path / "repo")
        patch = asyncio.run(_bare_agent()._worktree_patch(repo, base))

        # The grader's view is unchanged: the staged edit is still staged, the new file untracked.
        assert repo.run("git diff --cached --name-only").stdout.split() == ["staged.py"]
        assert repo.run("git ls-files --others --exclude-standard").stdout.split() == ["new.py"]
        # Applied to a fresh checkout of the starting commit, it reproduces the rollout's files.
        (tmp_path / "rollout.patch").write_text(patch)
        fresh = RepoSandbox(tmp_path)
        assert (
            fresh.run(f"git -C repo worktree add -q ../fresh {base} && git -C fresh apply ../rollout.patch").returncode
            == 0
        )
        for name in ("kept.py", "staged.py", "new.py"):
            assert (tmp_path / "fresh" / name).read_text() == (tmp_path / "repo" / name).read_text()
        assert not (tmp_path / "fresh" / "deleted.py").exists()
        assert not (tmp_path / "fresh" / "ignored.log").exists()

    def test_worktree_patch_without_a_base_is_none(self):
        assert asyncio.run(_bare_agent()._worktree_patch(FakeSandbox(_result()), None)) is None

    def test_an_unchanged_tree_is_an_empty_patch_not_a_missing_one(self):
        # OpenSandbox reports empty stdout as None; that is "no changes", not a failure.
        assert asyncio.run(_bare_agent()._worktree_patch(FakeSandbox(_result(stdout=None)), "abc")) == ""
        assert asyncio.run(_bare_agent()._worktree_patch(FakeSandbox(_result(return_code=128)), "abc")) is None


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
class TestTakeOffline:
    def test_resolver_points_at_a_closed_local_port(self, tmp_path, monkeypatch):
        resolv = tmp_path / "resolv.conf"
        resolv.write_text("search cluster.local\nnameserver 10.100.0.10\noptions ndots:1\n")
        monkeypatch.setattr(AGENT, "RESOLV_CONF", str(resolv))
        asyncio.run(_bare_agent()._take_offline(RepoSandbox(tmp_path)))
        assert resolv.read_text() == "nameserver 127.0.0.1\n"

    def test_a_sandbox_that_stays_online_fails_the_rollout(self):
        sandbox = FakeSandbox(_result(stdout="bash: /etc/resolv.conf: Read-only file system", return_code=1))
        with pytest.raises(RuntimeError, match="Read-only file system"):
            asyncio.run(_bare_agent()._take_offline(sandbox))


class TestContextOverflowWording:
    def test_both_vllm_wordings_match(self):
        assert AGENT.CONTEXT_OVERFLOW.search("This model's maximum context length is 128000 tokens. However, ...")
        assert AGENT.CONTEXT_OVERFLOW.search(
            "You passed 130049 input tokens and requested 1024 output tokens. However, the model's context "
            "length is only 131072 tokens, resulting in a maximum input length of 130048 tokens."
        )

    def test_oversized_max_tokens_is_not_an_overflow(self):
        # A misconfigured request, not a long transcript: it must fail loudly.
        assert not AGENT.CONTEXT_OVERFLOW.search(
            "max_tokens=140000 cannot be greater than max_model_len=max_total_tokens=131072."
        )


class Resetting:
    """A sandbox whose connection resets after `ok` successful commands."""

    def __init__(self, ok=0):
        self.ok = ok
        self.commands = []

    async def exec(self, command, timeout_s=None):
        self.commands.append(command)
        if len(self.commands) > self.ok:
            raise ConnectionResetError(104, "Connection reset by peer")
        return _result(stdout="fine\n")


class TestExecErrors:
    @pytest.mark.parametrize("return_code", [125, -1])
    def test_returned_sandbox_error_is_not_command_output_or_timeout(self, return_code: int) -> None:
        sandbox = FakeSandbox(SandboxExecResult(None, "execd failed to launch", return_code, "sandbox"))
        with pytest.raises(AGENT.SandboxExecError) as caught:
            asyncio.run(_bare_agent(step_timeout=0)._run(sandbox, "ls"))
        assert caught.value.failure_kind == "agent_run_error"
        assert "execd failed to launch" in str(caught.value)

    def test_a_reset_is_raised_not_answered_as_a_failed_command(self):
        with pytest.raises(AGENT.SandboxExecError) as caught:
            asyncio.run(_bare_agent()._run(Resetting(), "ls"))
        assert caught.value.failure_kind == "transport_peer_drop"
        assert "ConnectionResetError" in str(caught.value)

    def test_other_exec_failures_are_agent_run_errors(self):
        class Broken:
            async def exec(self, command, timeout_s=None):
                raise RuntimeError("provider returned garbage")

        with pytest.raises(AGENT.SandboxExecError) as caught:
            asyncio.run(_bare_agent()._run(Broken(), "ls"))
        assert caught.value.failure_kind == "agent_run_error"

    def test_the_first_failure_ends_the_attempt(self):
        # The prefix's second command fails to run: nothing after it may run, and
        # the candidate is never asked for a turn.
        agent = _bare_agent(wire="backticks")
        sandbox = Resetting(ok=1)
        agent._session_id_to_sandbox["s"] = sandbox

        async def must_not_generate(*args, **kwargs):
            raise AssertionError("the candidate ran after the sandbox failed")

        object.__setattr__(agent, "_generate", must_not_generate)
        prefix = [{"turn": i, "action": f"echo {i}", "content": f"cmd {i}"} for i in (1, 2, 3)]
        request = SimpleNamespace(
            session={AGENT.SESSION_ID_KEY: "s"},
            state=SimpleNamespace(_ng_prefix_pass_k_row={"prefix_pass_k": {"target_turn": 4, "prefix": prefix}}),
        )
        body = SimpleNamespace(
            input=[{"role": "user", "content": "task"}],
            model=None,
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
        )
        asyncio.run(agent.responses(request, body))

        stats = agent._session_id_to_stats["s"]
        assert len(sandbox.commands) == 2
        assert (stats["exec_errors"], stats["forwards"], stats["failure_kind"]) == (1, 0, "transport_peer_drop")


class TestAbortedAttemptIsMasked:
    """The review's case: a reset must not reach pass@k as an ordinary 0 (or a lucky 1)."""

    def _run(self, monkeypatch, verify, *, exec_result: SandboxExecResult | None = None):
        agent = _bare_agent(
            wire="backticks",
            resources_server=SimpleNamespace(name="rs"),
            record_git_state=True,
            commit_worktree=False,
            uncommit_worktree=False,
            offline=False,
        )
        sandbox = SimpleNamespace(stop=None, commands=[])

        async def stop():
            sandbox.commands.append("stop")

        sandbox.stop = stop

        async def connect(seed):
            return sandbox

        async def must_not_touch_git(*args, **kwargs):
            raise AssertionError("git step ran on the failed sandbox")

        async def head(sb):
            return "abc"

        async def exec_command(command, timeout_s=None):
            if exec_result is not None:
                return exec_result
            raise ConnectionResetError("Connection reset by peer")

        sandbox.exec = exec_command

        async def must_not_generate(*args, **kwargs):
            raise AssertionError("the candidate ran after the sandbox failed")

        class Reply:
            cookies = {}

            async def json(self):
                return {"sandbox_handle": "sb"}

        async def post(server_name, url_path, json, cookies):
            if url_path == "/verify":
                return verify(json)
            return Reply()

        object.__setattr__(agent, "server_client", SimpleNamespace(post=post))
        object.__setattr__(agent, "_connect_sandbox", connect)
        object.__setattr__(agent, "_head", head)
        object.__setattr__(agent, "_git_state", must_not_touch_git)
        object.__setattr__(agent, "_worktree_patch", must_not_touch_git)
        object.__setattr__(agent, "_generate", must_not_generate)

        async def ok(response):
            return None

        async def as_json(response):
            return response

        monkeypatch.setattr(AGENT, "raise_for_status", ok)
        monkeypatch.setattr(AGENT, "get_response_json", as_json)
        body = AGENT.PrefixPassKRunRequest.model_validate(
            {
                "responses_create_params": {"input": [{"role": "user", "content": "task"}]},
                "prefix_pass_k": {"prefix": [{"action": "ls", "content": "captured action"}]},
            }
        )
        request = SimpleNamespace(cookies={}, session={AGENT.SESSION_ID_KEY: "s"}, state=SimpleNamespace())
        result = asyncio.run(agent.run(request, body))
        return result, sandbox

    def test_a_verified_zero_is_masked(self, monkeypatch):
        result, sandbox = self._run(monkeypatch, lambda req: req | {"reward": 0.0})
        assert (result.reward, result.mask_sample, result.failure_kind) == (0.0, True, "transport_peer_drop")
        assert result.exec_errors == 1
        assert sandbox.commands == ["stop"]

    def test_a_lucky_pass_is_masked_too(self, monkeypatch):
        result, _ = self._run(monkeypatch, lambda req: req | {"reward": 1.0})
        assert (result.reward, result.mask_sample) == (0.0, True)

    @pytest.mark.parametrize("reward", [0.0, 1.0])
    def test_a_returned_sandbox_error_aborts_and_masks(self, monkeypatch, reward: float) -> None:
        result, sandbox = self._run(
            monkeypatch,
            lambda req: req | {"reward": reward},
            exec_result=SandboxExecResult(None, "execd failed to launch", 125, "sandbox"),
        )
        assert (result.exec_errors, result.forwards, result.reward, result.mask_sample) == (1, 0, 0.0, True)
        assert result.failure_kind == "agent_run_error"
        assert "execd failed to launch" in result.failure_reason
        assert sandbox.commands == ["stop"]

    def test_a_verifier_that_fails_on_the_broken_sandbox_still_yields_a_masked_sample(self, monkeypatch):
        def fails(req):
            raise RuntimeError("could not extract the patch")

        result, sandbox = self._run(monkeypatch, fails)
        assert (result.reward, result.mask_sample, result.failure_kind) == (0.0, True, "transport_peer_drop")
        assert sandbox.commands == ["stop"]


class TestSandboxIsReleasedOnEveryExit:
    """Fault injection: whatever ends an attempt, its sandbox is stopped and the session forgotten."""

    def _agent(self, monkeypatch, responses, verify):
        agent = _bare_agent(
            resources_server=SimpleNamespace(name="rs"),
            record_git_state=False,
            commit_worktree=False,
            uncommit_worktree=False,
            offline=False,
        )
        stops = []

        class Sandbox:
            async def stop(self):
                stops.append("stop")

        sandbox = Sandbox()

        async def connect(seed):
            return sandbox

        async def head(sb):
            return "abc"

        async def worktree_patch(sb, base):
            return ""

        class Reply:
            cookies = {}

            async def json(self):
                return {"sandbox_handle": "sb"}

        async def post(server_name, url_path, json, cookies):
            return verify(json) if url_path == "/verify" else Reply()

        async def ok(response):
            return None

        async def as_json(response):
            return response

        object.__setattr__(agent, "server_client", SimpleNamespace(post=post))
        object.__setattr__(agent, "_connect_sandbox", connect)
        object.__setattr__(agent, "_head", head)
        object.__setattr__(agent, "_worktree_patch", worktree_patch)
        object.__setattr__(agent, "responses", responses)
        monkeypatch.setattr(AGENT, "raise_for_status", ok)
        monkeypatch.setattr(AGENT, "get_response_json", as_json)
        body = AGENT.PrefixPassKRunRequest.model_validate(
            {"responses_create_params": {"input": [{"role": "user", "content": "task"}]}}
        )
        request = SimpleNamespace(cookies={}, session={AGENT.SESSION_ID_KEY: "s"}, state=SimpleNamespace())
        return agent, stops, request, body

    @staticmethod
    def _response():
        return AGENT.NeMoGymResponse(
            id="r",
            created_at=0,
            model="m",
            object="response",
            output=[],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
        )

    def _assert_released(self, agent, stops):
        assert stops == ["stop"]
        assert agent._session_id_to_sandbox == {} and agent._session_id_to_stats == {}

    def test_a_model_error(self, monkeypatch):
        async def responses(request, params):
            agent._session_id_to_stats["s"] = {"forwards": 1}
            raise RuntimeError("model server 500")

        agent, stops, request, body = self._agent(monkeypatch, responses, lambda req: req)
        with pytest.raises(RuntimeError, match="model server 500"):
            asyncio.run(agent.run(request, body))
        self._assert_released(agent, stops)

    def test_a_failed_verifier_request(self, monkeypatch):
        async def responses(request, params):
            return self._response()

        def verify(req):
            raise RuntimeError("verifier unreachable")

        agent, stops, request, body = self._agent(monkeypatch, responses, verify)
        with pytest.raises(RuntimeError, match="verifier unreachable"):
            asyncio.run(agent.run(request, body))
        self._assert_released(agent, stops)

    def test_a_cancelled_rollout(self, monkeypatch):
        async def responses(request, params):
            await asyncio.sleep(3600)

        agent, stops, request, body = self._agent(monkeypatch, responses, lambda req: req)

        async def cancel_mid_rollout():
            task = asyncio.ensure_future(agent.run(request, body))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_mid_rollout())
        self._assert_released(agent, stops)

    def test_a_completed_attempt(self, monkeypatch):
        async def responses(request, params):
            return self._response()

        agent, stops, request, body = self._agent(monkeypatch, responses, lambda req: req | {"reward": 1.0})
        result = asyncio.run(agent.run(request, body))
        assert result.reward == 1.0
        self._assert_released(agent, stops)
