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
"""Tests for `nemo_gym.intake`, driven against a stand-in for the data layer.

The stand-in is a real aiohttp app serving the routes the intake schema declares, so the client
is exercised over real HTTP with real status codes. The shapes each route answers with are the
ones the deployed service answered with when this client was written against it.
"""

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from nemo_gym import intake
from nemo_gym.intake import IntakeError


pytestmark = pytest.mark.unit

# A locator whose path the stand-in accepts, shaped like the ones the real service takes.
GOOD_LOCATOR = "cluster-a:/shared/rundirs/someone/20260914_174556-6481"
REFUSED_PATH = '{"error":"path is outside the roots this service accepts"}'

# The three states one job walks through, one body per poll.
WALKS_TO_DEDUP = [
    {"status": "queued", "parse_state": None},
    {"status": "claimed", "parse_state": None},
    {"status": "done", "parse_state": "parsed", "run_ids": [], "deduped": ["b3bc6825e392e437"]},
]


class FakeDataLayer:
    """The intake surface of a benchmark-run data layer, as far as this client uses it."""

    def __init__(self, job_states: list[dict] | None = None):
        self.job_states = job_states or WALKS_TO_DEDUP
        self.polls = 0
        self.submitted_payload: dict | None = None

        app = web.Application()
        app.add_routes(
            [
                web.post("/v1/intake", self.post_intake),
                web.get("/v1/intake/jobs/{job_id}", self.get_job),
            ]
        )
        self.server = TestServer(app)

    async def __aenter__(self) -> "FakeDataLayer":
        await self.server.start_server()
        self.base_url = str(self.server.make_url("")).rstrip("/")
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.server.close()

    async def post_intake(self, request: web.Request) -> web.Response:
        self.submitted_payload = await request.json()
        _, _, path = self.submitted_payload["lustre"].partition(":")
        if not path.startswith("/shared/"):
            return web.Response(status=400, text=REFUSED_PATH, content_type="application/json")
        return web.json_response({"job_id": "job-uuid", "status": "queued"}, status=202)

    async def get_job(self, request: web.Request) -> web.Response:
        if request.match_info["job_id"] != "job-uuid":
            return web.json_response({"error": "no such upload uuid"}, status=404)
        body = self.job_states[min(self.polls, len(self.job_states) - 1)]
        self.polls += 1
        return web.json_response(body)


class TestLocator:
    def test_a_locator_splits_into_cluster_and_path(self) -> None:
        assert intake.parse_lustre_locator("cluster-a:/shared/a/b") == ("cluster-a", "/shared/a/b")

    @pytest.mark.parametrize("locator", ["no-colon", ":/shared/a", "cluster-a:", "cluster-a:relative/path"])
    def test_malformed_locators_are_refused_before_any_request(self, locator: str) -> None:
        with pytest.raises(IntakeError, match="Malformed locator"):
            intake.parse_lustre_locator(locator)


class TestConfigDiscovery:
    def test_explicit_base_url_wins_and_loses_its_trailing_slash(self, monkeypatch) -> None:
        monkeypatch.setenv(intake.BASE_URL_ENV_VAR_NAME, "https://from-env")
        assert intake.resolve_base_url("https://explicit/") == "https://explicit"

    def test_env_var_is_the_fallback(self, monkeypatch) -> None:
        monkeypatch.setenv(intake.BASE_URL_ENV_VAR_NAME, "https://from-env/")
        assert intake.resolve_base_url(None) == "https://from-env"

    def test_no_service_configured_names_both_ways_to_configure_one(self, monkeypatch) -> None:
        monkeypatch.delenv(intake.BASE_URL_ENV_VAR_NAME, raising=False)
        with pytest.raises(IntakeError, match=f"--base-url.*{intake.BASE_URL_ENV_VAR_NAME}"):
            intake.resolve_base_url(None)


class TestStatusNormalisation:
    def test_the_decode_state_is_read_rather_than_the_transport_state(self) -> None:
        outcome = intake._read_status(
            {"status": "done", "parse_state": "parsed", "run_ids": [], "deduped": ["b3bc6825e392e437"]}
        )
        assert (outcome.state, outcome.run_ids, outcome.succeeded) == ("deduped", ["b3bc6825e392e437"], True)

    def test_a_fresh_commit_is_reported_as_one(self) -> None:
        outcome = intake._read_status({"status": "done", "parse_state": "parsed", "run_ids": ["fresh"]})
        assert (outcome.state, outcome.run_ids, outcome.succeeded) == ("committed", ["fresh"], True)

    def test_an_in_flight_state_is_passed_through_and_is_not_terminal(self) -> None:
        outcome = intake._read_status({"status": "claimed", "parse_state": None})
        assert outcome.state == "claimed" and not outcome.succeeded
        assert outcome.state not in intake._TERMINAL_STATES

    def test_a_failure_carries_the_services_reason(self) -> None:
        outcome = intake._read_status({"status": "done", "parse_state": "failed", "reason": "unparseable"})
        assert outcome.state == "failed" and not outcome.succeeded
        assert outcome.detail["reason"] == "unparseable"

    def test_a_body_naming_no_state_at_all_is_unknown_rather_than_a_success(self) -> None:
        assert intake._read_status({}).state == "unknown"


class TestSubmit:
    async def test_it_waits_through_the_real_intermediate_states_and_reports_the_dedup(self) -> None:
        async with FakeDataLayer() as service:
            submission, outcome = await intake.run_intake(
                base_url=service.base_url, locator=GOOD_LOCATOR, label="my-label", interval_s=0
            )

        assert service.submitted_payload == {"lustre": GOOD_LOCATOR, "label": "my-label"}
        assert submission.id == "job-uuid"
        assert submission.status_path == "/v1/intake/jobs/job-uuid"
        assert service.polls == 3, "the client must keep polling while the job is queued and claimed"
        assert (outcome.state, outcome.run_ids) == ("deduped", ["b3bc6825e392e437"])

    async def test_a_refusal_surfaces_the_services_own_words(self) -> None:
        async with FakeDataLayer() as service:
            with pytest.raises(IntakeError) as caught:
                await intake.run_intake(base_url=service.base_url, locator="cluster-a:/elsewhere/a/b/c")
        assert "roots this service accepts" in str(caught.value)
        assert "400" in str(caught.value)

    async def test_no_label_is_sent_when_none_was_given(self) -> None:
        async with FakeDataLayer() as service:
            await intake.run_intake(base_url=service.base_url, locator=GOOD_LOCATOR, wait=False)
        assert service.submitted_payload == {"lustre": GOOD_LOCATOR}

    async def test_no_wait_returns_the_handle_without_polling(self) -> None:
        async with FakeDataLayer() as service:
            submission, outcome = await intake.run_intake(base_url=service.base_url, locator=GOOD_LOCATOR, wait=False)
        assert outcome is None and service.polls == 0
        assert submission.id == "job-uuid"

    async def test_an_accept_naming_no_job_id_is_an_error_not_a_silent_success(self) -> None:
        async def accept_without_an_id(_request: web.Request) -> web.Response:
            return web.json_response({"status": "queued"}, status=202)

        app = web.Application()
        app.add_routes([web.post("/v1/intake", accept_without_an_id)])
        server = TestServer(app)
        await server.start_server()
        try:
            async with intake.new_session() as session:
                with pytest.raises(IntakeError, match="named no job id"):
                    await intake.submit(session, str(server.make_url("")).rstrip("/"), GOOD_LOCATOR)
        finally:
            await server.close()


class TestPolling:
    async def test_running_out_of_time_reports_nothing_rather_than_a_guess(self) -> None:
        async with FakeDataLayer(job_states=[{"status": "queued", "parse_state": None}]) as service:
            async with intake.new_session() as session:
                submission = await intake.submit(session, service.base_url, GOOD_LOCATOR)
                outcome = await intake.poll(session, service.base_url, submission, timeout_s=0.2, interval_s=0.05)
        assert outcome is None

    async def test_an_unknown_id_is_terminal_rather_than_polled_forever(self) -> None:
        async with FakeDataLayer() as service:
            async with intake.new_session() as session:
                submission = intake.Submission(id="ghost", status_path="/v1/intake/jobs/ghost", response={})
                outcome = await intake.poll(session, service.base_url, submission, interval_s=0)
        assert outcome.state == "failed"
        assert "no record" in outcome.detail["reason"]

    async def test_the_wait_callback_sees_each_intermediate_state(self) -> None:
        seen: list[str] = []
        async with FakeDataLayer() as service:
            await intake.run_intake(
                base_url=service.base_url,
                locator=GOOD_LOCATOR,
                interval_s=0,
                on_wait=lambda _waited, outcome: seen.append(outcome.state),
            )
        assert seen == ["queued", "claimed"]


class TestFetchStatus:
    async def test_a_known_job_is_read_back(self) -> None:
        committed = [{"status": "done", "parse_state": "parsed", "run_ids": ["r"]}]
        async with FakeDataLayer(job_states=committed) as service:
            async with intake.new_session() as session:
                outcome = await intake.fetch_status(session, service.base_url, "job-uuid")
        assert (outcome.state, outcome.run_ids) == ("committed", ["r"])

    async def test_an_unknown_job_is_named_in_the_error(self) -> None:
        async with FakeDataLayer() as service:
            async with intake.new_session() as session:
                with pytest.raises(IntakeError, match="stranger"):
                    await intake.fetch_status(session, service.base_url, "stranger")


class TestCommandLine:
    """The `gym eval intake` front end: argv in, exit code and printed payload out.

    The transport is stubbed here on purpose. What this layer owns is the mapping from a settled
    outcome to an exit code and to what lands on stdout; the HTTP behaviour behind it is covered
    over real HTTP in the classes above.
    """

    @staticmethod
    def _args(**overrides):
        import argparse

        defaults = dict(
            base_url="https://data-layer",
            lustre=None,
            status=None,
            label=None,
            no_wait=False,
            timeout=None,
            json=True,
        )
        return argparse.Namespace(**{**defaults, **overrides})

    @staticmethod
    def _settles_on(monkeypatch, outcome, *, calls=None):
        """Stub the transport so `run_intake` returns `outcome` for a submission that was accepted."""
        from nemo_gym.cli import intake as cli

        submission = intake.Submission(id="job-uuid", status_path="/v1/intake/jobs/job-uuid", response={})

        async def fake_run_intake(**kwargs):
            if calls is not None:
                calls.append(kwargs)
            return submission, outcome

        monkeypatch.setattr(cli.intake_api, "run_intake", fake_run_intake)
        return cli

    def test_a_committed_intake_exits_zero_and_prints_the_run_ids(self, monkeypatch, capsys) -> None:
        committed = intake.IntakeOutcome(state="committed", run_ids=["fresh-run"], detail={})
        cli = self._settles_on(monkeypatch, committed)

        with pytest.raises(SystemExit) as exited:
            cli.intake(self._args(lustre=GOOD_LOCATOR))

        assert exited.value.code == 0
        assert json.loads(capsys.readouterr().out) == {
            "job_id": "job-uuid",
            "status_path": "/v1/intake/jobs/job-uuid",
            "state": "committed",
            "run_ids": ["fresh-run"],
            "detail": {},
        }

    def test_a_dedup_also_exits_zero_because_the_run_is_in_the_catalog(self, monkeypatch, capsys) -> None:
        deduped = intake.IntakeOutcome(state="deduped", run_ids=["existing"], detail={})
        cli = self._settles_on(monkeypatch, deduped)

        with pytest.raises(SystemExit) as exited:
            cli.intake(self._args(lustre=GOOD_LOCATOR, json=False))

        assert exited.value.code == 0
        assert "existing" in capsys.readouterr().out

    def test_a_refused_parse_exits_with_the_failure_code(self, monkeypatch, capsys) -> None:
        failed = intake.IntakeOutcome(state="failed", run_ids=[], detail={"reason": "unparseable"})
        cli = self._settles_on(monkeypatch, failed)

        with pytest.raises(SystemExit) as exited:
            cli.intake(self._args(lustre=GOOD_LOCATOR, json=False))

        assert exited.value.code == cli.EXIT_FAILED
        assert "failed" in capsys.readouterr().out

    def test_a_parse_that_outlasts_the_timeout_exits_distinctly_from_a_failure(self, monkeypatch, capsys) -> None:
        cli = self._settles_on(monkeypatch, None)

        with pytest.raises(SystemExit) as exited:
            cli.intake(self._args(lustre=GOOD_LOCATOR, timeout=30, json=False))

        assert exited.value.code == cli.EXIT_STILL_PARSING
        out = capsys.readouterr().out
        assert "still parsing after 30s" in out
        assert "/v1/intake/jobs/job-uuid" in out, "a caller that ran out of patience needs the poll URL"

    def test_no_wait_exits_zero_and_names_where_to_poll(self, monkeypatch, capsys) -> None:
        calls: list[dict] = []
        cli = self._settles_on(monkeypatch, None, calls=calls)

        with pytest.raises(SystemExit) as exited:
            cli.intake(self._args(lustre=GOOD_LOCATOR, no_wait=True, json=False))

        assert exited.value.code == 0
        assert calls[0]["wait"] is False
        out = capsys.readouterr().out
        assert "submitted" in out and "/v1/intake/jobs/job-uuid" in out

    def test_the_locator_label_and_default_timeout_reach_the_transport(self, monkeypatch) -> None:
        calls: list[dict] = []
        cli = self._settles_on(monkeypatch, None, calls=calls)

        with pytest.raises(SystemExit):
            cli.intake(self._args(lustre=GOOD_LOCATOR, label="my-label"))

        assert calls[0]["locator"] == GOOD_LOCATOR
        assert calls[0]["label"] == "my-label"
        assert calls[0]["timeout_s"] == intake.DEFAULT_POLL_TIMEOUT_S

    def test_the_wait_reporter_writes_to_stderr_and_leaves_stdout_alone(self, monkeypatch, capsys) -> None:
        """The progress line must not land on stdout, or `--json` stops being parseable."""
        calls: list[dict] = []
        cli = self._settles_on(monkeypatch, None, calls=calls)

        with pytest.raises(SystemExit):
            cli.intake(self._args(lustre=GOOD_LOCATOR, no_wait=True))
        capsys.readouterr()

        waiting = intake.IntakeOutcome(state="claimed", run_ids=[], detail={})
        calls[0]["on_wait"](10.0, waiting)
        captured = capsys.readouterr()
        assert "claimed" in captured.err and "10s" in captured.err
        assert captured.out == ""

    def test_a_malformed_locator_is_refused_without_a_traceback(self, capsys) -> None:
        from nemo_gym.cli import intake as cli

        with pytest.raises(SystemExit) as exited:
            cli.intake(self._args(lustre="no-colon-here"))
        assert exited.value.code == cli.EXIT_REFUSED
        assert "Malformed locator" in capsys.readouterr().err

    def test_no_configured_service_is_refused_without_a_traceback(self, monkeypatch, capsys) -> None:
        from nemo_gym.cli import intake as cli

        monkeypatch.delenv(intake.BASE_URL_ENV_VAR_NAME, raising=False)
        with pytest.raises(SystemExit) as exited:
            cli.intake(self._args(base_url=None, status="anything"))
        assert exited.value.code == cli.EXIT_REFUSED
        assert intake.BASE_URL_ENV_VAR_NAME in capsys.readouterr().err

    def test_status_reports_a_known_job_and_exits_zero(self, monkeypatch, capsys) -> None:
        from nemo_gym.cli import intake as cli

        async def fake_fetch_status(_session, _base_url, job_id):
            assert job_id == "job-uuid"
            return intake.IntakeOutcome(state="committed", run_ids=["fresh-run"], detail={})

        monkeypatch.setattr(cli.intake_api, "fetch_status", fake_fetch_status)
        with pytest.raises(SystemExit) as exited:
            cli.intake(self._args(status="job-uuid", json=False))

        assert exited.value.code == 0
        assert "fresh-run" in capsys.readouterr().out

    def test_status_on_a_job_that_failed_exits_with_the_failure_code(self, monkeypatch) -> None:
        from nemo_gym.cli import intake as cli

        async def fake_fetch_status(_session, _base_url, _job_id):
            return intake.IntakeOutcome(state="failed", run_ids=[], detail={})

        monkeypatch.setattr(cli.intake_api, "fetch_status", fake_fetch_status)
        with pytest.raises(SystemExit) as exited:
            cli.intake(self._args(status="job-uuid"))
        assert exited.value.code == cli.EXIT_FAILED

    def test_a_service_refusal_is_reported_as_a_refusal_not_a_crash(self, monkeypatch, capsys) -> None:
        from nemo_gym.cli import intake as cli

        async def refuse(**_kwargs):
            raise IntakeError("400 Bad Request from /v1/intake: nope")

        monkeypatch.setattr(cli.intake_api, "run_intake", refuse)
        with pytest.raises(SystemExit) as exited:
            cli.intake(self._args(lustre=GOOD_LOCATOR))
        assert exited.value.code == cli.EXIT_REFUSED
        assert "400 Bad Request" in capsys.readouterr().err


def test_the_module_bakes_in_no_service_hostname() -> None:
    """A deployment's hostname belongs in its own configuration, not in this repository."""
    from pathlib import Path

    source = Path(intake.__file__).read_text()
    assert "https://" not in source.replace("http://www.apache.org", "")


def _leaf_help(group: str, command: str) -> str:
    """The `--help` of one `gym <group> <command>` leaf, as a user would read it."""
    from nemo_gym.cli.main import build_parser

    def _pick(parser, name: str):
        for action in parser._actions:
            if name in (getattr(action, "choices", None) or {}):
                return action.choices[name]
        raise AssertionError(f"no {name!r} subcommand")

    return _pick(_pick(build_parser(), group), command).format_help()


def test_the_cli_help_names_the_env_var_the_module_actually_reads() -> None:
    intake_help = _leaf_help("eval", "intake")
    assert intake.BASE_URL_ENV_VAR_NAME in intake_help
    assert f"{intake.DEFAULT_POLL_TIMEOUT_S:.0f}" in intake_help


def test_intake_refuses_hydra_overrides_it_cannot_apply() -> None:
    import argparse

    from nemo_gym.cli.main import _eval_intake

    parser = argparse.ArgumentParser()
    args = argparse.Namespace(_parser=parser, json=False)
    with pytest.raises(SystemExit):
        _eval_intake(args, ["+policy_model_name=gpt-4"])


def test_the_poll_loop_sleeps_between_polls_rather_than_spinning() -> None:
    """`interval_s=0` is a test affordance; the shipped default must not busy-poll a service."""
    assert intake.DEFAULT_POLL_INTERVAL_S >= 1.0
    assert asyncio.iscoroutinefunction(intake.poll)
