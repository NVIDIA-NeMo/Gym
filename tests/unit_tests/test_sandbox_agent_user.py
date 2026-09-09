# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from nemo_gym.sandbox.agent_user import check_agent_user, is_root_agent_user, normalize_agent_user


AGENT_USER_ERROR = "agent_user must be an account name, a uid, or null"
NORMALIZED_AGENT_USERS = [
    ("1000", 1000),
    ("0", 0),
    ("agent", "agent"),
    ("root", "root"),
    (1000, 1000),
    (0, 0),
    (None, None),
    # isdecimal, not isdigit: a superscript digit is not a uid and must not make int() raise.
    ("²", "²"),
]
# Booleans (pydantic lax mode would coerce `true` to uid 1) and empty / option-like names (`su` would parse them
# as options; shlex.quote leaves "-m" unquoted).
REJECTED_AGENT_USERS = [True, False, "", "-m", "--login", "-"]


class FakeExecutor:
    """Records `(command, user)` for every exec and answers `id` probes from a scripted table."""

    def __init__(self, root_result=None, agent_result=None):
        self.calls = []
        self.root_result = (
            root_result if root_result is not None else SimpleNamespace(stdout="0\n", stderr="", return_code=0)
        )
        self.agent_result = (
            agent_result
            if agent_result is not None
            else SimpleNamespace(stdout="1000\n1000\n", stderr="", return_code=0)
        )

    async def exec(self, command, *, user=None):
        self.calls.append((command, user))
        if command == "id -u":
            return self.root_result
        if command == "id -u && id -g":
            return self.agent_result
        raise AssertionError(f"unexpected command {command!r}")


@pytest.mark.parametrize(("value", "expected"), NORMALIZED_AGENT_USERS)
def test_normalize_agent_user_accepts_names_uids_and_null(value, expected):
    normalized = normalize_agent_user(value)
    assert normalized == expected
    assert type(normalized) is type(expected)


@pytest.mark.parametrize("value", REJECTED_AGENT_USERS)
def test_normalize_agent_user_rejects_bools_and_option_like_names(value):
    with pytest.raises(ValueError, match=AGENT_USER_ERROR):
        normalize_agent_user(value)


def test_normalize_agent_user_passes_other_types_through_for_pydantic():
    value = 1.5
    assert normalize_agent_user(value) is value


@pytest.mark.parametrize(
    ("agent_user", "expected"),
    [(None, True), ("root", True), (0, True), ("agent", False), (1000, False), ("0", False), ("Root", False)],
)
def test_is_root_agent_user(agent_user, expected):
    assert is_root_agent_user(agent_user) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_user", ["agent", 1000])
async def test_check_agent_user_accepts_non_root_identity(agent_user):
    executor = FakeExecutor()

    await check_agent_user(executor, agent_user)

    assert executor.calls == [("id -u", "root"), ("id -u && id -g", agent_user)]


@pytest.mark.asyncio
async def test_check_agent_user_accepts_name_whose_uid_differs_from_any_int():
    # A name is checked for non-root only; the uid match applies to int identities exclusively.
    executor = FakeExecutor(agent_result=SimpleNamespace(stdout="1001\n1001\n", stderr="", return_code=0))

    await check_agent_user(executor, "agent")

    assert executor.calls == [("id -u", "root"), ("id -u && id -g", "agent")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_user", "root_result", "agent_result", "message"),
    [
        ("agent", ("1000\n", 0), ("1000\n1000\n", 0), "image default user must be root"),
        ("agent", ("", 1), ("1000\n1000\n", 0), "image default user must be root"),
        ("agent", ("0\n", 1), ("1000\n1000\n", 0), "image default user must be root"),
        ("agent", (None, 0), ("1000\n1000\n", 0), "image default user must be root"),
        ("agent", ("0\n", 0), ("", 1), "could not run a command as 'agent'"),
        ("agent", ("0\n", 0), ("1000\n1000\n", 1), "could not run a command as 'agent'"),
        ("agent", ("0\n", 0), ("1000\n", 0), "could not run a command as 'agent'"),
        ("agent", ("0\n", 0), ("1000\n1000\n1000\n", 0), "could not run a command as 'agent'"),
        ("agent", ("0\n", 0), (None, 0), "could not run a command as 'agent'"),
        ("agent", ("0\n", 0), ("0\n0\n", 0), "still resolve to uid 0"),
        ("agent", ("0\n", 0), ("0\n1000\n", 0), "still resolve to uid 0"),
        ("agent", ("0\n", 0), ("1000\n0\n", 0), "resolve to gid 0"),
        (1000, ("0\n", 0), ("1001\n1001\n", 0), "run as uid 1000 resolved to uid 1001"),
        (1000, ("0\n", 0), ("0\n0\n", 0), "still resolve to uid 0"),
    ],
)
async def test_check_agent_user_rejects_each_violation(agent_user, root_result, agent_result, message):
    root_stdout, root_rc = root_result
    agent_stdout, agent_rc = agent_result
    executor = FakeExecutor(
        root_result=SimpleNamespace(stdout=root_stdout, stderr="err", return_code=root_rc),
        agent_result=SimpleNamespace(stdout=agent_stdout, stderr="err", return_code=agent_rc),
    )

    with pytest.raises(RuntimeError, match=message) as excinfo:
        await check_agent_user(executor, agent_user)

    text = str(excinfo.value)
    assert f"agent_user={agent_user!r} identity check failed" in text
    assert "stderr='err'" in text
    assert executor.calls[0] == ("id -u", "root")
    if "image default user must be root" in message:
        # Root violation short-circuits: nothing is ever run as the agent identity.
        assert executor.calls == [("id -u", "root")]
        assert f"return_code={root_rc} stdout={root_stdout!r}" in text
    else:
        assert executor.calls == [("id -u", "root"), ("id -u && id -g", agent_user)]
        assert f"return_code={agent_rc} stdout={agent_stdout!r}" in text


@pytest.mark.asyncio
async def test_check_agent_user_tolerates_none_stderr_in_error_text():
    executor = FakeExecutor(agent_result=SimpleNamespace(stdout=None, stderr=None, return_code=1))

    with pytest.raises(RuntimeError, match="could not run a command as 'agent'") as excinfo:
        await check_agent_user(executor, "agent")

    assert "return_code=1 stdout=None stderr=None" in str(excinfo.value)
