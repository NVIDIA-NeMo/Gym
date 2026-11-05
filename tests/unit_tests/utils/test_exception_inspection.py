# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from nemo_gym.utils.exception_inspection import provider_error_reason, sandbox_failure_prefix


def test_the_sandbox_prefix_names_known_facts_in_a_fixed_order_and_leaves_out_unknown_ones() -> None:
    assert (
        sandbox_failure_prefix(
            phase="agent",
            provider="opensandbox",
            sandbox_id="box-1",
            reason="SANDBOX::ENDED",
            http_status=410,
            exit_code=137,
            signal="SIGKILL",
            command="opencode_run",
        )
        == "[sandbox phase=agent provider=opensandbox sandbox=box-1 reason=SANDBOX::ENDED http=410 exit=137 "
        "signal=SIGKILL command=opencode_run] "
    )
    assert sandbox_failure_prefix(phase="verify", sandbox_id=None, reason="") == "[sandbox phase=verify] "


def test_each_sandbox_fact_is_one_bounded_token() -> None:
    prefix = sandbox_failure_prefix(phase="agent", reason="Evicted: a=b [pod]\nlimit", sandbox_id="x" * 300)
    assert prefix == "[sandbox phase=agent sandbox=" + "x" * 128 + " reason=Evicted:_a_b_pod_limit] "


def test_a_provider_error_reason_is_its_code_else_its_type() -> None:
    class SandboxApiException(Exception):
        pass

    coded = SandboxApiException("ended")
    coded.error = SimpleNamespace(code="SANDBOX::ENDED")
    assert provider_error_reason(coded) == "SANDBOX::ENDED"
    assert provider_error_reason(TimeoutError()) == "TimeoutError"
    uncoded = SandboxApiException("x")
    uncoded.error = SimpleNamespace(code=None)
    assert provider_error_reason(uncoded) == "SandboxApiException"
