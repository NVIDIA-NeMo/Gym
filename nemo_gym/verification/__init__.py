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
"""Helpers that resources servers share for verifying submissions.

sandbox_verification grades a submission in a fresh, single-use sandbox. It uses only the
provider-neutral API; core sandbox APIs and providers live in :mod:`nemo_gym.sandbox`.
"""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.verification.judge import (
        JudgeError,
        call_judge,
        judge_failsafe,
        normalize_math_judge_verdict,
        reraise_judge_errors,
    )
    from nemo_gym.verification.sandbox_verification import (
        RUNNER_FILE,
        SandboxVerificationConfig,
        SandboxVerificationError,
        run_verification_sandbox,
    )
    from nemo_gym.verification.verifier_fixture import VerifierFixture, exercise_verifier_fixture


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "JudgeError": "judge",
    "RUNNER_FILE": "sandbox_verification",
    "SandboxVerificationConfig": "sandbox_verification",
    "SandboxVerificationError": "sandbox_verification",
    "VerifierFixture": "verifier_fixture",
    "call_judge": "judge",
    "exercise_verifier_fixture": "verifier_fixture",
    "judge_failsafe": "judge",
    "normalize_math_judge_verdict": "judge",
    "reraise_judge_errors": "judge",
    "run_verification_sandbox": "sandbox_verification",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
