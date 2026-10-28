# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run harness processes inside task sandboxes under a shared supervisor.

The harness adapter installs its runtime, stages activation input, and parses
harness output. SandboxSession uploads the supervisor, runs and stops the harness
process, collects artifacts, and releases the sandbox. The supervisor client
implements the shell protocol over session control files. All three run on the
agent server.

Only process_supervisor.sh and harness_runner.sh are copied from this package into
the task sandbox; they need bash, not Python. The supervisor enforces deadlines, stops
the harness process group and writes its termination receipt. A forced stop also kills the
harness's descendants in other groups and processes that inherited its marker; processes
that had left the harness before the stop, and detached processes after a natural exit,
remain for the sandbox owner to stop. Core sandbox APIs and providers live in
:mod:`nemo_gym.sandbox`.
"""
