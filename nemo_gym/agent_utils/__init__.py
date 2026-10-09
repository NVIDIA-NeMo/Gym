# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run harness processes inside task sandboxes under a shared supervisor.

The harness adapter installs its runtime, stages activation input, and parses
harness output. SandboxSession uploads the supervisor, runs and stops the harness
process, collects artifacts, and releases the sandbox. The supervisor client
implements the shell protocol over session control files. All three run on the
agent server.

Only process_supervisor.py is copied from this package into the task sandbox. It
uses the standard library to enforce deadlines, kill and reap descendants, and
write the cleanup receipt. Core sandbox APIs and providers live in
:mod:`nemo_gym.sandbox`.
"""
