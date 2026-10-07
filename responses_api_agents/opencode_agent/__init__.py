# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Persisted reply identity emitted by the correlation-enabled OpenCode binary.
_assistant_message_header: bytes | None = b"x-opencode-assistant-message-id"

# Exact session identity emitted by official and correlation-enabled OpenCode builds.
_session_id_headers: tuple[bytes, ...] = (b"x-session-id", b"x-session-affinity")
