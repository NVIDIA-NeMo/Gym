# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provide the core training-token capture interfaces.

Training capture is separate from evaluation capture.
Middleware sets a request-scoped token sink.
The model server records a ``TokenEntry`` from its complete response.
Consumers call ``TokenSource.freeze`` for an atomic snapshot.
The snapshot includes entries and incomplete state.
Its ``snapshot_id`` identifies the exact frozen state.
``TokenCaptureStore`` is Gym's local sink and source implementation.
Framework transports may provide their own sink and source.
``finalize_rollout_token_capture`` rebuilds one rollout record from its frozen snapshot.
The incomplete state prevents training on a rollout that lost a model call.
Finalization freezes and rebuilds the rollout.
Finalization does not retire the snapshot.
The caller retires it only after durable handoff.
Retirement uses the frozen ``snapshot_id`` and version.
Failed or masked builds retain their capture evidence.

This package is token capture's public API.
Code outside ``nemo_gym/`` imports token capture names from ``nemo_gym.models.token_capture``,
not from ``nemo_gym.models``.
Every name it serves imports without Gym's server stack or an OpenAI client,
because frameworks implement its protocols.
Framework token staging has a public API of its own, ``nemo_gym.models.token_capture.staging``.
"""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.models.token_capture.adapters.vllm import VLLMCaptureAdapter
    from nemo_gym.models.token_capture.builder import (
        Chain,
        assert_prefix_contiguity,
        prefix_merging,
        project_chain_to_output_items,
        project_main_chain_response,
        run_builder,
    )
    from nemo_gym.models.token_capture.config import TokenCaptureConfig, token_capture_config
    from nemo_gym.models.token_capture.conformance import ConformanceError, ConformanceWarning, run_conformance
    from nemo_gym.models.token_capture.consumer import (
        clear_token_captures_for_rollouts,
        mask_incomplete_when_attributed_from_config,
        token_capture_dirs_from_config,
        trajectories_for_rollout,
        trajectories_from_source,
    )
    from nemo_gym.models.token_capture.delivery import finalize_rollout_token_capture, retire_rollout_token_capture
    from nemo_gym.models.token_capture.fingerprint import assistant_fingerprint, conversation_digest
    from nemo_gym.models.token_capture.lineage import (
        FileLineageStore,
        IncrementalLineageStore,
        InMemoryLineageStore,
        LineageIndex,
        RolloutLineage,
        stamp_continuation,
    )
    from nemo_gym.models.token_capture.protocols import (
        CAPTURE_REFUSAL_HEADER,
        CaptureAdmissionError,
        CaptureInputError,
        CaptureLedger,
        LineageMatch,
        LineageResolution,
        LineageResolver,
        RolloutRemovalPayload,
        RolloutRetiredError,
        TokenCaptureFrozenError,
        TokenCaptureRetiredError,
        TokenCaptureSnapshot,
        TokenSink,
        TokenSource,
        install_lineage_store,
        install_token_sink,
        install_token_source,
        installed_lineage_store,
        installed_token_sink,
        installed_token_source,
    )
    from nemo_gym.models.token_capture.records import (
        CONTEXT_LENGTH_EXCEEDED_ERROR_CODE,
        ENGINE_REFUSED_CALL_REASON,
        TOKEN_ENTRY_MIN_SCHEMA_VERSION,
        TOKEN_ENTRY_RECORD_SCHEMA_VERSION,
        TOKEN_FIELDS,
        UNCOMMITTED_CALL_REASON,
        UNRESOLVED_PARENT_REASON,
        UNSERVED_CALL_FAILURE_REASONS,
        EngineRefusalRecord,
        ParentResolutionStatus,
        TokenEntry,
        apply_capture_incomplete_verdict,
        compute_digest,
        cumulative_tokens,
        extract_token_fields,
        incomplete_reason_for_finish_reason,
        stamp_lineage,
    )
    from nemo_gym.models.token_capture.sink import (
        NG_CAPTURE_FIELD,
        NG_COMMIT_COORDS_FIELD,
        CaptureContext,
        capture_engine_refusal,
        capture_health_snapshot,
        capture_tokens,
        commit_entry,
        current_capture_context,
        mark_external_staging_committed,
        register_call_intent,
        reset_token_sink,
        resolve_parent,
        set_token_sink,
    )
    from nemo_gym.models.token_capture.staging.records import CallRecord, CaptureLedgerCommit
    from nemo_gym.models.token_capture.store import TokenCaptureStore, make_token_store, validate_rollout_id
    from nemo_gym.models.token_capture.terminal import TerminalAttribution, resolve_terminal


# What code outside the library, such as servers, imports from this package.
# Each name's submodule loads on first use.
# Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "CAPTURE_REFUSAL_HEADER": "protocols",
    "CONTEXT_LENGTH_EXCEEDED_ERROR_CODE": "records",
    "CallRecord": "staging.records",
    "CaptureAdmissionError": "protocols",
    "CaptureContext": "sink",
    "CaptureInputError": "protocols",
    "CaptureLedger": "protocols",
    "CaptureLedgerCommit": "staging.records",
    "Chain": "builder",
    "ConformanceError": "conformance",
    "ConformanceWarning": "conformance",
    "ENGINE_REFUSED_CALL_REASON": "records",
    "EngineRefusalRecord": "records",
    "FileLineageStore": "lineage",
    "InMemoryLineageStore": "lineage",
    "IncrementalLineageStore": "lineage",
    "LineageIndex": "lineage",
    "LineageMatch": "protocols",
    "LineageResolution": "protocols",
    "LineageResolver": "protocols",
    "NG_CAPTURE_FIELD": "sink",
    "NG_COMMIT_COORDS_FIELD": "sink",
    "ParentResolutionStatus": "records",
    "RolloutLineage": "lineage",
    "RolloutRemovalPayload": "protocols",
    "RolloutRetiredError": "protocols",
    "TOKEN_ENTRY_MIN_SCHEMA_VERSION": "records",
    "TOKEN_ENTRY_RECORD_SCHEMA_VERSION": "records",
    "TOKEN_FIELDS": "records",
    "TerminalAttribution": "terminal",
    "TokenCaptureConfig": "config",
    "TokenCaptureFrozenError": "protocols",
    "TokenCaptureRetiredError": "protocols",
    "TokenCaptureSnapshot": "protocols",
    "TokenCaptureStore": "store",
    "TokenEntry": "records",
    "TokenSink": "protocols",
    "TokenSource": "protocols",
    "UNCOMMITTED_CALL_REASON": "records",
    "UNRESOLVED_PARENT_REASON": "records",
    "UNSERVED_CALL_FAILURE_REASONS": "records",
    "VLLMCaptureAdapter": "adapters.vllm",
    "apply_capture_incomplete_verdict": "records",
    "assert_prefix_contiguity": "builder",
    "assistant_fingerprint": "fingerprint",
    "capture_engine_refusal": "sink",
    "capture_health_snapshot": "sink",
    "capture_tokens": "sink",
    "clear_token_captures_for_rollouts": "consumer",
    "commit_entry": "sink",
    "compute_digest": "records",
    "conversation_digest": "fingerprint",
    "cumulative_tokens": "records",
    "current_capture_context": "sink",
    "extract_token_fields": "records",
    "finalize_rollout_token_capture": "delivery",
    "incomplete_reason_for_finish_reason": "records",
    "install_lineage_store": "protocols",
    "install_token_sink": "protocols",
    "install_token_source": "protocols",
    "installed_lineage_store": "protocols",
    "installed_token_sink": "protocols",
    "installed_token_source": "protocols",
    "make_token_store": "store",
    "mark_external_staging_committed": "sink",
    "mask_incomplete_when_attributed_from_config": "consumer",
    "prefix_merging": "builder",
    "project_chain_to_output_items": "builder",
    "project_main_chain_response": "builder",
    "register_call_intent": "sink",
    "reset_token_sink": "sink",
    "resolve_parent": "sink",
    "resolve_terminal": "terminal",
    "retire_rollout_token_capture": "delivery",
    "run_builder": "builder",
    "run_conformance": "conformance",
    "set_token_sink": "sink",
    "stamp_continuation": "lineage",
    "stamp_lineage": "records",
    "token_capture_config": "config",
    "token_capture_dirs_from_config": "consumer",
    "trajectories_for_rollout": "consumer",
    "trajectories_from_source": "consumer",
    "validate_rollout_id": "store",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
