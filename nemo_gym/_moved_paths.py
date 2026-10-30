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
"""Old import paths for code moved by the `nemo_gym` package reorganization.

Every move is recorded here, with the release it first ships in.
The old path keeps working until `removed_in`, which is `since` plus `DEPRECATION_WINDOW_MINOR_RELEASES` minor releases.
Each use of an old path emits a `DeprecationWarning` that names the new path, both releases, and the running version.
A unit test fails once the running version reaches `removed_in` while the entry still exists, so the removal cannot be forgotten.

There are two kinds of move:

- A whole module or package moves (`MOVED_MODULES`).
  `MovedModuleFinder` serves the old name by importing the new module and registering the same module object under the old name.
  The module never runs twice, so its classes, registries, and context variables stay single.
  `mock.patch("<old path>.<name>")` still patches the code that runs.
  Submodules of a moved package resolve the same way.
- Names split out of a module that stays (`MOVED_NAMES`).
  The old module installs `moved_names_getattr` as its `__getattr__`, which returns the identical object from its new module.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import warnings
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Mapping, Optional


DEPRECATION_WINDOW_MINOR_RELEASES = 2


@dataclass(frozen=True)
class MovedPath:
    """Where a module or name moved to, and the release the move first ships in.

    Attributes:
        new: The new dotted path: a module for a module move; `module` or `module:new_name` for a name move.
        since: The first release (`MAJOR.MINOR.PATCH`) that ships the new path.
    """

    new: str
    since: str

    @property
    def removed_in(self) -> str:
        """The release that drops the old path."""
        major, minor, _ = (int(part) for part in self.since.split("."))
        return f"{major}.{minor + DEPRECATION_WINDOW_MINOR_RELEASES}.0"


# Old module path -> new module path.
# A key also covers every submodule under it.
MOVED_MODULES: dict[str, MovedPath] = {
    "nemo_gym.h2_ping_sidecar": MovedPath(new="nemo_gym.orchestration.h2_ping_sidecar", since="0.8.0"),
    "nemo_gym.cli.setup_command": MovedPath(new="nemo_gym.orchestration.server_venv", since="0.8.0"),
    "nemo_gym.testing": MovedPath(new="nemo_gym.server_runtime.testing", since="0.8.0"),
    "nemo_gym.testing.agent_session_conformance": MovedPath(
        new="nemo_gym.agents.testing.agent_session_conformance", since="0.8.0"
    ),
    "nemo_gym.observability": MovedPath(new="nemo_gym.rollouts", since="0.8.0"),
    "nemo_gym.rollout_collection": MovedPath(new="nemo_gym.rollouts.rollout_collection", since="0.8.0"),
    "nemo_gym.batch_status": MovedPath(new="nemo_gym.rollouts.batch_status", since="0.8.0"),
    "nemo_gym.rollout_health": MovedPath(new="nemo_gym.rollouts.rollout_health", since="0.8.0"),
    "nemo_gym.health": MovedPath(new="nemo_gym.rollouts.health", since="0.8.0"),
    "nemo_gym.rollout_outcomes": MovedPath(new="nemo_gym.rollouts.rollout_outcomes", since="0.8.0"),
    "nemo_gym.omitted_rollouts": MovedPath(new="nemo_gym.rollouts.omitted_rollouts", since="0.8.0"),
    "nemo_gym.harness_capabilities": MovedPath(new="nemo_gym.rollouts.harness_capabilities", since="0.8.0"),
    "nemo_gym.tasks.harbor.cli": MovedPath(new="nemo_gym.data.tasks.harbor.prepare", since="0.8.0"),
    "nemo_gym.prompt": MovedPath(new="nemo_gym.data.prompt", since="0.8.0"),
    "nemo_gym.vision_input": MovedPath(new="nemo_gym.data.vision_input", since="0.8.0"),
    "nemo_gym.task_data": MovedPath(new="nemo_gym.data.task_data", since="0.8.0"),
    "nemo_gym.task_materialization": MovedPath(new="nemo_gym.data.task_materialization", since="0.8.0"),
    "nemo_gym.tasks": MovedPath(new="nemo_gym.data.tasks", since="0.8.0"),
    "nemo_gym.dataset_metrics": MovedPath(new="nemo_gym.data.dataset_metrics", since="0.8.0"),
    "nemo_gym.dataset_orchestrator": MovedPath(new="nemo_gym.data.dataset_orchestrator", since="0.8.0"),
    "nemo_gym.train_data_utils": MovedPath(new="nemo_gym.data.dataset_preparation", since="0.8.0"),
    "nemo_gym.hf_utils": MovedPath(new="nemo_gym.data.hf_utils", since="0.8.0"),
    "nemo_gym.gitlab_utils": MovedPath(new="nemo_gym.data.gitlab_utils", since="0.8.0"),
    "nemo_gym.base_responses_api_agent": MovedPath(new="nemo_gym.agents.base_responses_api_agent", since="0.8.0"),
    "nemo_gym.agent_utils": MovedPath(new="nemo_gym.agents.utils", since="0.8.0"),
    "nemo_gym.agent_outcome": MovedPath(new="nemo_gym.agents.agent_outcome", since="0.8.0"),
    "nemo_gym.agent_activation_types": MovedPath(new="nemo_gym.agents.agent_activation_types", since="0.8.0"),
    "nemo_gym.sandbox.agent_tools": MovedPath(new="nemo_gym.agents.agent_tools", since="0.8.0"),
    "nemo_gym.skills": MovedPath(new="nemo_gym.agents.skills", since="0.8.0"),
    "nemo_gym.base_responses_api_model": MovedPath(new="nemo_gym.models.base_responses_api_model", since="0.8.0"),
    "nemo_gym.openai_utils": MovedPath(new="nemo_gym.models.openai_utils", since="0.8.0"),
    "nemo_gym.anthropic_utils": MovedPath(new="nemo_gym.models.anthropic_utils", since="0.8.0"),
    "nemo_gym.anthropic_converter": MovedPath(new="nemo_gym.models.anthropic_converter", since="0.8.0"),
    "nemo_gym.responses_converter": MovedPath(new="nemo_gym.models.chat_completions_converter", since="0.8.0"),
    "nemo_gym.chat_streaming": MovedPath(new="nemo_gym.models.chat_streaming", since="0.8.0"),
    "nemo_gym.responses_streaming": MovedPath(new="nemo_gym.models.responses_streaming", since="0.8.0"),
    "nemo_gym.token_id_capture": MovedPath(new="nemo_gym.models.token_capture", since="0.8.0"),
    "nemo_gym.server_utils": MovedPath(new="nemo_gym.server_runtime.base_server", since="0.8.0"),
    "nemo_gym._checkpoint.model": MovedPath(new="nemo_gym.models._checkpoint", since="0.8.0"),
    "nemo_gym._checkpoint.model_workers": MovedPath(new="nemo_gym.models._checkpoint_workers", since="0.8.0"),
    "nemo_gym._checkpoint": MovedPath(new="nemo_gym.server_runtime._checkpoint", since="0.8.0"),
    "nemo_gym.base_resources_server": MovedPath(new="nemo_gym.server_runtime.base_resources_server", since="0.8.0"),
    "nemo_gym.base_environment_server": MovedPath(
        new="nemo_gym.server_runtime.base_environment_server", since="0.8.0"
    ),
    "nemo_gym.session_routing": MovedPath(
        new="nemo_gym.server_runtime.request_handling.session_routing", since="0.8.0"
    ),
    "nemo_gym.rollout_correlation": MovedPath(
        new="nemo_gym.server_runtime.request_handling.rollout_context", since="0.8.0"
    ),
    "nemo_gym.mcp_auto_exposure": MovedPath(new="nemo_gym.server_runtime.mcp_auto_exposure", since="0.8.0"),
    "nemo_gym.server_status": MovedPath(new="nemo_gym.server_runtime.server_status", since="0.8.0"),
    "nemo_gym.episode_types": MovedPath(new="nemo_gym.server_runtime.episode_types", since="0.8.0"),
    "nemo_gym.single_agent_turn_types": MovedPath(
        new="nemo_gym.server_runtime.single_agent_turn_types", since="0.8.0"
    ),
    "nemo_gym.tool_access": MovedPath(new="nemo_gym.server_runtime.tool_access", since="0.8.0"),
    "nemo_gym.runtime_dir": MovedPath(new="nemo_gym.server_runtime.runtime_dir", since="0.8.0"),
    "nemo_gym.observability.verifiers": MovedPath(new="nemo_gym.server_runtime.verifier_observability", since="0.8.0"),
    "nemo_gym.resolved_config": MovedPath(new="nemo_gym.artifacts.resolved_config", since="0.8.0"),
    "nemo_gym.run_artifacts": MovedPath(new="nemo_gym.artifacts.run_artifacts", since="0.8.0"),
    "nemo_gym.run_exporter_registry": MovedPath(new="nemo_gym.artifacts.exporter_registry", since="0.8.0"),
    "nemo_gym.exporters": MovedPath(new="nemo_gym.artifacts.exporters", since="0.8.0"),
    "nemo_gym.reward_profile": MovedPath(new="nemo_gym.metrics.reward_profile", since="0.8.0"),
    "nemo_gym.metrics_config": MovedPath(new="nemo_gym.metrics.metrics_config", since="0.8.0"),
    "nemo_gym.judge": MovedPath(new="nemo_gym.verification.judge", since="0.8.0"),
    "nemo_gym.verifier_fixture": MovedPath(new="nemo_gym.verification.verifier_fixture", since="0.8.0"),
    "nemo_gym.verifier_utils": MovedPath(new="nemo_gym.verification", since="0.8.0"),
    "nemo_gym.rollout_observability": MovedPath(new="nemo_gym.trajectory.trajectory_record", since="0.8.0"),
    "nemo_gym.atif_export": MovedPath(new="nemo_gym.trajectory.atif_export", since="0.8.0"),
    "nemo_gym.atif_reverification": MovedPath(new="nemo_gym.trajectory.atif_reverification", since="0.8.0"),
    "nemo_gym.atif_json": MovedPath(new="nemo_gym.trajectory.atif_json", since="0.8.0"),
    "nemo_gym.atif_v1_7": MovedPath(new="nemo_gym.trajectory.atif_v1_7", since="0.8.0"),
    "nemo_gym.profiling": MovedPath(new="nemo_gym.telemetry.profiling", since="0.8.0"),
    "nemo_gym.failure_kinds": MovedPath(new="nemo_gym.core.failure_kinds", since="0.8.0"),
    "nemo_gym.config_types": MovedPath(new="nemo_gym.config.config_types", since="0.8.0"),
    "nemo_gym.global_config": MovedPath(new="nemo_gym.config.global_config", since="0.8.0"),
    "nemo_gym.secret_utils": MovedPath(new="nemo_gym.config.secret_utils", since="0.8.0"),
    "nemo_gym.server_metadata": MovedPath(new="nemo_gym.config.server_metadata", since="0.8.0"),
    "nemo_gym.decorators": MovedPath(new="nemo_gym.utils.decorators", since="0.8.0"),
    "nemo_gym.path_utils": MovedPath(new="nemo_gym.utils.path_utils", since="0.8.0"),
    "nemo_gym.install_lock": MovedPath(new="nemo_gym.utils.install_lock", since="0.8.0"),
    "nemo_gym.cli._compat": MovedPath(new="nemo_gym.utils.compat", since="0.8.0"),
    "nemo_gym.cli._venv_setup": MovedPath(new="nemo_gym.utils.venv_setup", since="0.8.0"),
}


def _names(new: str, since: str, names: str) -> dict[str, MovedPath]:
    """Entries for `MOVED_NAMES`: each whitespace-separated name in `names` moved to the module `new`."""
    return {name: MovedPath(new=new, since=since) for name in names.split()}


# Old module path -> {old name -> new location}, for names split out of a module that still exists.
MOVED_NAMES: dict[str, dict[str, MovedPath]] = {
    "nemo_gym.server_utils": {
        # Moved, or imported before the split, and no longer used by server_utils:
        # served by its __getattr__ with a warning.
        **_names(
            "nemo_gym.client.http_client",
            "0.8.0",
            """
            _detach_global_aiohttp_clients _format_upstream_error_body _GLOBAL_AIOHTTP_CLIENT
            _GLOBAL_AIOHTTP_CLIENT_CONFIG _GLOBAL_AIOHTTP_CLIENT_LOOP _GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY
            _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG _GLOBAL_AIOHTTP_CONTROL_CLIENT _make_keepalive_socket_factory
            _NUM_CLIENT_OS_ERROR _NUM_SERVER_DISCONNECTED_ERROR _PickleSafeRequestInfo _redacted_url
            _request_with_retries _RequestOptions _traced_request _UPSTREAM_ERROR_LOG_BODY_CHARS CIMultiDict
            ClientConnectorError ClientOSError ClientResponse ClientSession ClientTimeout
            close_global_aiohttp_client DISCONNECTED_CLIENT_OS_HELP_TEXT DISCONNECTED_CLIENT_OS_PRINT_INTERVAL
            DummyCookieJar Field format_exception_with_response_body get_global_aiohttp_client
            get_global_aiohttp_control_client get_response_json global_aiohttp_client_exit
            is_global_aiohttp_client_request_debug_enabled is_global_aiohttp_client_setup MAX_NUM_TRIES
            raise_for_status request ServerDisconnectedError set_global_aiohttp_client
            """,
        ),
        **_names(
            "nemo_gym.client.server_client",
            "0.8.0",
            """
            _connectable_client_host _TRACEPARENT_PATTERN apply_rollout_prefix ConfigDict ConnectionError
            get_server_url NEMO_GYM_MODEL_SERVER_BASE_URL_ENV_VAR_NAME NEMO_GYM_MODEL_SERVER_NAME_ENV_VAR_NAME
            rollout_path_prefix ServerStatus setup_server_client trace_context_path_segment
            """,
        ),
        **_names(
            "nemo_gym.config.config_types",
            "0.8.0",
            """
            BaseServerConfig HeadServerUnreachableError ROLLOUT_PATH_PREFIX TOKEN_CAPTURE_PATH_SEGMENT
            TRACE_CONTEXT_PATH_SEGMENT
            """,
        ),
        **_names(
            "nemo_gym.config.global_config",
            "0.8.0",
            """
            GlobalConfigDictParser GlobalConfigDictParserConfig HEAD_SERVER_KEY_NAME maybe_get_global_config_dict
            NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME OBSERVABILITY_ENABLED_KEY_NAME RAY_HEAD_NODE_ADDRESS_KEY_NAME
            """,
        ),
        **_names(
            "nemo_gym.core.episode",
            "0.8.0",
            """
            current_rollout_id current_verifier_invocation_id maybe_rollout_id_from_run_body rollout_context
            VERIFIER_INVOCATION_HEADER
            """,
        ),
        **_names("nemo_gym.server_runtime._checkpoint.settings", "0.8.0", "checkpoint_settings"),
        **_names(
            "nemo_gym.server_runtime.entrypoint_flags",
            "0.8.0",
            """
            _declared_class_flag _parse_entrypoint _run_webserver_class_names entrypoint_class_flag
            entrypoint_server_class_names
            """,
        ),
        **_names(
            "nemo_gym.server_runtime.ray_support",
            "0.8.0",
            """
            _declared_ray_enabled _get_ray _NEMO_GYM_STARTED_RAY_CLUSTER DictConfig entrypoint_ray_enabled
            initialize_ray open_dict ray_is_installed
            """,
        ),
        **_names(
            "nemo_gym.server_runtime.registry_server",
            "0.8.0",
            """
            DEFAULT_HEAD_SERVER_PORT HeadServer PrivateAttr ServerInstanceDisplayConfig
            """,
        ),
        **_names(
            "nemo_gym.server_runtime.request_handling.disconnect",
            "0.8.0",
            """
            _cancelled_request_route create_task_group Match Message
            """,
        ),
        **_names(
            "nemo_gym.server_runtime.request_handling.errors",
            "0.8.0",
            """
            _bounded_validation_error_value _escaped_log_prefix _escaped_log_text _log_validation_exception
            _VALIDATION_ERROR_LOG_BODY_CHARS _VALIDATION_ERROR_LOG_ERRORS_CHARS _VALIDATION_ERROR_LOG_FIELD_CHARS
            _VALIDATION_ERROR_LOG_LOC_ITEMS _VALIDATION_ERROR_LOG_MAX_ERRORS _validation_error_summaries
            _validation_errors_log_text request_validation_exception_handler
            """,
        ),
        **_names(
            "nemo_gym.server_runtime.request_handling.request_context",
            "0.8.0",
            """
            ASGIApp Receive RequestContextIsolationMiddleware Scope Send
            """,
        ),
        **_names(
            "nemo_gym.server_runtime.request_handling.rollout_context",
            "0.8.0",
            """
            ROUTED_PATH_STATE_KEY scope_with_routed_path
            """,
        ),
        **_names(
            "nemo_gym.server_runtime.request_handling.trace_context",
            "0.8.0",
            """
            _restore_trace_segment_on_redirect _ROLLOUT_PREFIX_PATTERN _ROLLOUT_PREFIX_RE _TRACE_CONTEXT_PATH_RE
            TraceContextPathMiddleware
            """,
        ),
        **_names("nemo_gym.telemetry._fallbacks", "0.8.0", "safe_set_span_attributes"),
        **_names(
            "nemo_gym.telemetry.connection_pool",
            "0.8.0",
            """
            build_connection_pool_connector QueueTimedTCPConnector reset_server_name set_server_name
            """,
        ),
        **_names("nemo_gym.telemetry.observer", "0.8.0", "request_counts"),
        **_names("nemo_gym.telemetry.profiling", "0.8.0", "ProfilingMiddlewareInputConfig"),
        **_names(
            "nemo_gym.utils.server_process",
            "0.8.0",
            """
            _has_injected_global_config_env get_nemo_gym_fastapi_num_workers is_nemo_gym_fastapi_entrypoint
            IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME
            NEMO_GYM_FASTAPI_NUM_WORKERS
            """,
        ),
        # Moved, and still imported by server_utils for its own use, so these old paths do not warn yet.
        **_names(
            "nemo_gym.client.http_client",
            "0.8.0",
            """
            _format_upstream_error_log _set_tcp_keepalive GlobalAIOHTTPAsyncClientConfig
            """,
        ),
        **_names("nemo_gym.client.server_client", "0.8.0", "ServerClient SESSION_ID_KEY"),
        **_names("nemo_gym.server_runtime.ray_support", "0.8.0", "_connect_server_to_ray maybe_ray_cluster_exit"),
        **_names(
            "nemo_gym.server_runtime.request_handling.disconnect",
            "0.8.0",
            """
            ClientDisconnectCancellationMiddleware
            """,
        ),
        **_names("nemo_gym.server_runtime.request_handling.errors", "0.8.0", "_validation_exception_handler"),
        **_names(
            "nemo_gym.server_runtime.request_handling.request_context",
            "0.8.0",
            """
            install_request_context_isolation
            """,
        ),
        **_names("nemo_gym.server_runtime.request_handling.trace_context", "0.8.0", "install_trace_context_path"),
        **_names("nemo_gym.telemetry.profiling", "0.8.0", "ProfilingMiddlewareConfig"),
        **_names(
            "nemo_gym.utils.server_process",
            "0.8.0",
            """
            is_nemo_gym_fastapi_worker set_is_nemo_gym_fastapi_entrypoint set_is_nemo_gym_fastapi_worker
            set_nemo_gym_fastapi_num_workers
            """,
        ),
    },
    "nemo_gym.cli.utils": {
        # Moved, and still imported by cli/utils for its own use, so these old paths do not warn yet.
        **_names("nemo_gym.utils.text", "0.8.0", "did_you_mean"),
    },
    "nemo_gym.openai_utils": {
        **_names(
            "nemo_gym.core.responses_api_types",
            "0.8.0",
            """
            CHAT_REQUEST_IGNORED_CLIENT_FIELDS CHAT_REQUEST_PROVIDER_EXTENSION_FIELDS
            ChatCompletionModeration NeMoGymAdditionalTools NeMoGymChatCompletion
            NeMoGymChatCompletionAllowedToolChoiceParam NeMoGymChatCompletionAllowedToolsParam
            NeMoGymChatCompletionAssistantMessageForTrainingParam
            NeMoGymChatCompletionAssistantMessageParam NeMoGymChatCompletionContentPartFileParam
            NeMoGymChatCompletionContentPartImageParam NeMoGymChatCompletionContentPartInputAudioParam
            NeMoGymChatCompletionContentPartParam NeMoGymChatCompletionContentPartTextParam
            NeMoGymChatCompletionContentPartVideoUrlParam NeMoGymChatCompletionCreateParamsNonStreaming
            NeMoGymChatCompletionCustomToolParam NeMoGymChatCompletionDeveloperMessageParam
            NeMoGymChatCompletionMessage NeMoGymChatCompletionMessageCustomToolCall
            NeMoGymChatCompletionMessageCustomToolCallParam NeMoGymChatCompletionMessageForTraining
            NeMoGymChatCompletionMessageParam NeMoGymChatCompletionMessageToolCall
            NeMoGymChatCompletionMessageToolCallFunctionParam NeMoGymChatCompletionMessageToolCallParam
            NeMoGymChatCompletionMessageToolCallUnion NeMoGymChatCompletionMessageToolCallUnionParam
            NeMoGymChatCompletionMessageToolCallsParam NeMoGymChatCompletionOutputMessage
            NeMoGymChatCompletionPredictionContentParam NeMoGymChatCompletionSystemMessageParam
            NeMoGymChatCompletionToolChoiceOptionParam NeMoGymChatCompletionToolMessageParam
            NeMoGymChatCompletionToolParam NeMoGymChatCompletionToolUnionParam
            NeMoGymChatCompletionUserMessageParam NeMoGymChoice
            NeMoGymCodeInterpreterContainerAutoParam NeMoGymCodeInterpreterToolParam
            NeMoGymCompactionTrigger NeMoGymCompoundFilter NeMoGymComputerCallOutput
            NeMoGymContainerAutoParam NeMoGymContainerNetworkPolicyAllowlistParam
            NeMoGymContainerNetworkPolicyParam NeMoGymContent NeMoGymEasyInputMessage
            NeMoGymEasyInputMessageForTraining NeMoGymFileSearchToolParam NeMoGymFunction
            NeMoGymFunctionCallOutput NeMoGymFunctionDefinition NeMoGymFunctionShellToolParam
            NeMoGymFunctionToolParam NeMoGymImageGenerationCall NeMoGymInputAudio NeMoGymInputVideoPart
            NeMoGymLocalEnvironmentParam NeMoGymLocalShellCall NeMoGymLocalShellCallOutput
            NeMoGymLogprob NeMoGymLogprobTopLogprob NeMoGymMcpApprovalResponse NeMoGymMessage
            NeMoGymMessageForTraining NeMoGymNamespaceToolParam NeMoGymResponse
            NeMoGymResponseAdditionalTools NeMoGymResponseApplyPatchToolCall
            NeMoGymResponseApplyPatchToolCallOutput NeMoGymResponseCodeInterpreterToolCall
            NeMoGymResponseCompactionItem NeMoGymResponseComputerCallOutput
            NeMoGymResponseComputerToolCall NeMoGymResponseCreateParamsNonStreaming
            NeMoGymResponseCustomToolCall NeMoGymResponseCustomToolCallForTraining
            NeMoGymResponseCustomToolCallOutput NeMoGymResponseCustomToolCallOutputItem
            NeMoGymResponseFileSearchToolCall NeMoGymResponseFunctionCallOutput
            NeMoGymResponseFunctionShellToolCall NeMoGymResponseFunctionShellToolCallOutput
            NeMoGymResponseFunctionToolCall NeMoGymResponseFunctionToolCallForTraining
            NeMoGymResponseFunctionWebSearch NeMoGymResponseInput NeMoGymResponseInputContentList
            NeMoGymResponseInputContentPart NeMoGymResponseInputItem NeMoGymResponseInputText
            NeMoGymResponseInputTokensDetails NeMoGymResponseLocalShellCallOutput
            NeMoGymResponseMcpApprovalRequest NeMoGymResponseMcpApprovalResponse NeMoGymResponseMcpCall
            NeMoGymResponseMcpListTools NeMoGymResponseOutputItem NeMoGymResponseOutputMessage
            NeMoGymResponseOutputMessageForTraining NeMoGymResponseOutputRefusal
            NeMoGymResponseOutputText NeMoGymResponseOutputTokensDetails NeMoGymResponseReasoningItem
            NeMoGymResponseReasoningItemForTraining NeMoGymResponseToolSearchCall
            NeMoGymResponseToolSearchOutputItem NeMoGymResponseUsage NeMoGymSummary NeMoGymToolChoice
            NeMoGymToolChoiceAllowedParam NeMoGymToolParam REQUIRED_TOKEN_METADATA_FIELDS
            RESPONSES_TO_TRAIN RoutedExperts TOKEN_METADATA_FIELDS TokenIDLogProbMixin
            TokenIDLogProbTypedDictMixin _ITEM_STATUSES _MESSAGE_ROLES _OUTPUT_CONTENT_PART_TYPES
            _RESPONSE_INPUT_ITEM_TAG_BY_CLASS _RESPONSE_OUTPUT_ITEM_TAG_BY_CLASS _SIMPLE_MESSAGE_ROLES
            _TRAINABLE_ITEM_TAGS _TRAINING_TAG_SUFFIX _add_optional_token_counts
            _discriminate_message_item _discriminate_response_input_item
            _discriminate_response_output_item _discriminate_untyped_input_item
            _first_content_part_type _normalize_output_item_for_replay _normalize_tool_for_replay
            _register_item_tags _require_response_output_item_type _strip_outer_tool_call_names
            _tag_for_item_instance _training_variant_tag _validate_atomic_token_metadata
            _validate_input_video_part accumulate_response_usage training_variant_of
            """,
        ),
        # openai_utils imported these and they stay importable through it.
        **_names(
            "nemo_gym.core.responses_api_types",
            "0.8.0",
            """
            ActionFind ActionOpenPage ActionSearch Annotation ApplyPatchToolParam BeforeValidator
            ChatCompletion ChatCompletionAllowedToolChoiceParam ChatCompletionAllowedToolsParam
            ChatCompletionAssistantMessageParam ChatCompletionAudioParam
            ChatCompletionContentPartFileParam ChatCompletionContentPartImageParam
            ChatCompletionContentPartTextParam ChatCompletionCustomToolParam
            ChatCompletionDeveloperMessageParam ChatCompletionMessage
            ChatCompletionMessageCustomToolCall ChatCompletionMessageCustomToolCallParam
            ChatCompletionMessageToolCall ChatCompletionMessageToolCallParam
            ChatCompletionNamedToolChoiceCustomParam ChatCompletionNamedToolChoiceParam
            ChatCompletionPredictionContentParam ChatCompletionStreamOptionsParam
            ChatCompletionSystemMessageParam ChatCompletionToolMessageParam ChatCompletionToolParam
            ChatCompletionUserMessageParam ChatModel Choice CodeInterpreter
            CodeInterpreterContainerCodeInterpreterToolAuto CompactionTrigger ComparisonFilter
            CompoundFilter ComputerCallOutput ComputerToolParam ComputerUsePreviewToolParam ConfigDict
            ContainerAutoParam ContainerNetworkPolicyAllowlistParam ContainerNetworkPolicyDisabledParam
            ContainerNetworkPolicyDomainSecretParam ContainerReferenceParam ContentArrayOfContentPart
            ContextManagement CustomToolParam Discriminator FileSearchToolParam FunctionDefinition
            FunctionShellToolParam FunctionToolParam ImageGeneration ImageGenerationCall
            InlineSkillParam InputAdditionalTools JsonValue LocalEnvironmentParam LocalShell
            LocalShellCall LocalShellCallOutput LocalSkillParam Mcp McpApprovalRequest
            McpApprovalResponse McpCall McpListTools Metadata Moderation NamespaceToolParam
            NamespaceToolParamTool OutputAdditionalTools OutputLocalShellCallOutput
            OutputMcpApprovalResponse Reasoning ReasoningContent ReasoningEffort Response
            ResponseApplyPatchToolCall ResponseApplyPatchToolCallOutput ResponseCodeInterpreterToolCall
            ResponseCompactionItem ResponseComputerToolCall ResponseComputerToolCallOutputItem
            ResponseConversationParamParam ResponseCustomToolCall ResponseCustomToolCallOutput
            ResponseCustomToolCallOutputItem ResponseFileSearchToolCall ResponseFormat
            ResponseFunctionCallOutputItemListParam ResponseFunctionShellToolCall
            ResponseFunctionShellToolCallOutput ResponseFunctionToolCallOutputItem
            ResponseFunctionWebSearch ResponseIncludable ResponseInputContentParam
            ResponseInputTextParam ResponseInputTokensDetails ResponseOutputTokensDetails
            ResponsePromptParam ResponseTextConfigParam ResponseToolSearchCall
            ResponseToolSearchOutputItem ResponseUsage ResponsesModel SkillReferenceParam StreamOptions
            Summary Tag ToolChoiceAllowedParam ToolChoiceApplyPatchParam ToolChoiceCustomParam
            ToolChoiceFunctionParam ToolChoiceMcpParam ToolChoiceOptions ToolChoiceShellParam
            ToolChoiceTypesParam ToolSearchToolParam WebSearchOptions WebSearchPreviewToolParam
            WebSearchToolParam field_serializer model_serializer
            """,
        ),
        # openai_utils imported this from typing_extensions.
        **_names("typing_extensions", "0.8.0", "TypedDict"),
        # Added on main after the split was cut.
        **_names(
            "nemo_gym.core.responses_api_types",
            "0.8.0",
            "NeMoGymResponseReasoningInputItem NeMoGymResponseReasoningInputItemForTraining",
        ),
    },
    "nemo_gym.rollout_correlation": {
        **_names(
            "nemo_gym.core.episode",
            "0.8.0",
            """
            VERIFIER_INVOCATION_HEADER _ROLLOUT_ID _VERIFIER_INVOCATION_ID current_episode_id
            current_rollout_id current_verifier_invocation_id maybe_rollout_id_from_run_body
            verifier_invocation_context
            """,
        ),
        # rollout_correlation imported these and they stay importable through it.
        **_names(
            "nemo_gym.config.global_config",
            "0.8.0",
            """
            ATTEMPT_INDEX_KEY_NAME ROLLOUT_ID_KEY_NAME ROLLOUT_INDEX_KEY_NAME TASK_INDEX_KEY_NAME
            """,
        ),
        # rollout_correlation imported these and they stay importable through it.
        **_names("nemo_gym.core.episode", "0.8.0", "EpisodeId"),
        # rollout_correlation imported these and they stay importable through it.
        **_names("pydantic", "0.8.0", "BaseModel"),
        # Moved, and still imported by rollout_correlation for its own use.
        **_names("nemo_gym.core.episode", "0.8.0", "ROLLOUT_ID_PATTERN rollout_context"),
    },
    "nemo_gym.episode_types": {
        **_names("nemo_gym.core.episode", "0.8.0", "_CAPTURE_KEY_ATTEMPT"),
        # Moved, and still imported by episode_types for its own use.
        **_names("nemo_gym.core.episode", "0.8.0", "EpisodeId TaskId"),
    },
    "nemo_gym.rollout_collection": {
        **_names("nemo_gym.artifacts.rollout_rows", "0.8.0", "_MODEL_CALL_PAYLOAD_KEYS"),
        # Moved, and still imported by rollout_collection for its own use.
        **_names(
            "nemo_gym.artifacts.rollout_rows",
            "0.8.0",
            "NG_DISPATCH_DRAINED_KEY NG_FAILURE_CLASS_KEY NG_NO_PERSIST_KEY _rollout_for_export _strip_capture_payloads",
        ),
        # Split out of rollout_collection.
        **_names("nemo_gym.rollouts.rollout_perf", "0.8.0", "_build_ng_perf"),
        **_names(
            "nemo_gym.rollouts.trajectory_builder",
            "0.8.0",
            """
            _build_trajectory_record _nonnegative_int _trajectory_identity _turn_content _turns_from_model_calls
            """,
        ),
        # Moved, and still imported by rollout_collection for its own use.
        **_names("nemo_gym.rollouts.rollout_perf", "0.8.0", "DispatchLatencyTracker NG_PERF_KEY _attach_ng_perf"),
        **_names("nemo_gym.rollouts.trajectory_builder", "0.8.0", "_attach_trajectory_record _has_observation_gap"),
        # rollout_collection imported these for code that moved out of it.
        **_names(
            "nemo_gym.trajectory.trajectory_record",
            "0.8.0",
            """
            AgentInvocation AgentObservationBundle ModelCallRef ObservationGap ToolCallObservation
            TrajectoryModelCall TrajectoryRecord TrajectoryTokenStats TrajectoryToolCall TrajectoryTurn
            VerifierInvocation
            """,
        ),
    },
    "nemo_gym.health.types": {
        **_names(
            "nemo_gym.artifacts.run_files",
            "0.8.0",
            "QUALITY_SUMMARY_FILENAME QUALITY_SUMMARY_SOURCES_KEY ROLLOUT_VERDICTS_FILENAME",
        ),
    },
    "nemo_gym.observability.model_calls": {
        # Split out of observability.model_calls.
        **_names(
            "nemo_gym.trajectory.model_call_record",
            "0.8.0",
            "_as_arguments _cache_signal _prompt_size _token_count _tool_calls_and_reasoning _usage_detail_token extract_token_stats",
        ),
        # Moved, and still imported by observability.model_calls for its own use.
        **_names(
            "nemo_gym.trajectory.model_call_record",
            "0.8.0",
            "ModelCallRecord aggregate_model_call_records build_model_call_record",
        ),
        # observability.model_calls imported this for code that moved out of it.
        **_names("nemo_gym.config.config_types", "0.8.0", "ModelServerRef"),
    },
    "nemo_gym.batch_status": {
        # Moved, and still imported by batch_status for its own use.
        **_names("nemo_gym.artifacts.run_files", "0.8.0", "AGGREGATION_ERROR_KEY"),
    },
    "nemo_gym.base_responses_api_model": {
        # Split out of base_responses_api_model.
        **_names(
            "nemo_gym.models.model_call_capture",
            "0.8.0",
            """
            CaptureStore ModelExecutionOutcome _CLIENT_SESSION_HEADER _CaptureMiddleware _OBSERVED_PATHS
            _PENDING_SUFFIX _ROLLOUT_PATH_RE _TERMINAL_SSE_LINES _classify_exception _classify_status
            _consume_terminal_sse_event _exception_http_details _fail_uncommitted_external_call
            _headers_content_type _parse_sse_events _pending_token _preserve_capture_prefix_on_redirect
            _reconstruct_anthropic_sse _reconstruct_chat_sse _reconstruct_responses_sse
            _reconstruct_streamed_response _record _unique_request_header _validate_rollout_id
            make_capture_store start_model_execution
            """,
        ),
        # Moved, and still imported by base_responses_api_model for its own use.
        **_names(
            "nemo_gym.models.model_call_capture",
            "0.8.0",
            """
            ModelCallCaptureConfig _SSE_KEEPALIVE install_model_call_capture
            """,
        ),
        # base_responses_api_model imported these for code that moved out of it.
        **_names("nemo_gym.config.config_types", "0.8.0", "ROLLOUT_PATH_PREFIX TOKEN_CAPTURE_PATH_SEGMENT"),
        **_names("nemo_gym.core.episode", "0.8.0", "VERIFIER_INVOCATION_HEADER rollout_context"),
        **_names("nemo_gym.models._checkpoint", "0.8.0", "attach_capture_context ledger_removal_refusal"),
        **_names("nemo_gym.core.responses_api_types", "0.8.0", "SSE_KEEPALIVE"),
        **_names("nemo_gym.models.model_call_capture", "0.8.0", "model_validator"),
        **_names("nemo_gym.models.responses_streaming", "0.8.0", "is_responses_lifecycle_chunk"),
        **_names(
            "nemo_gym.models.token_capture",
            "0.8.0",
            """
            CaptureContext installed_lineage_store installed_token_sink reset_token_sink set_token_sink
            """,
        ),
        **_names("nemo_gym.models.token_capture.control_routes", "0.8.0", "install_rollout_control_routes"),
        **_names("nemo_gym.models.token_capture.lineage", "0.8.0", "FileLineageStore InMemoryLineageStore"),
        **_names("nemo_gym.models.token_capture.protocols", "0.8.0", "ParentResolutionMode"),
        **_names("nemo_gym.models.token_capture.records", "0.8.0", "UNCOMMITTED_CALL_REASON"),
        **_names("nemo_gym.models.token_capture.store", "0.8.0", "make_token_store"),
        **_names("nemo_gym.server_runtime.request_handling.rollout_context", "0.8.0", "scope_with_routed_path"),
    },
    "nemo_gym.chat_streaming": {
        # Split out of chat_streaming.
        **_names("nemo_gym.core.responses_api_types", "0.8.0", "SSE_KEEPALIVE"),
    },
    "nemo_gym.tasks.harbor.cli": {
        # Split out of cli.
        **_names(
            "nemo_gym.cli.eval",
            "0.8.0",
            """
            COMPOSE_OVERLAY_KEY GPU_SANDBOX_PROVIDER_KEY HARBOR_SERVER_KEY SANDBOX_BLOCK_KEY
            SANDBOX_IMAGES_AUTH_KEY SANDBOX_PROVIDER_OPTIONS_KEY _block_provider_name _block_shared_storage
            _block_volume_sources _clear_collector_metrics _collector_split _patterns _requested_output
            _resolved_collection _resumable_cache _resume_conflict _rows_name _write_summary
            compose_overlay_from_overrides compose_platform_from_overrides gpu_sandbox_provider_from_overrides
            gpu_shared_storage_from_overrides gpu_volume_sources_from_overrides pairing_override_from_overrides
            registry_credentials_from_overrides run_target sandbox_provider_from_overrides
            shared_storage_from_overrides validate_target volume_sources_from_overrides
            """,
        ),
        # cli imported these for code that moved out of it.
        **_names("nemo_gym.cli.eval", "0.8.0", "ValidationError"),
        **_names("nemo_gym.cli.utils", "0.8.0", "exit_cleanly_on_config_error"),
        **_names("nemo_gym.config.config_types", "0.8.0", "ConfigError"),
        **_names(
            "nemo_gym.data.tasks.harbor.build_options",
            "0.8.0",
            """
            _block_from_overrides _merged_cli_config build_options_from_overrides
            """,
        ),
        **_names("nemo_gym.data.tasks.harbor.build_plan", "0.8.0", "BuildError"),
        **_names("nemo_gym.data.tasks.harbor.hub", "0.8.0", "datasets_dir"),
        **_names(
            "nemo_gym.data.tasks.harbor.materialize",
            "0.8.0",
            """
            ENVIRONMENT_SERVER_IMPL ENVIRONMENT_SERVER_INSTANCE
            """,
        ),
        **_names("nemo_gym.data.tasks.harbor.registry", "0.8.0", "ImageConfigError credentials_for_image_prefixes"),
        **_names(
            "nemo_gym.data.tasks.harbor.run_summary",
            "0.8.0",
            """
            NO_REFERENCE_SOLUTION output_signatures write_run_summary
            """,
        ),
        **_names(
            "nemo_gym.data.tasks.harbor.validation_report",
            "0.8.0",
            """
            _task_digests _validation_key _validation_rows summarize_validation
            """,
        ),
        **_names("nemo_gym.sandbox.compose_config", "0.8.0", "deployment_platform"),
        **_names("nemo_gym.sandbox.config", "0.8.0", "SANDBOX_BLOCK_VOLUME_SOURCES_KEY validate_volume_sources"),
        **_names("nemo_gym.utils.path_utils", "0.8.0", "materialized_path_for"),
    },
    "nemo_gym.cli.env": {
        # Split out of env.
        **_names(
            "nemo_gym.orchestration.server_supervisor",
            "0.8.0",
            """
            RAY_ENABLE_AUTO_CONNECT_ENV_VAR_NAME ServerSpinupTimeoutError _BASE_URL_KEY_SUFFIX _ConfiguredServer
            _DEFAULT_MODEL_ENDPOINT_READINESS_TIMEOUT_SEC _DEFAULT_SERVER_SPINUP_TIMEOUT_SEC _ENDPOINT_ANSWERING
            _ENDPOINT_POLL_INTERVAL_SEC _ENDPOINT_PROBE_TIMEOUT_SEC _ENDPOINT_REFUSED _ENDPOINT_STARTING
            _ENDPOINT_STARTING_STATUS_CODES _ENDPOINT_UNRESOLVABLE _FORCE_KILL_REAP_TIMEOUT_SEC
            _GRACEFUL_SHUTDOWN_TIMEOUT_SEC _MODEL_SERVER_TYPE _SERVER_PROBE_TIMEOUT_SEC _collect_model_endpoints
            _configured_servers _endpoint_probe_url _is_dns_failure _model_endpoint_timeout_seconds
            _probe_endpoint _ray_server_names _server_launch_command _server_spinup_timeout_seconds
            _startup_outcome _wait_for_model_endpoints
            """,
        ),
        # Moved, and still imported by env for its own use.
        **_names("nemo_gym.orchestration.server_supervisor", "0.8.0", "RunHelper _resolve_server_dir"),
        # env imported these for code that moved out of it.
        **_names("nemo_gym.client.server_client", "0.8.0", "ServerClient ServerStatus"),
        **_names(
            "nemo_gym.config.global_config",
            "0.8.0",
            """
            DRY_RUN_KEY_NAME HEAD_SERVER_KEY_NAME MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME
            NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME
            SERVER_SPINUP_TIMEOUT_SECONDS_KEY_NAME
            """,
        ),
        **_names(
            "nemo_gym.orchestration.h2_ping_sidecar.launcher",
            "0.8.0",
            """
            H2PingSidecarManager start_h2_ping_sidecar
            """,
        ),
        **_names(
            "nemo_gym.orchestration.server_supervisor",
            "0.8.0",
            """
            ListConfig _resolve_under_cwd_or_install pprint
            """,
        ),
        **_names(
            "nemo_gym.server_runtime.ray_support",
            "0.8.0",
            """
            entrypoint_ray_enabled initialize_ray ray_is_installed
            """,
        ),
        **_names("nemo_gym.server_runtime.registry_server", "0.8.0", "HeadServer ServerInstanceDisplayConfig"),
        **_names("nemo_gym.telemetry.config", "0.8.0", "MemoryProfilingConfig"),
        **_names(
            "nemo_gym.telemetry.memory",
            "0.8.0",
            """
            MemoryProfiler ServerMemoryTarget process_tree_memory_supported
            """,
        ),
        **_names("nemo_gym.telemetry.metrics", "0.8.0", "record_active_servers"),
        **_names(
            "nemo_gym.telemetry.setup",
            "0.8.0",
            """
            configure_telemetry_env get_telemetry init_telemetry is_telemetry_metrics_enabled
            memory_profiling_config_from_env shutdown_telemetry telemetry_config_from_global_config
            """,
        ),
        **_names("nemo_gym.telemetry.startup", "0.8.0", "STARTUP_SETUP_DONE_NS_ENV SupervisorStartup"),
    },
}

# Names renamed with the token capture package (token_id_capture -> token_capture), at each old module path that
# defined or imported them. Kept apart so a module path that MOVED_NAMES already lists keeps its other entries.
_TOKEN_CAPTURE = "nemo_gym.models.token_capture"
_RENAMED_TOKEN_CAPTURE_NAMES: dict[str, dict[str, MovedPath]] = {
    "nemo_gym.token_id_capture": {
        "TokenIdCaptureConfig": MovedPath(new=f"{_TOKEN_CAPTURE}.config:TokenCaptureConfig", since="0.8.0"),
        "token_id_capture_dirs_from_config": MovedPath(
            new=f"{_TOKEN_CAPTURE}.consumer:token_capture_dirs_from_config", since="0.8.0"
        ),
    },
    "nemo_gym.token_id_capture.config": {
        "TOKEN_ID_CAPTURE_BLOCK": MovedPath(new=f"{_TOKEN_CAPTURE}.config:TOKEN_CAPTURE_BLOCK", since="0.8.0"),
        "TokenIdCaptureConfig": MovedPath(new=f"{_TOKEN_CAPTURE}.config:TokenCaptureConfig", since="0.8.0"),
        "TokenIdCaptureSettings": MovedPath(new=f"{_TOKEN_CAPTURE}.config:TokenCaptureSettings", since="0.8.0"),
        "token_id_capture_config": MovedPath(new=f"{_TOKEN_CAPTURE}.config:token_capture_config", since="0.8.0"),
        "token_id_capture_enabled_for_agent": MovedPath(
            new=f"{_TOKEN_CAPTURE}.config:token_capture_enabled_for_agent", since="0.8.0"
        ),
    },
    "nemo_gym.token_id_capture.consumer": {
        "token_id_capture_dirs_from_config": MovedPath(
            new=f"{_TOKEN_CAPTURE}.consumer:token_capture_dirs_from_config", since="0.8.0"
        ),
    },
    "nemo_gym.global_config": {
        "TOKEN_ID_CAPTURE_BLOCK": MovedPath(new="nemo_gym.config.global_config:TOKEN_CAPTURE_BLOCK", since="0.8.0"),
    },
    "nemo_gym.base_responses_api_agent": {
        "TOKEN_ID_CAPTURE_BLOCK": MovedPath(new="nemo_gym.config.global_config:TOKEN_CAPTURE_BLOCK", since="0.8.0"),
    },
    "nemo_gym._checkpoint.model": {
        "token_id_capture_enabled_for_agent": MovedPath(
            new=f"{_TOKEN_CAPTURE}.config:token_capture_enabled_for_agent", since="0.8.0"
        ),
    },
    "nemo_gym.rollout_collection": {
        "TokenIdCaptureConfig": MovedPath(new=f"{_TOKEN_CAPTURE}.config:TokenCaptureConfig", since="0.8.0"),
        "token_id_capture_dirs_from_config": MovedPath(
            new=f"{_TOKEN_CAPTURE}.consumer:token_capture_dirs_from_config", since="0.8.0"
        ),
        "token_id_capture_enabled_for_agent": MovedPath(
            new=f"{_TOKEN_CAPTURE}.config:token_capture_enabled_for_agent", since="0.8.0"
        ),
    },
}
for _module, _renamed in _RENAMED_TOKEN_CAPTURE_NAMES.items():
    _listed = MOVED_NAMES.setdefault(_module, {})
    assert not set(_listed) & set(_renamed), f"{_module} lists a renamed token capture name twice"
    _listed.update(_renamed)


def _running_version() -> str:
    from nemo_gym.package_info import __version__

    return __version__


def deprecation_message(old: str, moved: MovedPath, new: str) -> str:
    """The warning text for a use of the old path `old`, which now lives at `new`."""
    return (
        f"`{old}` moved to `{new}` in nemo_gym {moved.since} and the old path will be removed in nemo_gym "
        f"{moved.removed_in} (running {_running_version()}). Import from `{new}` instead."
    )


def _file_prefix(path: str) -> str:
    # CPython matches skip_file_prefixes only as a strict prefix; "module." also keeps "module_utils.py" out.
    return os.path.splitext(path)[0] + "."


_STDLIB = os.path.dirname(os.__file__)
# Frames a moved-path warning skips to reach the code that used the old path: this module, the import machinery,
# the package root (`from nemo_gym import <moved module>`), and pkgutil and mock (`mock.patch("<old path>.name")`).
_SKIP_FILE_PREFIXES = (
    _file_prefix(__file__),
    _file_prefix(os.path.join(os.path.dirname(__file__), "__init__.py")),
    os.path.dirname(importlib.__file__) + os.sep,
    "<frozen importlib",
    _file_prefix(os.path.join(_STDLIB, "pkgutil.py")),
    _file_prefix(os.path.join(_STDLIB, "unittest", "mock.py")),
)


def warn_moved(message: str, *, old_module: str) -> None:
    """Warn about a moved path at the frame that used it, past the getter that calls this and `old_module`'s own
    `__getattr__` wrappers."""
    files = [sys._getframe(1).f_code.co_filename, getattr(sys.modules.get(old_module), "__file__", None)]
    skip = _SKIP_FILE_PREFIXES + tuple(_file_prefix(file) for file in files if file)
    warnings.warn(message, DeprecationWarning, skip_file_prefixes=skip)


def _resolve(fullname: str, moves: Mapping[str, MovedPath]) -> Optional[tuple[str, str, MovedPath]]:
    """Return `(old key, new module name, entry)` for the longest moved prefix of `fullname`, if any."""
    old = fullname
    while True:
        moved = moves.get(old)
        if moved is not None:
            return old, moved.new + fullname[len(old) :], moved
        old, sep, _ = old.rpartition(".")
        if not sep:
            return None


class MovedModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve old module paths as aliases of the modules that replaced them.

    It must sit first on `sys.meta_path`.
    Once an old package name is bound to the new package, the normal path finder would find the old package's submodules through the new package's `__path__`.
    It would then run a second copy of each submodule under its old name.
    """

    def __init__(self, moves: Mapping[str, MovedPath]) -> None:
        self._moves = moves
        self._own_specs: dict[str, importlib.machinery.ModuleSpec] = {}

    def find_spec(self, fullname: str, path: Any = None, target: Optional[ModuleType] = None) -> Any:
        resolved = _resolve(fullname, self._moves)
        if resolved is None:
            return None
        _, new_name, _ = resolved
        # Report the new module's file and whether it is a package.
        # `importlib.util.find_spec(<old path>).origin` then names the file that runs.
        new_spec = importlib.util.find_spec(new_name)
        if new_spec is None:
            return None
        return importlib.machinery.ModuleSpec(
            fullname, self, origin=new_spec.origin, is_package=new_spec.submodule_search_locations is not None
        )

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        old, new_name, moved = _resolve(spec.name, self._moves)
        warnings.warn(
            deprecation_message(spec.name, moved, new_name),
            DeprecationWarning,
            skip_file_prefixes=_SKIP_FILE_PREFIXES,
        )
        # The import system keeps the module's own __name__ and __loader__, and sets __spec__ to the alias spec;
        # exec_module puts the module's own spec back.
        # It binds the module under the old name in sys.modules and on the old parent package.
        module = importlib.import_module(new_name)
        self._own_specs[module.__name__] = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        # The module already ran when it was imported under its new name.
        # Its own spec keeps `importlib.util.find_spec(<new path>)` and `importlib.reload` working.
        module.__spec__ = self._own_specs.pop(module.__name__, module.__spec__)

    def get_code(self, fullname: str) -> Any:
        """Give `python -m <old path>` the new module's code, with the same warning."""
        _, new_name, moved = _resolve(fullname, self._moves)
        warnings.warn(deprecation_message(fullname, moved, new_name), DeprecationWarning, stacklevel=2)
        return importlib.util.find_spec(new_name).loader.get_code(new_name)


def install_moved_module_finder() -> None:
    """Put the finder for `MOVED_MODULES` first on `sys.meta_path`; calling it again changes nothing."""
    if not any(isinstance(finder, MovedModuleFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, MovedModuleFinder(MOVED_MODULES))


def _names_were_listed_under(module_name: str) -> list[str]:
    """`module_name`, then each old path that `MOVED_MODULES` maps to it, directly or as part of a moved package."""
    paths = [module_name]
    for old, moved in MOVED_MODULES.items():
        if module_name == moved.new or module_name.startswith(moved.new + "."):
            paths.append(old + module_name[len(moved.new) :])
    return paths


def moved_names_getattr(module_name: str) -> Callable[[str], Any]:
    """Build a module `__getattr__` that serves the names `MOVED_NAMES[module_name]` lists.

    Assign the result to `__getattr__` in the old module.
    In a package, it also serves each moved submodule that `MOVED_MODULES` lists under the package,
    so `package.old_submodule` resolves without an import of it first.
    Names the old module still defines or imports never reach it.
    If the module itself moves later, its entries stay under its old path, and this `__getattr__`
    keeps serving them once the module runs under its new name.
    """

    warned: set[str] = set()

    def __getattr__(name: str) -> Any:
        for listed_as in _names_were_listed_under(module_name):
            moved = MOVED_NAMES.get(listed_as, {}).get(name)
            if moved is not None:
                break
        else:
            for listed_as in _names_were_listed_under(module_name):
                if f"{listed_as}.{name}" in MOVED_MODULES:
                    # The finder warns, and binds the module on the package.
                    return importlib.import_module(f"{listed_as}.{name}")
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        new_module, _, new_name = moved.new.partition(":")
        new_name = new_name or name
        # Look the name up on every access, so a patch of the new module shows through; warn once per old path.
        value = getattr(importlib.import_module(new_module), new_name)
        if name not in warned:
            warn_moved(
                deprecation_message(f"{listed_as}.{name}", moved, f"{new_module}.{new_name}"), old_module=module_name
            )
            warned.add(name)
        return value

    return __getattr__
