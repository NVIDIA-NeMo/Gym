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
        # Moved, and still imported by server_utils for its own use.
        **_names(
            "nemo_gym.client.http_client",
            "0.8.0",
            "GlobalAIOHTTPAsyncClientConfig _detach_global_aiohttp_clients _format_upstream_error_log _set_tcp_keepalive global_aiohttp_client_exit is_global_aiohttp_client_request_debug_enabled",
        ),
        **_names(
            "nemo_gym.client.server_client", "0.8.0", "SESSION_ID_KEY ServerClient ServerStatus _TRACEPARENT_PATTERN"
        ),
        **_names("nemo_gym.telemetry.profiling", "0.8.0", "ProfilingMiddlewareConfig"),
        **_names(
            "nemo_gym.client.http_client",
            "0.8.0",
            """
            DISCONNECTED_CLIENT_OS_HELP_TEXT DISCONNECTED_CLIENT_OS_PRINT_INTERVAL MAX_NUM_TRIES
            _GLOBAL_AIOHTTP_CLIENT _GLOBAL_AIOHTTP_CLIENT_CONFIG _GLOBAL_AIOHTTP_CLIENT_LOOP
            _GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG
            _GLOBAL_AIOHTTP_CONTROL_CLIENT _NUM_CLIENT_OS_ERROR _NUM_SERVER_DISCONNECTED_ERROR
            _PickleSafeRequestInfo _UPSTREAM_ERROR_LOG_BODY_CHARS _format_upstream_error_body
            _make_keepalive_socket_factory _redacted_url _request_with_retries _traced_request
            format_exception_with_response_body get_global_aiohttp_client
            get_global_aiohttp_control_client get_response_json is_global_aiohttp_client_setup
            raise_for_status request set_global_aiohttp_client
            """,
        ),
        **_names(
            "nemo_gym.client.server_client",
            "0.8.0",
            """
            NEMO_GYM_MODEL_SERVER_BASE_URL_ENV_VAR_NAME NEMO_GYM_MODEL_SERVER_NAME_ENV_VAR_NAME
            _connectable_client_host apply_rollout_prefix get_server_url rollout_path_prefix
            setup_server_client trace_context_path_segment
            """,
        ),
        **_names("nemo_gym.telemetry.profiling", "0.8.0", "ProfilingMiddlewareInputConfig"),
        # Names server_utils imported and no longer uses; each resolves where it is defined or now imported.
        **_names("nemo_gym._checkpoint.settings", "0.8.0", "checkpoint_settings"),
        **_names(
            "nemo_gym.client.http_client",
            "0.8.0",
            """
            CIMultiDict ClientConnectorError ClientOSError ClientResponse ClientSession ClientTimeout
            DummyCookieJar Field ServerDisconnectedError _RequestOptions
            """,
        ),
        **_names("nemo_gym.client.server_client", "0.8.0", "ConfigDict ConnectionError"),
        **_names("nemo_gym.config.config_types", "0.8.0", "HeadServerUnreachableError"),
        **_names(
            "nemo_gym.config.global_config",
            "0.8.0",
            """
            GlobalConfigDictParser GlobalConfigDictParserConfig HEAD_SERVER_KEY_NAME
            OBSERVABILITY_ENABLED_KEY_NAME maybe_get_global_config_dict
            """,
        ),
        **_names(
            "nemo_gym.core.episode",
            "0.8.0",
            """
            VERIFIER_INVOCATION_HEADER current_rollout_id current_verifier_invocation_id
            maybe_rollout_id_from_run_body
            """,
        ),
        **_names("nemo_gym.telemetry._fallbacks", "0.8.0", "safe_set_span_attributes"),
        **_names(
            "nemo_gym.telemetry.connection_pool",
            "0.8.0",
            """
            QueueTimedTCPConnector build_connection_pool_connector reset_server_name set_server_name
            """,
        ),
        **_names("nemo_gym.utils.server_process", "0.8.0", "get_nemo_gym_fastapi_num_workers"),
        **_names(
            "nemo_gym.utils.server_process",
            "0.8.0",
            """
            IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME
            NEMO_GYM_FASTAPI_NUM_WORKERS _has_injected_global_config_env
            get_nemo_gym_fastapi_num_workers is_nemo_gym_fastapi_entrypoint is_nemo_gym_fastapi_worker
            set_is_nemo_gym_fastapi_entrypoint set_is_nemo_gym_fastapi_worker
            set_nemo_gym_fastapi_num_workers
            """,
        ),
        # server_utils imported this before the split and no longer uses it.
        **_names("nemo_gym.config.global_config", "0.8.0", "NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME"),
    },
    "nemo_gym.cli.utils": _names("nemo_gym.utils.text", "0.8.0", "did_you_mean"),
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
        **_names("nemo_gym.core.responses_api_types", "0.8.0", "BaseModel"),
        # Moved, and still imported by rollout_correlation for its own use.
        **_names("nemo_gym.core.episode", "0.8.0", "ROLLOUT_ID_PATTERN rollout_context"),
    },
    "nemo_gym.episode_types": {
        **_names("nemo_gym.core.episode", "0.8.0", "_CAPTURE_KEY_ATTEMPT"),
        # Moved, and still imported by episode_types for its own use.
        **_names("nemo_gym.core.episode", "0.8.0", "EpisodeId TaskId"),
    },
}


def _running_version() -> str:
    from nemo_gym.package_info import __version__

    return __version__


def deprecation_message(old: str, moved: MovedPath, new: str) -> str:
    """The warning text for a use of the old path `old`, which now lives at `new`."""
    return (
        f"`{old}` moved to `{new}` in nemo_gym {moved.since} and the old path will be removed in nemo_gym "
        f"{moved.removed_in} (running {_running_version()}). Import from `{new}` instead."
    )


# Frames to skip so the warning points at the code that imported the old path: this file and importlib.
_SKIP_FILE_PREFIXES = (__file__, os.path.dirname(importlib.__file__) + os.sep, "<frozen importlib")


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
        # The import system keeps the module's own __name__, __spec__ and __loader__.
        # It binds the module under the old name in sys.modules and on the old parent package.
        return importlib.import_module(new_name)

    def exec_module(self, module: ModuleType) -> None:
        # The module already ran when it was imported under its new name.
        pass


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
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        new_module, _, new_name = moved.new.partition(":")
        new_name = new_name or name
        # Look the name up on every access, so a patch of the new module shows through; warn once per old path.
        value = getattr(importlib.import_module(new_module), new_name)
        if name not in warned:
            warnings.warn(
                deprecation_message(f"{listed_as}.{name}", moved, f"{new_module}.{new_name}"),
                DeprecationWarning,
                stacklevel=2,
            )
            warned.add(name)
        return value

    return __getattr__
