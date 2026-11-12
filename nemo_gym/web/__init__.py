# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light contracts shared by NeMo Gym web agents and runtimes."""

from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.web.actions import MAX_SCROLL_AMOUNT, ActionParseError, parse_nano_omni_tool_calls
    from nemo_gym.web.api_models import (
        WebCloseResponse,
        WebEvaluateRequest,
        WebEvaluateResponse,
        WebResetRequest,
        WebSeedSessionRequest,
        WebSeedSessionResponse,
        WebSessionStatusResponse,
        WebStepRequest,
        WebStepResponse,
        WebVerifyRequest,
        WebVerifyResponse,
    )
    from nemo_gym.web.artifacts import WebArtifactStore
    from nemo_gym.web.browser_instance import TRANSPORT, RemoteWebBackend
    from nemo_gym.web.browser_session import (
        BrowserSessionError,
        BrowserSessionHandle,
        BrowserSessionProvider,
        BrowserSessionSpec,
        LocalProcessBrowserSessionProvider,
        RenewableBrowserSessionProvider,
        create_browser_session_provider,
        list_browser_session_providers,
        register_browser_session_provider,
    )
    from nemo_gym.web.chromium import ensure_chromium
    from nemo_gym.web.composed_backend import ComposedWebBackend, WebBrowserDriver, WebTaskEvaluator
    from nemo_gym.web.computer_use import NANO_OMNI_SYSTEM_PROMPT, nano_omni_tools
    from nemo_gym.web.datasets import adapt_webvoyager_record, load_json_records, write_jsonl
    from nemo_gym.web.judge_evidence import compact_webvoyager_judge_evidence, expand_webvoyager_judge_screenshots
    from nemo_gym.web.models import (
        BROWSER_TARGET_CLOSED_STATUS,
        CAPTCHA_BUDGET_EXHAUSTED_STATUS,
        WebAction,
        WebActionLimits,
        WebActionProfile,
        WebArtifactRef,
        WebBenchmark,
        WebImage,
        WebObservation,
        WebObservationProfile,
        WebRuntimeProfile,
        WebStepResult,
        WebTab,
        WebTask,
        WebVerifierResult,
    )
    from nemo_gym.web.operation_runner import (
        DirectWebOperationRunner,
        ThreadAffineWebOperationRunner,
        WebOperationRunner,
    )
    from nemo_gym.web.resources_server import WebResourcesServer
    from nemo_gym.web.session import (
        BenchmarkPreconditionError,
        CapacityUnavailableError,
        EvaluatorConfigurationError,
        EvaluatorInfrastructureError,
        SessionConflictError,
        SessionNotFoundError,
        WebSessionState,
    )
    from nemo_gym.web.session_manager import WebSessionManager
    from nemo_gym.web.site_pool import LocalSiteLockPool, SiteLease, SitePool, UnmanagedSitePool
    from nemo_gym.web.task_images import resolve_task_image_url
    from nemo_gym.web.visual_browser import (
        RESET_WAIT_UNTIL,
        VisualBrowserDriver,
        VisualBrowserDriverConfig,
        VisualBrowserEvaluationContext,
        is_retryable_navigation_transport_error,
    )


_PUBLIC_API = {
    "ActionParseError": "actions",
    "BROWSER_TARGET_CLOSED_STATUS": "models",
    "BenchmarkPreconditionError": "session",
    "BrowserSessionError": "browser_session",
    "BrowserSessionHandle": "browser_session",
    "BrowserSessionProvider": "browser_session",
    "BrowserSessionSpec": "browser_session",
    "CAPTCHA_BUDGET_EXHAUSTED_STATUS": "models",
    "CapacityUnavailableError": "session",
    "ComposedWebBackend": "composed_backend",
    "DirectWebOperationRunner": "operation_runner",
    "EvaluatorConfigurationError": "session",
    "EvaluatorInfrastructureError": "session",
    "LocalProcessBrowserSessionProvider": "browser_session",
    "LocalSiteLockPool": "site_pool",
    "MAX_SCROLL_AMOUNT": "actions",
    "NANO_OMNI_SYSTEM_PROMPT": "computer_use",
    "RESET_WAIT_UNTIL": "visual_browser",
    "RemoteWebBackend": "browser_instance",
    "RenewableBrowserSessionProvider": "browser_session",
    "SessionConflictError": "session",
    "SessionNotFoundError": "session",
    "SiteLease": "site_pool",
    "SitePool": "site_pool",
    "TRANSPORT": "browser_instance",
    "ThreadAffineWebOperationRunner": "operation_runner",
    "UnmanagedSitePool": "site_pool",
    "VisualBrowserDriver": "visual_browser",
    "VisualBrowserDriverConfig": "visual_browser",
    "VisualBrowserEvaluationContext": "visual_browser",
    "WebAction": "models",
    "WebActionLimits": "models",
    "WebActionProfile": "models",
    "WebArtifactRef": "models",
    "WebArtifactStore": "artifacts",
    "WebBenchmark": "models",
    "WebBrowserDriver": "composed_backend",
    "WebCloseResponse": "api_models",
    "WebEvaluateRequest": "api_models",
    "WebEvaluateResponse": "api_models",
    "WebImage": "models",
    "WebObservation": "models",
    "WebObservationProfile": "models",
    "WebOperationRunner": "operation_runner",
    "WebResetRequest": "api_models",
    "WebResourcesServer": "resources_server",
    "WebRuntimeProfile": "models",
    "WebSeedSessionRequest": "api_models",
    "WebSeedSessionResponse": "api_models",
    "WebSessionManager": "session_manager",
    "WebSessionState": "session",
    "WebSessionStatusResponse": "api_models",
    "WebStepRequest": "api_models",
    "WebStepResponse": "api_models",
    "WebStepResult": "models",
    "WebTab": "models",
    "WebTask": "models",
    "WebTaskEvaluator": "composed_backend",
    "WebVerifierResult": "models",
    "WebVerifyRequest": "api_models",
    "WebVerifyResponse": "api_models",
    "adapt_webvoyager_record": "datasets",
    "compact_webvoyager_judge_evidence": "judge_evidence",
    "create_browser_session_provider": "browser_session",
    "ensure_chromium": "chromium",
    "expand_webvoyager_judge_screenshots": "judge_evidence",
    "is_retryable_navigation_transport_error": "visual_browser",
    "list_browser_session_providers": "browser_session",
    "load_json_records": "datasets",
    "nano_omni_tools": "computer_use",
    "parse_nano_omni_tool_calls": "actions",
    "register_browser_session_provider": "browser_session",
    "resolve_task_image_url": "task_images",
    "write_jsonl": "datasets",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
