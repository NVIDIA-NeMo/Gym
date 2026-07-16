# SPDX-License-Identifier: Apache-2.0
"""Prototype test servers, built from UNMODIFIED in-tree resources_servers/ code.

This branch is based on origin/main, so the resources_servers/ handlers are pristine. This module
imports them directly (no snapshot needed) — which is the whole point: the same unmodified handler
files are MCP-enabled by setting expose_tools_over_mcp=True (run_webserver auto-installs). It only:
  * instantiates those unmodified server classes (configs + fixtures, exactly like their tests do),
  * adds the ONE dispatcher override Brian's design allows (workplace ``mcp_tool_inventory``),
  * provides trivially concrete aviary plumbing (a DummyEnv dataset — fixture, not handler code)
    when the optional fhaviary dependency is installed; otherwise the aviary case is skipped.

Zero decorators. Zero handler edits.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock


PROTO_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROTO_DIR.parents[1]  # prototypes/mcp_auto_exposure/ -> repo root
sys.path.insert(0, str(REPO_ROOT))  # the real, unmodified resources_servers/ + nemo_gym

from fastapi import FastAPI  # noqa: E402
from pydantic import Field  # noqa: E402

from nemo_gym.config_types import ModelServerRef  # noqa: E402
from nemo_gym.mcp_auto_exposure import maybe_auto_expose  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402

# ---- unmodified in-tree classes (resources_servers/, pristine on this origin/main-based branch) --
from resources_servers.finance_sec_search.app import (  # noqa: E402
    FinanceAgentResourcesServer,
    FinanceAgentResourcesServerConfig,
)
from resources_servers.workplace_assistant.app import (  # noqa: E402
    WorkbenchResourcesServer,
    WorkbenchResourcesServerConfig,
)
from resources_servers.workplace_assistant.utils import get_tools  # noqa: E402


# aviary (the plumbing-exposed case) needs the optional fhaviary package. Import lazily so the
# finance + workplace demonstrations run without it.
try:
    from aviary.core import DummyEnv, TaskDataset  # noqa: E402

    from resources_servers.aviary.app import AviaryResourcesServer  # noqa: E402
    from resources_servers.aviary.schemas import AviaryResourcesServerConfig  # noqa: E402

    AVIARY_AVAILABLE = True
except ImportError:
    AVIARY_AVAILABLE = False


MOCK_TICKERS = {
    "0": {"ticker": "AAPL", "cik_str": "320193", "title": "APPLE INC."},
    "1": {"ticker": "NVDA", "cik_str": "1045810", "title": "NVIDIA CORP"},
}

WORKBENCH_TOOLKITS = [
    "email",
    "calendar",
    "analytics",
    "project_management",
    "customer_relationship_manager",
]


# ==================================================================================================
# (a) finance_sec_search — typed fixed-route case. Zero decorators, zero handler edits; the one
#     addition is the toolless-catch-all declaration (its /{tool_name} route backs no tools).
# ==================================================================================================


def build_finance(expose: bool = True) -> tuple[FinanceAgentResourcesServer, FastAPI]:
    cache_dir = tempfile.mkdtemp(prefix="spike_finance_cache_")
    (Path(cache_dir) / "tickers.json").write_text(json.dumps(MOCK_TICKERS))  # skip SEC.gov download
    config = FinanceAgentResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="",
        name="finance_sec_search_spike",
        cache_dir=cache_dir,
        judge_prompt_template="{question} {expected_answer} {generated_answer}",
        retrieval_system_prompt="You answer questions from stored documents.",
        # Truthy ref so retrieve_information proceeds to its storage lookup; the storage-error
        # paths return before any LLM call, so the mock ServerClient is never used.
        retrieval_model_server=ModelServerRef(type="responses_api_models", name="spike_fake_model"),
    )
    server = SpikeFinanceResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    app = server.setup_webserver()
    if expose:
        maybe_auto_expose(server, app)
    return server, app


class SpikeFinanceResourcesServer(FinanceAgentResourcesServer):
    expose_tools_over_mcp: ClassVar[bool] = True
    # finance's /{tool_name} catch-all only returns error strings for unknown tool names — it
    # backs no tools. Declaring that silences the missing-inventory warning (author knowledge
    # the harvest cannot recover). One attribute, zero handler edits. ClassVar because Gym
    # servers are pydantic models (a bare attribute would be rejected as an unannotated field).
    mcp_toolless_catchall_paths: ClassVar[frozenset[str]] = frozenset({"/{tool_name}"})


# Same invariant as SpikeWorkbench below: in production the declaration lives on the real class in
# its own app.py, so the class NAME (which seeds get_session_middleware_key) must not change here.
SpikeFinanceResourcesServer.__name__ = "FinanceAgentResourcesServer"


# ==================================================================================================
# (b) workplace_assistant — dispatcher case (one catch-all route, 27 tools). Brian's 2.d: the
#     server overrides ONE function returning the tool inventory; calls route through the
#     existing catch-all. Handlers untouched.
# ==================================================================================================


class SpikeWorkbenchResourcesServer(WorkbenchResourcesServer):
    expose_tools_over_mcp: ClassVar[bool] = True

    def mcp_tool_inventory(self) -> list[dict]:
        schemas = get_tools(WORKBENCH_TOOLKITS)["schemas"]
        return [
            {"name": s["name"], "input_schema": s["parameters"], "description": s.get("description")} for s in schemas
        ]


# In production Brian's design puts mcp_tool_inventory() directly on WorkbenchResourcesServer in
# its own app.py — the class NAME (which seeds get_session_middleware_key) would not change. Keep
# that invariant here so the exposed and plain instances share the same session secret/cookie name.
SpikeWorkbenchResourcesServer.__name__ = "WorkbenchResourcesServer"


def build_workplace(expose: bool = True) -> tuple[WorkbenchResourcesServer, FastAPI]:
    config = WorkbenchResourcesServerConfig(
        host="127.0.0.1", port=8080, entrypoint="", name="workplace_assistant_spike"
    )
    cls = SpikeWorkbenchResourcesServer if expose else WorkbenchResourcesServer
    server = cls(config=config, server_client=MagicMock(spec=ServerClient))
    app = server.setup_webserver()
    if expose:
        maybe_auto_expose(server, app)
    return server, app


# ==================================================================================================
# (c) aviary — plumbing-exposed case (/step + /close typed with env_id). The abstract origin/main
#     server needs a concrete dataset; DummyEnv (from the aviary library itself) keeps the spike
#     network-free. Handler code untouched.
# ==================================================================================================


if AVIARY_AVAILABLE:

    class DummyTaskDataset(TaskDataset):
        def get_new_env_by_idx(self, idx: int) -> DummyEnv:
            # end_immediately=False keeps episodes alive across multiple /step calls.
            return DummyEnv(task=f"dummy-task-{idx}", end_immediately=False)

        def __len__(self) -> int:
            return 1000

    class SpikeAviaryResourcesServer(AviaryResourcesServer[DummyEnv, DummyTaskDataset]):
        expose_tools_over_mcp: ClassVar[bool] = True
        dataset: DummyTaskDataset = Field(default_factory=DummyTaskDataset)

    def build_aviary(expose: bool = True):
        config = AviaryResourcesServerConfig(host="127.0.0.1", port=8080, entrypoint="", name="aviary_spike")
        server = SpikeAviaryResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        if expose:
            maybe_auto_expose(server, app)
        return server, app

else:

    def build_aviary(expose: bool = True):
        raise RuntimeError("aviary case requires the optional fhaviary package (pip install fhaviary)")
