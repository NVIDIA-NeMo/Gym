"""Pin-move guard for the hermes_agent wrapper.

The wrapper in ``responses_api_agents/hermes_agent/app.py`` constructs and drives
hermes-agent's ``AIAgent`` directly. When the hermes-agent pin moves, a removed or
renamed constructor kwarg (or a dropped method) turns a full eval into a *silent*
0-resolved run rather than an error. These tests inspect the pinned build's API in
milliseconds so a break shows up in CI instead of after an 89-task rollout.

They skip when hermes-agent is not installed in the test env (e.g. a gym-only checkout);
they only assert when the pinned package is present.
"""
import inspect

import pytest

run_agent = pytest.importorskip("run_agent")  # hermes-agent exposes AIAgent here

# The kwargs the gym wrapper passes to AIAgent(...). Keep in sync with app.py.
# Note: reasoning_config / request_overrides are the *remapped* replacements for the
# old insert_reasoning / temperature; if a future pin drops them the wrapper's
# signature guard would silently discard them (reasoning/sampling lost), which is
# exactly what these assertions are here to catch.
WRAPPER_KWARGS = {
    "base_url",
    "api_key",
    "model",
    "reasoning_config",
    "request_overrides",
    "max_iterations",
    "enabled_toolsets",
    "disabled_toolsets",
    "quiet_mode",
    "skip_context_files",
    "skip_memory",
    "save_trajectories",
}


def _init_params():
    return inspect.signature(run_agent.AIAgent.__init__).parameters


def test_aiagent_accepts_all_wrapper_kwargs():
    params = _init_params()
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    missing = WRAPPER_KWARGS - set(params)
    assert has_var_kw or not missing, (
        f"pinned hermes-agent AIAgent no longer accepts {sorted(missing)}; update the "
        "hermes_agent wrapper (responses_api_agents/hermes_agent/app.py) to match."
    )


def test_aiagent_run_conversation_present():
    assert hasattr(run_agent.AIAgent, "run_conversation"), (
        "pinned hermes-agent AIAgent lost run_conversation; the wrapper's run path is broken."
    )


def test_aiagent_build_api_kwargs_present():
    assert hasattr(run_agent.AIAgent, "_build_api_kwargs"), (
        "pinned hermes-agent dropped _build_api_kwargs; the wrapper's enable-thinking patch "
        "(chat_template_kwargs) no longer has a hook."
    )
