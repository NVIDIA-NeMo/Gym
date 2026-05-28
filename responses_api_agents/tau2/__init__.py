# Vendored runtime patch for tau2-bench.
#
# vLLM >= 0.16.0 renamed the assistant-message response field
# `reasoning_content` -> `reasoning`. Upstream tau2-bench (pinned at
# e6e23241) only reads `reasoning_content`, so the cross-turn replay in
# `to_litellm_messages()` silently no-ops on modern vLLM.
#
# Complication: tau2/__init__.py eagerly imports `tau2.agent.llm_agent`
# and `tau2.user.user_simulator`, both of which do
# `from tau2.utils.llm_utils import generate`. That `from … import …`
# binds the *original* function into those modules **before** our patch
# can run. So patching just `tau2.utils.llm_utils.generate` is not
# enough — we also have to rebind every module that already pulled in
# the unpatched reference.

import importlib.util
import sys
from pathlib import Path

_PATCH_PATH = Path(__file__).parent / "_patches" / "llm_utils.py"
if _PATCH_PATH.exists():
    # Trigger normal loading of tau2.utils.llm_utils (and its transitive
    # deps via tau2/__init__.py's eager imports).
    import tau2.utils.llm_utils as _llm_utils

    _original_generate = getattr(_llm_utils, "generate", None)
    _original_to_litellm_messages = getattr(_llm_utils, "to_litellm_messages", None)

    # Load our patched copy under a fresh name so we don't clobber the
    # already-imported upstream module.
    _spec = importlib.util.spec_from_file_location(
        "tau2.utils.llm_utils_patched", _PATCH_PATH
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["tau2.utils.llm_utils_patched"] = _mod
    _spec.loader.exec_module(_mod)

    _patched = {
        "generate": getattr(_mod, "generate", None),
        "to_litellm_messages": getattr(_mod, "to_litellm_messages", None),
    }

    # Rebind on the canonical module so future `from … import …` gets the new one.
    for _name, _new in _patched.items():
        if _new is not None:
            setattr(_llm_utils, _name, _new)

    # Also rebind on every already-imported tau2 module that grabbed a
    # direct reference (via `from tau2.utils.llm_utils import generate`).
    _originals_by_name = {
        "generate": _original_generate,
        "to_litellm_messages": _original_to_litellm_messages,
    }
    for _mod_name, _imported_mod in list(sys.modules.items()):
        if not _mod_name.startswith("tau2."):
            continue
        if _mod_name == "tau2.utils.llm_utils":
            continue
        for _name, _orig in _originals_by_name.items():
            if _orig is None:
                continue
            if getattr(_imported_mod, _name, None) is _orig:
                setattr(_imported_mod, _name, _patched[_name])
