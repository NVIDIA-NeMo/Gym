# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from responses_api_agents.swe_agents.openclaw.patch_openclaw import (
    PATCHES,
    SENTINEL,
    SENTINEL_SURFACE_TOOL,
    apply_patch_to_text,
    patch_dist,
)


# Minified-bundle excerpts containing each real marker line surrounded by neighbouring code, so
# the patches are exercised against realistic shapes. A single BUNDLE holds all three markers
# because patch_dist requires every patch's marker to be present somewhere (else it fails loud).
_GUARD_SNIPPET = (
    "function installToolResultContextGuard(params) {\n"
    "\tconst contextWindowTokens = Math.max(1, Math.floor(params.contextWindowTokens));\n"
    "\tconst maxContextChars = Math.max(1024, Math.floor(contextWindowTokens * 4 * PREEMPTIVE_OVERFLOW_RATIO));\n"
    "\tconst maxSingleToolResultChars = Math.max(1024, Math.floor(contextWindowTokens * 2 * 0.5));\n"
    "\tif (exceedsPreemptiveOverflowThreshold({ messages, maxContextChars })) "
    "throw new Error(PREEMPTIVE_CONTEXT_OVERFLOW_MESSAGE);\n"
    "}\n"
)
_EXECUTOR_SNIPPET = (
    "async function compactEmbeddedPiSessionDirect(params) {\n"
    "\tif (hasExplicitCompactionModel(params)) return await compactEmbeddedPiSessionDirectOnce(params);\n"
    "\treturn { ok: true, compacted: true };\n"
    "}\n"
    "async function compactEmbeddedPiSessionDirectOnce(params) {\n"
    "\tconst trigger = params.trigger ?? 'manual';\n"
    "}\n"
)
_HARNESS_SNIPPET = (
    "async function maybeCompactAgentHarnessSession(params) {\n"
    "\tconst harness = selectAgentHarness(params);\n"
    "\treturn harness.compact(params);\n"
    "}\n"
)
# Fix #4 markers, in their real shapes. RECORD lives in tool-call-id-*.js (isAllowedToolCallName);
# REPLAY lives in selection-*.js (resolveReplayToolCallName). Both keep the structural checks above
# the patched return line.
_RECORD_SNIPPET = (
    "function isAllowedToolCallName(name, allowedToolNames) {\n"
    '\tif (typeof name !== "string") return false;\n'
    "\tconst trimmed = name.trim();\n"
    "\tif (!trimmed) return false;\n"
    "\tif (trimmed.length > TOOL_CALL_NAME_MAX_CHARS || !TOOL_CALL_NAME_RE.test(trimmed)) return false;\n"
    "\tif (!allowedToolNames) return true;\n"
    "\treturn allowedToolNames.has(normalizeLowercaseStringOrEmpty(trimmed));\n"
    "}\n"
)
_REPLAY_SNIPPET = (
    "function resolveReplayToolCallName(rawName, rawId, allowedToolNames) {\n"
    "\tif (rawName.length > REPLAY_TOOL_CALL_NAME_MAX_CHARS * 2) return null;\n"
    "\tconst trimmed = normalizeToolCallNameForDispatch(rawName, allowedToolNames, rawId).trim();\n"
    "\tif (!trimmed || trimmed.length > REPLAY_TOOL_CALL_NAME_MAX_CHARS || /\\s/.test(trimmed)) return null;\n"
    "\tif (!allowedToolNames || allowedToolNames.size === 0) return trimmed;\n"
    "\treturn resolveExactAllowedToolName(trimmed, allowedToolNames);\n"
    "}\n"
)
_BUNDLE = _GUARD_SNIPPET + _EXECUTOR_SNIPPET + _HARNESS_SNIPPET + _RECORD_SNIPPET + _REPLAY_SNIPPET

_PATCH_BY_NAME = {p["name"]: p for p in PATCHES}
_SNIPPET_BY_NAME = {
    "preemptive-overflow-guard": _GUARD_SNIPPET,
    "compaction-executor": _EXECUTOR_SNIPPET,
    "harness-compaction-entry": _HARNESS_SNIPPET,
    "surface-unknown-tool-record": _RECORD_SNIPPET,
    "surface-unknown-tool-replay": _REPLAY_SNIPPET,
}
_COMPACTION_NAMES = {
    "preemptive-overflow-guard",
    "compaction-executor",
    "harness-compaction-entry",
}
_SURFACE_TOOL_NAMES = {"surface-unknown-tool-record", "surface-unknown-tool-replay"}


def _write_bundle(dist_dir, text, name="selection-ei714fjJ.js"):
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / name).write_text(text, encoding="utf-8")
    return dist_dir / name


def test_all_patches_registered():
    assert {p["name"] for p in PATCHES} == _COMPACTION_NAMES | _SURFACE_TOOL_NAMES


@pytest.mark.parametrize("name", list(_PATCH_BY_NAME))
def test_apply_patch_to_text_applies_once(name):
    patch = _PATCH_BY_NAME[name]
    out, status = apply_patch_to_text(_SNIPPET_BY_NAME[name], patch)
    assert status == "applied"
    assert patch["replace"] in out  # the marker is rewritten to the sentinel-bearing replacement
    expected_sentinel = SENTINEL if name in _COMPACTION_NAMES else SENTINEL_SURFACE_TOOL
    assert expected_sentinel in out


def test_executor_patches_emit_a_greppable_suppression_signal():
    # Both executor no-ops log an explicit, greppable line so training can count context-pressure
    # events. The guard patch must NOT log (it is silenced upstream, by design).
    for name in ("compaction-executor", "harness-compaction-entry"):
        assert "auto-compaction suppressed" in _PATCH_BY_NAME[name]["replace"]
        assert "console.error" in _PATCH_BY_NAME[name]["replace"]
    assert "console.error" not in _PATCH_BY_NAME["preemptive-overflow-guard"]["replace"]


def test_surface_record_patch_drops_only_the_allowlist_gate():
    # isAllowedToolCallName: the final allowlist-membership return becomes `return true`, so any
    # structurally-valid name is kept -- but the format/length checks ABOVE it are untouched.
    patch = _PATCH_BY_NAME["surface-unknown-tool-record"]
    out, _ = apply_patch_to_text(_RECORD_SNIPPET, patch)
    assert "allowedToolNames.has(" not in out  # the membership gate is gone
    assert "return true;" + SENTINEL_SURFACE_TOOL in out
    # structural validation preserved (a malformed/oversized name is still rejected upstream)
    assert "TOOL_CALL_NAME_RE.test(trimmed)" in out
    assert "TOOL_CALL_NAME_MAX_CHARS" in out


def test_surface_replay_patch_falls_back_to_the_validated_name():
    # resolveReplayToolCallName: unknown names fall back to `?? trimmed` instead of dropping;
    # near-matches still canonicalize via resolveExactAllowedToolName, and the
    # length/whitespace guards above are preserved.
    patch = _PATCH_BY_NAME["surface-unknown-tool-replay"]
    out, _ = apply_patch_to_text(_REPLAY_SNIPPET, patch)
    assert "resolveExactAllowedToolName(trimmed, allowedToolNames) ?? trimmed;" in out
    assert SENTINEL_SURFACE_TOOL in out
    # the pre-checks that still drop genuinely-unusable calls are untouched
    assert "REPLAY_TOOL_CALL_NAME_MAX_CHARS" in out
    assert "/\\s/.test(trimmed)" in out


def test_surface_tool_patches_carry_their_own_sentinel_not_the_compaction_one():
    for name in _SURFACE_TOOL_NAMES:
        replace = _PATCH_BY_NAME[name]["replace"]
        assert SENTINEL_SURFACE_TOOL in replace
        assert SENTINEL not in replace  # never conflate the two patch families


def test_guard_patch_removes_the_overflow_budget_expression():
    # Replacement-style patch (not inject-style): the original throw-budget expression is gone.
    patch = _PATCH_BY_NAME["preemptive-overflow-guard"]
    out, _ = apply_patch_to_text(_GUARD_SNIPPET, patch)
    assert patch["find"] not in out
    assert "Number.POSITIVE_INFINITY" in out


@pytest.mark.parametrize("name", list(_PATCH_BY_NAME))
def test_apply_patch_to_text_idempotent(name):
    patch = _PATCH_BY_NAME[name]
    once, _ = apply_patch_to_text(_SNIPPET_BY_NAME[name], patch)
    twice, status = apply_patch_to_text(once, patch)
    assert status == "already"
    assert twice == once


def test_apply_patch_to_text_missing_marker():
    _, status = apply_patch_to_text("nothing relevant here", PATCHES[0])
    assert status == "missing"


def test_executor_marker_does_not_match_the_once_variant():
    # The executor marker must hit compactEmbeddedPiSessionDirect(params), NOT the longer
    # compactEmbeddedPiSessionDirectOnce(params) that immediately follows it.
    patch = _PATCH_BY_NAME["compaction-executor"]
    assert _EXECUTOR_SNIPPET.count(patch["find"]) == 1


def test_apply_patch_to_text_rejects_ambiguous_marker():
    doubled = _GUARD_SNIPPET + _GUARD_SNIPPET
    with pytest.raises(RuntimeError, match="not unique"):
        apply_patch_to_text(doubled, PATCHES[0])


def test_patch_dist_applies_all_patches(tmp_path):
    bundle = _write_bundle(tmp_path / "dist", _BUNDLE)
    status = patch_dist(tmp_path / "dist")
    assert status == {name: "applied" for name in _COMPACTION_NAMES | _SURFACE_TOOL_NAMES}
    text = bundle.read_text(encoding="utf-8")
    assert "Number.POSITIVE_INFINITY" in text
    assert text.count(SENTINEL) == 3
    assert text.count(SENTINEL_SURFACE_TOOL) == 2
    # the benign per-result truncation budget must be left intact
    assert "maxSingleToolResultChars" in text
    # the no-op executor returns the benign "nothing to compact" shape
    assert "compacted: false" in text


def test_patch_dist_across_multiple_files(tmp_path):
    # Real layout: the guard + harness + replay markers live in selection-*.js, the executor in
    # compact-*.js, the record marker in tool-call-id-*.js. patch_dist must satisfy every patch
    # across the whole dist/*.js set.
    dist = tmp_path / "dist"
    _write_bundle(dist, _GUARD_SNIPPET + _HARNESS_SNIPPET + _REPLAY_SNIPPET, name="selection-ei714fjJ.js")
    _write_bundle(dist, _EXECUTOR_SNIPPET, name="compact-Be1VaHAE.js")
    _write_bundle(dist, _RECORD_SNIPPET, name="tool-call-id-CSvCHqYu.js")
    status = patch_dist(dist)
    assert all(s == "applied" for s in status.values())


def test_patch_dist_ignores_dts_stubs(tmp_path):
    # Marker substrings living in nested *.d.ts stubs must NOT be patched: patch_dist
    # only globs top-level dist/*.js.
    dist = tmp_path / "dist"
    _write_bundle(dist, _BUNDLE)
    stub_dir = dist / "plugin-sdk" / "src"
    stub_dir.mkdir(parents=True)
    stub = stub_dir / "tool-result-context-guard.d.ts"
    stub.write_text(_BUNDLE, encoding="utf-8")
    patch_dist(dist)
    assert SENTINEL not in stub.read_text(encoding="utf-8")


# No markers at all, or only a partial set (e.g. just the guard) -> both must fail loud on the
# same "NOT FOUND" path, so a future OpenClaw bump that drops one marker can never silently leave
# that path enabled.
@pytest.mark.parametrize("contents", ["const unrelated = 1;\n", _GUARD_SNIPPET], ids=["none", "partial"])
def test_patch_dist_missing_marker_fails_loud(tmp_path, contents):
    _write_bundle(tmp_path / "dist", contents)
    with pytest.raises(SystemExit, match="NOT FOUND"):
        patch_dist(tmp_path / "dist")


def test_patch_dist_no_bundles_fails_loud(tmp_path):
    (tmp_path / "dist").mkdir()
    with pytest.raises(SystemExit, match="no dist"):
        patch_dist(tmp_path / "dist")
