#!/usr/bin/env bash
# clean.sh — Remove source code from a DeNovoSWE image while preserving
# the installed environment and essential config files.
#
# Usage: bash clean.sh <workdir>
#   e.g. bash clean.sh /workspace/pep8

set -euo pipefail

WORKDIR="${1:?Usage: bash clean.sh <workdir>}"
cd "$WORKDIR"

# ── 0. Anti-peek setup — discover all Python interpreters + helpers ──────────
# Multi-Python images (system Python + conda env + venvs) used to slip
# through because the original ``pip uninstall`` only targeted whichever
# ``pip`` happened to be on PATH.  Now we run uninstall against EVERY
# Python interpreter we can find, and follow up with physical
# ``rm -rf`` of any site-packages / dist-packages residue, plus a wipe
# of pip's wheel cache (otherwise the agent can
# ``unzip ~/.cache/pip/wheels/<pkg>*.whl`` to recover source).

# Collect every python binary on PATH + standard install roots + conda
# env bins.  Deduped, only executable entries kept.
_discover_pythons() {
    {
        compgen -c 2>/dev/null \
            | grep -E '^python(2|3)?(\.[0-9]+)?$' \
            | while read -r c; do command -v "$c" 2>/dev/null; done
        ls /usr/bin/python* /usr/local/bin/python* 2>/dev/null
        ls /opt/*/bin/python /opt/conda/envs/*/bin/python \
           /opt/miniconda*/envs/*/bin/python 2>/dev/null
    } 2>/dev/null | sort -u | while read -r p; do
        # Filter out non-interpreter shims like ``python-config``,
        # ``python3.7m-config``, ``python-argcomplete-*`` that ``ls
        # /usr/bin/python*`` will pull in but cannot run ``-m pip``.
        local base
        base=$(basename "$p")
        case "$base" in
            python|python[0-9]|python[0-9].[0-9]|python[0-9].[0-9][0-9]|\
            python[0-9]m|python[0-9].[0-9]m|python[0-9].[0-9][0-9]m)
                ;;
            *)
                continue
                ;;
        esac
        [ -x "$p" ] || continue
        # Verify it actually behaves like a Python interpreter — some
        # ``python`` paths are setuptools entry-point shims that crash
        # on ``-c "pass"``.  ``timeout 5`` so a misbehaving shim
        # (sitecustomize that pings a remote, etc.) doesn't stall
        # the whole clean step.
        timeout 5 "$p" -c "pass" >/dev/null 2>&1 || continue
        echo "$p"
    done
    return 0
}
# ``|| true`` so we never blow up clean.sh on a multi-Python-discovery
# hiccup; an empty ``_PYTHON_BINS`` is fine — the strategy-side bare
# ``pip`` fallback still does the basic work.
_PYTHON_BINS=$(_discover_pythons || true)

# Generate name variants (handles dash/underscore/dot drift) used by
# the purge helpers below.  Echoes one variant per line.
_name_variants() {
    local n="$1"
    [ -z "$n" ] && return
    printf '%s\n%s\n%s\n%s\n%s\n' \
        "$n" "${n//-/_}" "${n//_/-}" "${n//./_}" "${n//./-}" \
        | sort -u
}

# Accumulator: every package name we successfully resolve via the
# strategies below is appended here so the post-strategy purge pass
# can target each.  Set -u tolerates empty arrays from bash 4.4+.
_RESOLVED_PKG_NAMES=()

# ── 1. Uninstall locally-installed packages (editable or regular) ────────────
# Try to find the package name from setup.py / pyproject.toml / setup.cfg
_try_uninstall() {
    local pkg_name="$1"
    [ -z "$pkg_name" ] && return
    # Track so the post-strategy purge functions can find residue
    # (dist-info / __editable__ shims / apt-installed counterparts).
    _RESOLVED_PKG_NAMES+=("$pkg_name")
    # Uninstall from EVERY interpreter; agent peeking commonly targets
    # an interpreter different from the one on PATH (e.g. conda env).
    for py in $_PYTHON_BINS; do
        "$py" -m pip uninstall -y "$pkg_name" 2>/dev/null || true
    done
    # Fall back to the bare ``pip`` for the (rare) case where no
    # discovered interpreter has pip but ``pip`` itself exists.
    pip uninstall -y "$pkg_name" 2>/dev/null || true
}

# Strategy A: parse package name from setup.py
if [ -f setup.py ]; then
    pkg=$(python3 -c "
import ast, sys
try:
    with open('setup.py') as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and hasattr(node, 'keywords'):
            for kw in node.keywords:
                if kw.arg == 'name' and isinstance(kw.value, ast.Constant):
                    print(kw.value.value)
                    sys.exit(0)
except Exception:
    pass
" 2>/dev/null || true)
    _try_uninstall "$pkg"
fi

# Strategy B: parse from pyproject.toml
if [ -f pyproject.toml ]; then
    pkg=$(python3 -c "
import re, sys
try:
    with open('pyproject.toml') as f:
        content = f.read()
    m = re.search(r'^\s*name\s*=\s*[\"\\']([^\"\\']+)[\"\\']', content, re.MULTILINE)
    if m:
        print(m.group(1))
except Exception:
    pass
" 2>/dev/null || true)
    _try_uninstall "$pkg"
fi

# Strategy C: parse from setup.cfg
if [ -f setup.cfg ]; then
    pkg=$(python3 -c "
import configparser, sys
try:
    c = configparser.ConfigParser()
    c.read('setup.cfg')
    print(c.get('metadata', 'name', fallback=''))
except Exception:
    pass
" 2>/dev/null || true)
    _try_uninstall "$pkg"
fi

# Strategy D: try uninstalling by directory name (common convention)
dir_name=$(basename "$WORKDIR")
_try_uninstall "$dir_name"

# Strategy E: brute-force — find all .egg-link / .dist-info that point here
for egg_link in $(find "$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || echo /dev/null)" -name "*.egg-link" 2>/dev/null); do
    if grep -q "$WORKDIR" "$egg_link" 2>/dev/null; then
        pkg_name=$(basename "$egg_link" .egg-link)
        _try_uninstall "$pkg_name"
    fi
done


# ── 1.1. Anti-peek purge ────────────────────────────────────────────────────
# Even after a successful ``pip uninstall``, a few residues commonly
# survive:
#   * compiled extension dirs that pip skipped because they were not
#     listed in RECORD,
#   * ``__editable__<pkg>_*_finder.py`` + ``.pth`` shim left by recent
#     setuptools editable installs,
#   * dist-info / egg-info directories from concurrent installs of the
#     same package via different toolchains (poetry + pip + apt),
#   * the entire pip wheel cache at ``~/.cache/pip/wheels``, which an
#     agent could unzip to recover the source we just removed,
#   * apt-installed ``python3-<pkg>`` Debian packages parked in
#     ``/usr/lib/python3/dist-packages`` — pip uninstall has no idea
#     these exist.
#
# Each helper is best-effort: failures are swallowed so a single
# missing tool (no apt, read-only filesystem, etc.) doesn't take
# clean.sh down with it.

_purge_site_packages_traces() {
    local pkg_name="$1"
    [ -z "$pkg_name" ] && return
    # All site-packages / dist-packages roots reachable from the
    # common install prefixes.  ``maxdepth 8`` covers conda envs,
    # virtualenvs, system Python, /opt vendor installs.
    # ``|| true`` is essential — ``find`` returns non-zero whenever
    # it hits a permission-denied or vanished-during-walk directory,
    # which under ``set -euo pipefail`` would kill clean.sh.
    local sp_dirs
    sp_dirs=$(find /usr /opt /root /home -maxdepth 8 -type d \
        \( -name 'site-packages' -o -name 'dist-packages' \) \
        2>/dev/null | sort -u || true)
    local variants
    variants=$(_name_variants "$pkg_name" || true)
    for sp in $sp_dirs; do
        for v in $variants; do
            # Package directory itself (covers the common case).
            [ -d "$sp/$v" ] && rm -rf "$sp/$v"
            # Metadata directories from any version + toolchain.
            find "$sp" -maxdepth 1 \( \
                -name "${v}-*.dist-info" -o -name "${v}-*.egg-info" \
            \) -exec rm -rf {} + 2>/dev/null || true
            # Editable install shims (``__editable__<pkg>_*.py``,
            # ``<pkg>.pth``, ``<pkg>.egg-link``).
            find "$sp" -maxdepth 1 \( \
                -name "__editable__*${v}*" \
                -o -name "${v}*.pth" \
                -o -name "${v}.egg-link" \
            \) -delete 2>/dev/null || true
        done
    done
}

_purge_pip_caches() {
    # Standard locations.  ``${HOME:-/root}`` because ``set -u`` would
    # otherwise crash on images with HOME unset (rare but seen).
    rm -rf "${HOME:-/root}/.cache/pip" /root/.cache/pip /tmp/pip-* \
        2>/dev/null || true
    for h in /home/*; do
        [ -d "$h/.cache/pip" ] && rm -rf "$h/.cache/pip" 2>/dev/null
    done
    # Some pip configs put the cache elsewhere — ``pip cache purge``
    # respects per-config locations.  Run for every interpreter so
    # multi-Python images get their caches scrubbed too.
    for py in $_PYTHON_BINS; do
        "$py" -m pip cache purge 2>/dev/null || true
    done
}

# Purge leftover build-temp / install-trace directories under /tmp.
# Real bite: SWIG-generated wrappers from a previous ``pip install``
# survive in ``/tmp/tmp<rand>.build-temp_<pkg>.<pkg>/api/python3/<pkg>.py``
# and let an agent ``cat`` the entire Python binding — observed
# inflating ``biojppm_rapidyaml_pr551`` from ~0 to 1.0.  scikit-build,
# cmake-build-extension, maturin, cffi, cython, pep517 all leak
# similarly-named scratch dirs.  ``find -maxdepth 1`` keeps the wipe
# bounded to /tmp top-level so we don't recurse into the agent's own
# /tmp/<run>/ working tree.
_purge_build_temp_dirs() {
    find /tmp -maxdepth 1 \( \
        -name "*.build-temp*" \
        -o -name "tmp*.build-temp*" \
        -o -name "*.build-lib*" \
        -o -name "*scikit_build*" \
        -o -name "pip-build-env-*" \
        -o -name "pip-wheel-*" \
        -o -name "pip-req-build-*" \
        -o -name "pip-modern-metadata-*" \
        -o -name "pip-install-*" \
        -o -name "*cmake_build_extension*" \
        -o -name "tmp*.egg-info" \
        -o -name "tmp*.dist-info" \
        -o -name "tmp*_build_dir*" \
        -o -name "tmp*-cython-*" \
        -o -name "tmp*-maturin-*" \
    \) -exec rm -rf {} + 2>/dev/null || true
    # Also nuke ``.so`` / ``.pyd`` / ``.dylib`` extension binaries +
    # generated ``.py`` wrappers sitting directly under /tmp or in
    # tmp* dirs at /tmp root — those are the actual source-leak files
    # rather than the wrapper dir itself.
    find /tmp -maxdepth 3 -type f \( \
        -name "*.so" -o -name "*.pyd" -o -name "*.dylib" \
        -o -name "_*.so" \
    \) -path "/tmp/tmp*" -delete 2>/dev/null || true
}

_purge_apt_python_packages() {
    local pkg_name="$1"
    [ -z "$pkg_name" ] && return
    command -v dpkg-query >/dev/null 2>&1 || return
    command -v apt-get   >/dev/null 2>&1 || return
    local variants
    variants=$(_name_variants "$pkg_name" || true)
    for v in $variants; do
        # Match exact ``python3-<v>`` (or ``python-<v>``) and the
        # ``python3-<v>-anything`` extension/subpackages.  Skip if
        # there's no match.  ``grep`` returning exit 1 (no matches)
        # is the common case — wrap with ``|| true`` so pipefail
        # doesn't kill the script.
        { dpkg-query -W -f='${binary:Package}\n' 2>/dev/null \
            | grep -E "^python3?-${v}(\$|-)" \
            || true; } \
            | while read -r p; do
                # ``--no-auto-remove`` avoids pulling unrelated
                # transitively-installed deps along.
                apt-get remove -y --no-auto-remove "$p" 2>/dev/null || true
            done
    done
}

# Run the purges for every resolved package name (strategies A-E may
# resolve different aliases for the same dist, e.g. ``foo-bar`` vs
# ``foo_bar`` — purge handles each independently).
if [ "${#_RESOLVED_PKG_NAMES[@]}" -gt 0 ]; then
    # Dedupe before iterating to keep the log noise down.
    while read -r _n; do
        [ -z "$_n" ] && continue
        _purge_site_packages_traces "$_n"
        _purge_apt_python_packages "$_n"
    done < <(printf '%s\n' "${_RESOLVED_PKG_NAMES[@]}" | sort -u)
fi
# Cache wipe is global — only run once.
_purge_pip_caches
# Same for /tmp build-temp leaks.
_purge_build_temp_dirs

# ── 1.5. Remove ALL test directories and test files ────────────────────────
# Test directories are removed entirely (including fixtures, conftest, etc.)
# These will be restored from test_patch during evaluation.
# Case-insensitive: matches test, Test, TEST, Tests, TESTS, etc.
find "$WORKDIR" -type d | while read -r d; do
    dirname_lower=$(basename "$d" | tr '[:upper:]' '[:lower:]')
    case "$dirname_lower" in
        test|tests|testsuite|testsuites|testing|test_suite)
            rm -rf "$d"
            ;;
    esac
done

# Remove individual test files outside test directories (case-insensitive)
find "$WORKDIR" -type f \( \
    -iname "test_*.py" -o -iname "*_test.py" -o -iname "*_tests.py" \
    -o -iname "conftest.py" \
\) -delete 2>/dev/null || true

# Remove test config files (will be restored by test_patch if needed)
find "$WORKDIR" -type f -name ".coveragerc" -delete 2>/dev/null || true

# ── 2. Remove source code files (*.py, *.pyx, *.pxd, etc.) ──────────────────
# Preserve essential config/build files
_PRESERVE_PATTERN='(pyproject\.toml|setup\.py|setup\.cfg|requirements.*\.txt|MANIFEST\.in|Dockerfile|\.python-version|Makefile|poetry\.lock|Pipfile\.lock|pdm\.lock|uv\.lock|\.gitignore|\.gitattributes|LICENSE.*|LICENCE.*)$'

# Remove all Python source files except preserved ones
find "$WORKDIR" -type f \( \
    -name "*.py" -o -name "*.pyx" -o -name "*.pxd" -o -name "*.pyi" \
\) | grep -vE "$_PRESERVE_PATTERN" | while read -r f; do
    rm -f "$f"
done

# ── 3. Remove caches, build artifacts, and generated files ───────────────────
find "$WORKDIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name ".tox" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name ".nox" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name ".hypothesis" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name ".venv" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name "venv" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name ".direnv" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name ".vscode" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name ".idea" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true

# Source-leak channels not covered by the cache-cleanup above.
# Even though step 2 deleted all *.py/*.pyx/*.pxd/*.pyi sources, these
# parallel channels carry the same information:
#   * ``.eggs/`` holds full source of any easy_install-style package.
#   * ``vendor`` / ``vendored`` / ``third_party`` / ``contrib`` hold
#     upstream snapshots of the project being rebuilt.
#   * ``target`` / ``crates`` / ``.cargo`` + ``*.rs`` / ``*.rlib`` /
#     ``*.rmeta`` — Rust source backs the Python wrapper for PyO3
#     projects; *.py was already wiped but the Rust impl leaked.
#   * ``*.snap`` / ``*.expect`` / ``__snapshots__/`` — pytest-insta /
#     insta golden outputs pin removed code's behaviour exactly.
#   * ``examples`` / ``samples`` / ``demo`` — frequently import + call
#     the symbols the agent is asked to rebuild.
# Note: Cargo.toml is preserved on purpose; see denovo_swe_subdoc/partial_clean.sh.
find "$WORKDIR" -type d \( -name ".eggs" -o -name "vendor" -o -name "vendored" \
    -o -name "third_party" -o -name "contrib" \) -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d \( -name "target" -o -name "crates" -o -name ".cargo" \) \
    -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type f \( -name "Cargo.lock" \
    -o -name "*.rs" -o -name "*.rlib" -o -name "*.rmeta" \
    -o -name "*.snap" -o -name "*.expect" \) -delete 2>/dev/null || true
find "$WORKDIR" -type d \( -name "examples" -o -name "example" \
    -o -name "samples" -o -name "sample" -o -name "event_samples" \
    -o -name "demo" -o -name "demos" -o -name "aws_doc_sdk_examples_tools" \) \
    -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -type d -name "__snapshots__" -exec rm -rf {} + 2>/dev/null || true

# Remove specific file patterns
find "$WORKDIR" -type f \( \
    -name "*.pyc" -o -name "*.pyo" \
    -o -name "*.so" -o -name "*.pyd" -o -name "*.dll" -o -name "*.dylib" \
    -o -name "*.egg" -o -name "*.whl" \
    -o -name ".coverage" -o -name "coverage.xml" -o -name "coverage.json" \
    -o -name ".DS_Store" \
    -o -name "*.min.js" -o -name "*.min.css" \
    -o -name ".env" -o -name ".envrc" \
\) -delete 2>/dev/null || true

# Remove JSON files (may contain cached data / source maps)
find "$WORKDIR" -type f -name "*.json" -delete 2>/dev/null || true

# Remove generated docs HTML
find "$WORKDIR" -type d -name "_build" -exec rm -rf {} + 2>/dev/null || true
find "$WORKDIR" -path "*/docs/_build" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove log / trace / artifact files
find "$WORKDIR" -type f \( -name "*.log" -o -name "*.trace" \) -delete 2>/dev/null || true

# Remove Jupyter notebooks (may contain source code)
find "$WORKDIR" -type f -name "*.ipynb" -delete 2>/dev/null || true

# Remove RST/MD doc files (may contain code examples that leak implementation)
find "$WORKDIR" -type f \( -name "*.rst" -o -name "*.md" \) -delete 2>/dev/null || true

# Remove YAML/TOML files that aren't config (e.g., CI configs may have code snippets)
# But preserve essential ones
find "$WORKDIR" -type f \( -name "*.yml" -o -name "*.yaml" \) | while read -r f; do
    basename_f=$(basename "$f")
    if [[ "$basename_f" =~ ^(pyproject\.toml|tox\.ini|pytest\.ini|setup\.cfg)$ ]]; then
        continue
    fi
    rm -f "$f"
done

# ── 4. Clean git history to prevent source code recovery ─────────────────────
# Completely rebuild .git to guarantee no old objects survive.
# `git gc --prune=now` is not 100% reliable — loose objects or packfiles
# can retain old source code. Reinitializing is the only safe way.
cd "$WORKDIR"
if [ -d .git ]; then
    rm -rf .git

    git init 2>/dev/null
    git config user.email "clean@denovoswe.local"
    git config user.name "DeNovoSWE-Clean"
    git add -A 2>/dev/null || true
    git commit --allow-empty -m "cleaned workspace" 2>/dev/null || true
fi

# ── 5. Remove empty directories left behind ──────────────────────────────────
find "$WORKDIR" -type d -empty -delete 2>/dev/null || true

echo "DeNovoSWE clean completed for $WORKDIR"
