# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency rules from SWE-Together 891d19e (Apache-2.0).

Keep registry parsing and workspace-package discovery aligned with the reference.
Decode URL paths before matching so encoded package names cannot bypass denials.
Model routing is owned separately by Gym; this policy never grants model access.
"""

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version


POLICY_VERSION = 2
ALLOW_GROUPS: dict[str, frozenset[str]] = {
    "catalog": frozenset(
        {
            "models.dev",
            "models.opencode.ai",  # opencode's model catalog (pricing + reasoning variants)
        }
    ),
    "registries": frozenset(
        {
            # Python
            "pypi.org",
            "files.pythonhosted.org",
            # Node
            "registry.npmjs.org",
            "registry.yarnpkg.com",
            # Rust
            "crates.io",
            "static.crates.io",
            "index.crates.io",
            # Debian/Ubuntu + NodeSource (opencode's apt fallback, `apt-get install ripgrep`)
            "archive.ubuntu.com",
            "security.ubuntu.com",
            "ports.ubuntu.com",
            "deb.debian.org",
            "security.debian.org",
            "deb.nodesource.com",
        }
    ),
}

LEAK_VECTORS: dict[str, str] = {
    "github.com": "task source repository",
    "githubusercontent.com": "raw GitHub content",
    "githubassets.com": "GitHub",
    "github.io": "GitHub Pages",
    "ghcr.io": "GitHub container registry",
    "gitlab.com": "source forge",
    "bitbucket.org": "source forge",
    "codeberg.org": "source forge",
    "sourcegraph.com": "code search over GitHub",
    "cdn.jsdelivr.net": "CDN mirror of GitHub repositories",
    "unpkg.com": "CDN mirror of npm/GitHub content",
    "gh-proxy.com": "GitHub proxy",
    "ghproxy.com": "GitHub proxy",
    "ghproxy.net": "GitHub proxy",
    "ghfast.top": "GitHub proxy",
    "ghps.cc": "GitHub proxy",
    "kkgithub.com": "GitHub mirror",
    "fastgit.org": "GitHub mirror",
    "gitclone.com": "GitHub mirror",
    "githubfast.com": "GitHub mirror",
    "r.jina.ai": "read-through web proxy",
    "proxy.golang.org": "Go module proxy serves any GitHub repo verbatim",
    "sum.golang.org": "Go checksum db (module cache is pre-populated)",
    "goproxy.io": "Go module proxy",
    "goproxy.cn": "Go module proxy",
    "huggingface.co": "model hub (task images ship what they need)",
    "hf.co": "model hub",
    "hf-mirror.com": "model hub mirror",
    "cloudflare-dns.com": "DNS-over-HTTPS",
    "dns.google": "DNS-over-HTTPS",
    "dns.quad9.net": "DNS-over-HTTPS",
    "doh.opendns.com": "DNS-over-HTTPS",
    "duckduckgo.com": "web search",
    "bing.com": "web search",
    "google.com": "web search",
    "web.archive.org": "web archive of GitHub pages",
    "archive.org": "web archive",
}

#: Bare IP literals are never proxied: an IP was obtained out-of-band (DoH or a
#: hard-coded GitHub address) and there is no hostname to apply policy to.
_IP_LITERAL = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$|^\[?[0-9a-f:]+\]?$", re.I)
_HOSTNAME = re.compile(r"^(?=.{1,253}$)(?!-)[a-z0-9-]{1,63}(?<!-)(?:\.(?!-)[a-z0-9-]{1,63}(?<!-))*$", re.I)


def leak_vector(host: str) -> str | None:
    """The :data:`LEAK_VECTORS` reason if ``host`` is, or is under, a known leak vector."""
    h = host.lower()
    for vector, why in LEAK_VECTORS.items():
        if h == vector or h.endswith("." + vector):
            return f"{vector} ({why})"
    return None


class TaskAllowError(ValueError):
    pass


def validate_task_allow(hosts: list[str]) -> list[str]:
    """Normalise a task's ``[network] allow`` list; reject leak vectors, IPs, wildcards."""
    out: list[str] = []
    for raw in hosts:
        if not isinstance(raw, str):
            raise TaskAllowError(f"[network] allow entries must be strings: {raw!r}")
        h = raw.strip().rstrip(".").lower()
        if "*" in h or "/" in h or ":" in h:
            raise TaskAllowError(f"[network] allow must list bare hostnames, got {raw!r}")
        if _IP_LITERAL.match(h) or not _HOSTNAME.match(h):
            raise TaskAllowError(f"[network] allow: not a hostname: {raw!r}")
        leak = leak_vector(h)
        if leak:
            raise TaskAllowError(f"[network] allow: {raw!r} is a known leak vector — {leak}")
        out.append(h)
    return sorted(set(out))


def task_allow_from_toml(task_dir: Path) -> list[str]:
    """``[network] allow`` from ``<task_dir>/task.toml`` (empty when absent)."""
    p = Path(task_dir) / "task.toml"
    if not p.is_file():
        return []
    with p.open("rb") as f:
        data = tomllib.load(f)
    hosts = (data.get("network") or {}).get("allow") or []
    if not isinstance(hosts, list):
        raise TaskAllowError(f"{p}: [network] allow must be a list")
    return validate_task_allow(hosts)


def allowed_host(host: str, task_allow: list[str] = ()) -> bool:
    host = host.lower().rstrip(".")
    return bool(
        _HOSTNAME.fullmatch(host)
        and not _IP_LITERAL.fullmatch(host)
        and not leak_vector(host)
        and (host in task_allow or any(host in hosts for hosts in ALLOW_GROUPS.values()))
    )


REGISTRY_HOSTS: dict[str, str] = {
    "registry.npmjs.org": "npm",
    "registry.yarnpkg.com": "npm",
    "pypi.org": "pypi",
    "files.pythonhosted.org": "pypi_files",
    "crates.io": "crates",
    "index.crates.io": "crates_index",
    "static.crates.io": "crates_dl",
}

_PEP503 = re.compile(r"[-_.]+")


def norm_pypi(name: str) -> str:
    return _PEP503.sub("-", name).lower()


def norm_npm(name: str) -> str:
    return name.replace("%2f", "/").replace("%2F", "/").lower()


def norm_crate(name: str) -> str:
    return name.replace("_", "-").lower()


class DenyPackages:
    """The per-trial set of package names that must not come from a registry."""

    def __init__(
        self, npm: list[str] | None = None, pypi: list[str] | None = None, crates: list[str] | None = None
    ) -> None:
        self.npm = frozenset(norm_npm(n) for n in (npm or []) if n)
        self.pypi = frozenset(norm_pypi(n) for n in (pypi or []) if n)
        self.crates = frozenset(norm_crate(n) for n in (crates or []) if n)

    @classmethod
    def from_dict(cls, d: dict | None) -> "DenyPackages":
        d = d or {}
        return cls(d.get("npm"), d.get("pypi"), d.get("crates"))

    def to_dict(self) -> dict:
        return {"npm": sorted(self.npm), "pypi": sorted(self.pypi), "crates": sorted(self.crates)}

    def __bool__(self) -> bool:
        return bool(self.npm or self.pypi or self.crates)


_NPM_TARBALL = re.compile(r"^/(@[^/]+/)?([^/]+)/-/", re.I)
_PYPI_FILE = re.compile(r"^/packages/[^/]+/[^/]+/[^/]+/([^/]+)$")


def _pypi_distribution_projects(filename: str) -> set[str]:
    """Parse wheel tags and legacy archive name/version boundaries.

    Legacy versions can contain hyphens too. Keep every valid interpretation so
    an ambiguous source filename cannot hide a denylisted project.
    """
    filename = filename.removesuffix(".metadata")
    if filename.endswith(".whl"):
        try:
            name, _, _, _ = parse_wheel_filename(filename)
            return {norm_pypi(name)}
        except InvalidWheelFilename:
            # Preserve the reference policy's short name-version.whl spelling.
            if filename.count("-") != 1:
                return set()
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tar.Z", ".tgz", ".tar", ".zip", ".whl"):
        if filename.endswith(suffix):
            stem = filename[: -len(suffix)]
            break
    else:
        return set()
    projects = set()
    for boundary in re.finditer("-", stem):
        name, version = stem[: boundary.start()], stem[boundary.end() :]
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", name):
            continue
        try:
            Version(version)
        except InvalidVersion:
            continue
        projects.add(norm_pypi(name))
    return projects


def registry_decision(host: str, method: str, path: str, deny: DenyPackages) -> str | None:
    """A ``deny:`` reason for a registry request, or None when it may proceed."""
    kind = REGISTRY_HOSTS.get(host.lower())
    if kind is None:
        return None
    raw_path, _, query = path.partition("?")
    p = unquote(raw_path)
    if kind == "npm":
        if p.startswith("/-/v1/search") or p.startswith("/-/all") or p.startswith("/search"):
            return "deny:npm registry search"
        segs = [s for s in p.split("/") if s]
        name = None
        if segs:
            if segs[0] == "-" and len(segs) >= 3 and segs[1] == "package":
                name = segs[2]
            elif segs[0].startswith("@") and len(segs) >= 2 and "/" not in segs[0] and "%2f" not in segs[0].lower():
                name = f"{segs[0]}/{segs[1]}"
            else:
                name = segs[0]
        if name and norm_npm(name) in deny.npm:
            return f"deny:task's own npm package {norm_npm(name)}"
        m = _NPM_TARBALL.match(p)
        if m and norm_npm((m.group(1) or "") + m.group(2)) in deny.npm:
            return f"deny:task's own npm package {norm_npm((m.group(1) or '') + m.group(2))}"
        return None
    if kind == "pypi":
        if p.startswith("/search") or (p == "/" and "q=" in query):
            return "deny:pypi search"
        segs = [s for s in p.split("/") if s]
        if len(segs) >= 2 and segs[0] in ("simple", "pypi", "project", "p"):
            if norm_pypi(segs[1]) in deny.pypi:
                return f"deny:task's own PyPI project {norm_pypi(segs[1])}"
        return None
    if kind == "pypi_files":
        m = _PYPI_FILE.match(p)
        if m:
            projects = _pypi_distribution_projects(m.group(1))
            if not projects:
                return "deny:unrecognized PyPI distribution filename"
            if denied := projects & deny.pypi:
                return f"deny:task's own PyPI distribution {min(denied)}"
        return None
    if kind in ("crates", "crates_index", "crates_dl"):
        if kind == "crates" and (
            p.startswith("/api/v1/crates") and ("q=" in query or p.rstrip("/") == "/api/v1/crates")
        ):
            return "deny:crates.io search"
        segs = [s for s in p.split("/") if s]
        name = None
        if kind == "crates_dl" and len(segs) >= 2 and segs[0] == "crates":
            name = segs[1]
        elif kind == "crates" and len(segs) >= 4 and segs[:3] == ["api", "v1", "crates"]:
            name = segs[3]
        elif kind == "crates_index" and segs and segs[-1] != "config.json":
            name = segs[-1]
        if name and norm_crate(name) in deny.crates:
            return f"deny:task's own crate {norm_crate(name)}"
        return None
    return None


#: Shell run inside the container at trial start to list the workspace's own
#: package names. POSIX sh + find/sed/tr only (several task images have no python3).
#: Output: one line per ``kind<TAB>name``; the host turns it into the denylist.
_PKGSCAN_TEMPLATE = r"""set +e
ROOT=__ROOT__
find "$ROOT" -maxdepth 7 \( -name node_modules -o -name .git -o -name target -o -name dist -o -name build -o -name .venv -o -name venv -o -name vendor -o -name site-packages -o -name __pycache__ -o -name .tox -o -name .cache \) -prune -o -type f \( -name package.json -o -name pyproject.toml -o -name setup.py -o -name setup.cfg -o -name Cargo.toml \) -print 2>/dev/null | while IFS= read -r f; do
  case "$f" in
    */package.json) tr ',{}' '\n\n\n' < "$f" | sed -n 's/^[[:space:]]*"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/npm\t\1/p' | head -1 ;;
    */pyproject.toml) sed -n "s/^[[:space:]]*name[[:space:]]*=[[:space:]]*[\"']\([^\"']*\)[\"'].*/pypi\t\1/p" "$f" ;;
    */setup.py) tr ',()' '\n\n\n' < "$f" | sed -n "s/^[[:space:]]*name[[:space:]]*=[[:space:]]*[\"']\([^\"']*\)[\"'].*/pypi\t\1/p" ;;
    */setup.cfg) sed -n 's/^[[:space:]]*name[[:space:]]*=[[:space:]]*\([^[:space:]]*\).*/pypi\t\1/p' "$f" | head -1 ;;
    */Cargo.toml) sed -n "s/^[[:space:]]*name[[:space:]]*=[[:space:]]*[\"']\([^\"']*\)[\"'].*/crates\t\1/p" "$f" ;;
  esac
done
b=$(basename "$ROOT"); printf 'npm\t%s\npypi\t%s\ncrates\t%s\n' "$b" "$b" "$b"
echo SWT_PKGSCAN_DONE
"""


def workspace_packages_script(workdir: str) -> str:
    import shlex

    return _PKGSCAN_TEMPLATE.replace("__ROOT__", shlex.quote(workdir))


def parse_workspace_packages(stdout: str) -> dict[str, list[str]]:
    """``kind<TAB>name`` lines from :func:`workspace_packages_script` → denylist dict."""
    out: dict[str, set[str]] = {"npm": set(), "pypi": set(), "crates": set()}
    for line in stdout.splitlines():
        if "\t" not in line:
            continue
        kind, _, name = line.partition("\t")
        name = name.strip()
        if kind in out and 0 < len(name) < 200 and not name.startswith("$") and "{" not in name:
            out[kind].add(name)
    return {k: sorted(v) for k, v in out.items()}
