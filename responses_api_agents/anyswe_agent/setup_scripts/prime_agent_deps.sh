#!/bin/bash
# Prime Agent ships through an r2.dev bucket the network blocks, and its package.json points its
# sibling packages there too. Install the same checksummed tarballs from the GitHub release instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_portable_python.sh"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

NODE_VERSION="${NODE_VERSION:-22.19.0}"
PRIME_AGENT_VERSION="${PRIME_AGENT_VERSION:-0.7.0}"

install_portable_python
install_nemo_gym_deps

if [ "$("$DEPS_DIR/bin/node" --version 2>/dev/null || true)" != "v${NODE_VERSION}" ]; then
    node_url="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz"
    curl -fsSL "$node_url" | tar xJ -C "$DEPS_DIR" --strip-components=1
fi
export PATH="$DEPS_DIR/bin:$PATH"

V=$PRIME_AGENT_VERSION
WORK=$(mktemp -d)
mkdir -p "$WORK/fixed"
cd "$WORK"
for p in prime-agent prime-agent-ai prime-agent-core prime-agent-tui; do
    curl -fsSL -o "$p-$V.tgz" "https://github.com/PrimeIntellect-ai/prime-agent/releases/download/v$V/$p-$V.tgz"
done
if [ "$V" = 0.7.0 ]; then
    sha256sum -c - <<SUMS
88b6578518c72cd51a825bc80f28e0fef9a64c67de4a7d6fd7afd7ca1b34da0b  prime-agent-$V.tgz
7cdbb3e835f48dd103325f7a351ce540b27af4d161aeb9c7b9bdcc12fe7909af  prime-agent-ai-$V.tgz
0313373089831d9a2ce06e874fab8b9c05762c0094ff9fc202908cf7db7f99cd  prime-agent-core-$V.tgz
3225f7f92e87db80fe2c9005d1f7770735ae625c32935ef2283688fc9bd33951  prime-agent-tui-$V.tgz
SUMS
fi
for p in prime-agent-core prime-agent; do
    mkdir -p "src/$p" && tar xzf "$p-$V.tgz" -C "src/$p"
    node -e '
        const fs = require("fs"), [f, dir] = process.argv.slice(1), d = JSON.parse(fs.readFileSync(f));
        for (const [k, v] of Object.entries(d.dependencies || {})) {
            const m = /r2\.dev\/releases\/v[^/]+\/([^/]+\.tgz)$/.exec(v);
            if (m) d.dependencies[k] = "file:" + (m[1].startsWith("prime-agent-core") ? dir + "/fixed/" : dir + "/") + m[1];
        }
        fs.writeFileSync(f, JSON.stringify(d, null, 2));
    ' "src/$p/package/package.json" "$WORK"
    ! grep -q r2.dev "src/$p/package/package.json"
    (cd "src/$p/package" && npm pack --ignore-scripts --pack-destination "$WORK/fixed" >/dev/null)
done
npm install -g --prefix "$DEPS_DIR" --no-fund --no-audit "$WORK/fixed/prime-agent-$V.tgz"
cd / && rm -rf "$WORK"
"$DEPS_DIR/bin/prime-agent" --version

# Prime's only code tool is an ipython kernel. Without PRIME_AGENT_KERNEL_PYTHON it bootstraps a venv per
# run with uv and PyPI, and task images have no uv. Preinstall what that bootstrap would, so the config can
# set PRIME_AGENT_KERNEL_PYTHON to this runtime's python.
"$DEPS_DIR/bin/python3" -m pip install ipykernel dill requests httpx pyyaml tomli python-dotenv pandas numpy scipy \
    beautifulsoup4 lxml pydantic tyro "$DEPS_DIR/lib/node_modules/prime-agent/dist/prime-agent-runtime"
"$DEPS_DIR/bin/python3" -c "import ipykernel, rlm; from rlm.harness import HarnessEntry; print('prime kernel ok')"
