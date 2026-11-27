#!/usr/bin/env python3
"""One-time extraction of a warm SWE-smith $HOME (Maven/Gradle/etc build-tool caches) that every
SWE-smith rollout's app.py restore-copies into its own fresh per-rollout $HOME before the
container starts -- see app.py's CUSTOM-SWESMITH-HOME-CACHE block and SWESMITH_HOME_CACHE_DIR
in .env.example. Run from presync.sbatch under REBUILD_ONE_TIME_ARTIFACTS=true (network-requiring,
same reason as the other artifacts in that block).

Why SWE-smith's own VM sidecar (nemo_gym.vm_launcher), not a plain `apptainer exec` on the host:
SWE-smith's container images ship /testbed owned by root (confirmed: `drwxr-xr-x root root`), and
this script needs to run a REAL build (Maven writes scratch files directly under /testbed/target),
which fails outright as a non-root, non-fakeroot user ("Error creating dot file for archiving
instructions. /testbed/target/.plxarc", confirmed directly) -- exactly the ownership problem
_uses_vm_sidecar's docstring in app.py describes, and `--fakeroot` isn't a safe substitute on this
cluster (confirmed directly: "Failed to create user namespace: maximum number of user namespaces
exceeded" on a busy shared node). The real fix already exists for the rollout path (the VM
sidecar) -- reuse it here too instead of inventing a second one.

This only targets apache/dubbo today (the one JVM repo this was validated against). Re-run with
different --repo-image/--test-cmd to warm additional JVM repos into the same cache dir --
Maven/Gradle caches are keyed by group:artifact:version, not by repo, so one JVM repo's warm
dependency tree is a reasonable start for others that share common dependencies, and is harmless
(fail-open, see app.py) for any that don't.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

GYM_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(GYM_DIR))

DEFAULT_SWESMITH_SIF_DIR = os.environ.get(
    "SWESMITH_SIF_DIR", "/leonardo_scratch/large/userexternal/nruggeri/images/swesmith"
)
DEFAULT_GOLDEN_IMAGE_DIR = os.environ.get("VM_GOLDEN_IMAGE_DIR", "/leonardo_scratch/fast/BOOST_EUROPA/nruggeri/images/vm")
DEFAULT_SETUP_DIR = str(GYM_DIR / "responses_api_agents" / "swe_agents" / "swe_swesmith_setup")
DEFAULT_TEST_CMD = "cd /testbed && ./mvnw -B test -pl dubbo-common -am -Dtest=ConfigUtilsTest -DfailIfNoTests=false"


def find_default_repo_image(sif_dir: str) -> str:
    candidates = sorted(Path(sif_dir).glob("swesmith.*.apache_1776_dubbo.*.sif"))
    if not candidates:
        return ""
    return str(candidates[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-image", default=None, help="Path to the SWE-smith .sif image to warm the cache from")
    parser.add_argument("--test-cmd", default=DEFAULT_TEST_CMD, help="Build/test command run inside /testbed")
    parser.add_argument("--cache-dir", default=os.environ.get("SWESMITH_HOME_CACHE_DIR", ""))
    parser.add_argument("--sif-dir", default=DEFAULT_SWESMITH_SIF_DIR)
    parser.add_argument("--golden-image-dir", default=DEFAULT_GOLDEN_IMAGE_DIR)
    parser.add_argument("--setup-dir", default=DEFAULT_SETUP_DIR)
    args = parser.parse_args()

    if not args.cache_dir:
        print("=== extract_swesmith_home_cache: SWESMITH_HOME_CACHE_DIR not set (see .env.example) -- skipping ===")
        return 0

    repo_image = args.repo_image or find_default_repo_image(args.sif_dir)
    if not repo_image or not Path(repo_image).is_file():
        print(f"=== extract_swesmith_home_cache: no apache/dubbo .sif found under {args.sif_dir} -- skipping ===")
        return 0

    if not Path(args.golden_image_dir, "golden.qcow2").is_file():
        print(f"=== extract_swesmith_home_cache: no golden VM image at {args.golden_image_dir} -- skipping ===")
        return 0

    from nemo_gym import vm_launcher

    # Avoid a real ray.init() -- this script runs standalone from presync.sbatch, no Ray cluster
    # exists yet, and a real local Ray boot thrashes badly on a busy shared node (see vm_launcher's
    # own _cluster_scope_id() docstring for why it's keyed off the GCS address at all). A fixed
    # scope id is fine here: this script never runs concurrently with itself on the same node
    # (it's a one-shot presync step), unlike the real training-job case _cluster_scope_id exists
    # for.
    vm_launcher._cluster_scope_id = lambda: "swesmith-home-cache-extraction"

    cache_dir = Path(args.cache_dir)
    cache_parent = cache_dir.parent
    cache_parent.mkdir(parents=True, exist_ok=True)
    staging_dir = cache_parent / f"{cache_dir.name}.staging"
    subprocess.run(["rm", "-rf", str(staging_dir)], check=False)
    staging_dir.mkdir(parents=True, exist_ok=True)

    swesmith_dir = f"{args.setup_dir}/SWE-smith"
    swesmith_image = f"{args.setup_dir}/swesmith.img"
    sif_parent = str(Path(repo_image).parent)

    print(f"=== extract_swesmith_home_cache: booting VM sidecar (golden image: {args.golden_image_dir}) ===")
    info = vm_launcher.ensure_vm(
        golden_image_dir=args.golden_image_dir,
        # cache_parent (not just cache_dir) so the staging sibling directory is visible too --
        # 9p shares are mounted at the exact path given, not implicitly covering parent/siblings.
        lustre_share_roots=[sif_parent, args.setup_dir, str(cache_parent)],
        block_device_mounts=[(swesmith_image, f"{swesmith_dir}/venv")],
    )
    print(f"=== extract_swesmith_home_cache: VM ready on 127.0.0.1:{info.ssh_port} ===")

    # Build directly into the (shared, 9p-mounted-at-the-same-path) staging dir as HOME -- no
    # separate guest-local build + copy-back needed, unlike the per-rollout guest_home_dir in
    # app.py: this is a one-time, non-concurrent build, so the "root-owned files don't propagate
    # ownership back through 9p" footgun that forces guest-local paths for rollouts doesn't apply
    # here (nothing downstream checks these files' ownership, only their content/readability).
    # [VALIDATION FINDING 2026-11-27] This golden VM image's qemu-packages.txt installs Debian's
    # `singularity-container` package, whose binary is literally named `singularity`, not
    # `apptainer` -- confirmed directly by SSHing into a real booted VM (no `apptainer` binary
    # anywhere on its root filesystem). Every real config's `apptainer_binary: apptainer` (e.g.
    # configs/swesmith-vm-sidecar-debug.yaml) would hit the exact same "command not found" this
    # extraction script did on first attempt. Out of scope to fix broadly here (pre-existing,
    # affects every VM-sidecar dataset type, not just this cache feature) -- falls back to
    # `singularity` so this script still works; flagged for a separate fix.
    binary_check = subprocess.run(
        f"ssh -q -i {shlex.quote(info.ssh_key)} -p {info.ssh_port} -o BatchMode=yes "
        f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@127.0.0.1 "
        f"'command -v apptainer || command -v singularity'",
        shell=True,
        capture_output=True,
        text=True,
    )
    apptainer_bin = (binary_check.stdout.strip().splitlines() or ["apptainer"])[0] or "apptainer"
    print(f"=== extract_swesmith_home_cache: using '{apptainer_bin}' as the container binary ===")

    container_cmd = (
        f"mkdir -p {shlex.quote(str(staging_dir))} && "
        f"{apptainer_bin} exec --contain --writable-tmpfs --cleanenv --pid "
        f"--no-mount tmp,bind-paths -H {shlex.quote(str(staging_dir))} --env TMPDIR=/tmp "
        f"--mount type=bind,src={shlex.quote(swesmith_dir)},dst=/swesmith_setup "
        f"{shlex.quote(repo_image)} bash -c {shlex.quote(args.test_cmd)}"
    )
    ssh_cmd = (
        f"ssh -q -i {shlex.quote(info.ssh_key)} -p {info.ssh_port} -o BatchMode=yes "
        f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 "
        f"root@127.0.0.1 {shlex.quote(container_cmd)}"
    )

    print(f"=== extract_swesmith_home_cache: warming {staging_dir} from {repo_image} ===")
    print(f"=== test command: {args.test_cmd} ===")
    t0 = time.time()
    proc = subprocess.run(ssh_cmd, shell=True)
    elapsed = time.time() - t0
    print(f"=== extract_swesmith_home_cache: one-time extraction build took {elapsed:.1f}s (rc={proc.returncode}) ===")

    # [CUSTOM-SWESMITH-HOME-CACHE-TAR 2026-11-27] app.py's per-rollout restore reads a single
    # tarball, not this exploded directory -- confirmed directly that `cp -a`/`shutil.copytree`
    # on ~180 small files over the VM sidecar's virtio-9p share took 97-138s, almost entirely
    # per-file round-trip overhead (a raw bulk-bytes benchmark over the same mount hit ~884
    # MB/s, so bandwidth was never the bottleneck). Build the tarball from the staging dir
    # BEFORE promotion, so both the directory and its tarball are produced from the exact same
    # build output and promoted together.
    staging_tar = cache_parent / f"{cache_dir.name}.staging.tar"
    subprocess.run(["rm", "-f", str(staging_tar)], check=False)
    tar_proc = subprocess.run(
        ["tar", "-cf", str(staging_tar), "-C", str(staging_dir), "."],
        capture_output=True,
        text=True,
    )
    if tar_proc.returncode != 0:
        print(f"=== extract_swesmith_home_cache: WARNING: tar packaging failed, rc={tar_proc.returncode}: {tar_proc.stderr} ===")

    # Promote atomically regardless of build rc: a failed/partial build's downloaded artifacts
    # are still a strict improvement over an empty cache for whatever DID get fetched before the
    # failure (fail-open per-rollout restore just gets a smaller head start), and this script
    # must never block the rest of presync.sbatch on a flaky one-off build failure.
    old_dir = cache_parent / f"{cache_dir.name}.old"
    old_tar = cache_parent / f"{cache_dir.name}.tar.old"
    cache_tar = cache_parent / f"{cache_dir.name}.tar"
    subprocess.run(["rm", "-rf", str(old_dir)], check=False)
    subprocess.run(["rm", "-f", str(old_tar)], check=False)
    if cache_dir.exists():
        cache_dir.rename(old_dir)
    if cache_tar.exists():
        cache_tar.rename(old_tar)
    staging_dir.rename(cache_dir)
    if staging_tar.is_file():
        staging_tar.rename(cache_tar)
    subprocess.run(["rm", "-rf", str(old_dir)], check=False)
    subprocess.run(["rm", "-f", str(old_tar)], check=False)

    size = subprocess.run(["du", "-sh", str(cache_dir)], capture_output=True, text=True).stdout.split()
    tar_size = subprocess.run(["du", "-sh", str(cache_tar)], capture_output=True, text=True).stdout.split() if cache_tar.is_file() else []
    print(
        f"=== extract_swesmith_home_cache: done, cache at {cache_dir} ({size[0] if size else '?'}), "
        f"tar at {cache_tar} ({tar_size[0] if tar_size else '?'}) ==="
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
