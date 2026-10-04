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
"""Toolchain setup the Apptainer ``swe_agents`` harness applied to every container.

``responses_api_agents/swe_agents/app.py`` (``_build_apptainer_command``) gave both the agent's
container and the evaluator's container a Maven/Gradle mirror, a Chrome wrapper for Karma, and
a few JVM/.NET variables. Sandbox resources servers opt in with
``sandbox_config.legacy_container_setup: true`` so a run migrated from that harness sees the
same build environment on both sides. The Apptainer-only steps (rewriting ``/etc/hosts`` and
pre-creating ``/var/run/postgresql`` for uid namespacing) are not carried over.
"""

from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox


MAVEN_MIRROR_URL = "https://maven-central.storage-download.googleapis.com/maven2/"

# Same paths swe_rebench's verification ships its mirror to, so an eval sandbox that gets both
# holds one copy of each file rather than two Gradle init scripts.
MAVEN_SETTINGS_PATH = "/root/.m2/settings.xml"
GRADLE_INIT_PATH = "/root/.gradle/init.d/nemo_gym_mirror.gradle"
CHROME_WRAPPER_PATH = "/tmp/chrome-wrapper.sh"

LEGACY_CONTAINER_ENV: dict[str, str] = {
    # preferIPv6Addresses: dual-stack networks misrouted Maven/Gradle over IPv6. Robolectric's
    # MavenArtifactFetcher bypasses project repositories, so it is pointed at the mirror directly.
    "_JAVA_OPTIONS": f"-Djava.net.preferIPv6Addresses=false -Drobolectric.dependency.repo.url={MAVEN_MIRROR_URL}",
    # Gradle only auto-loads init scripts from $GRADLE_USER_HOME/init.d.
    "GRADLE_USER_HOME": "/root/.gradle",
    # Cap the CoreCLR heap reservation at 8 GiB.
    "DOTNET_GCHeapHardLimit": "0x200000000",
    # Karma's Chrome/ChromeHeadless launchers read CHROME_BIN, the Chromium ones CHROMIUM_BIN.
    "CHROME_BIN": CHROME_WRAPPER_PATH,
    "CHROMIUM_BIN": CHROME_WRAPPER_PATH,
}

# Execs the image's real Chrome with the flags a root, small-/dev/shm container needs. Karma
# configs that hardcode flags through customLaunchers bypass it.
_CHROME_WRAPPER = (
    "#!/bin/sh\n"
    "for b in /opt/google/chrome/google-chrome /opt/google/chrome/chrome "
    "/usr/bin/google-chrome-stable /usr/bin/google-chrome "
    "/usr/lib/chromium/chrome /usr/bin/chromium-browser /usr/bin/chromium; do\n"
    '  if [ -x "$b" ] && [ "$(realpath "$b")" != "$(realpath "$0")" ]; then\n'
    '    exec "$b" --no-sandbox --disable-dev-shm-usage "$@"\n'
    "  fi\n"
    "done\n"
    "echo 'chrome-wrapper.sh: no real chrome binary found' >&2\n"
    "exit 1\n"
)

# Uploaded files arrive without the execute bit. Some images point GRADLE_USER_HOME at another
# home through gradle.properties or a wrapper script, so the init script is copied there too.
LEGACY_CONTAINER_SETUP_COMMAND = (
    f"chmod +x {CHROME_WRAPPER_PATH}; "
    "for d in /home/gradle/.gradle/init.d /home/user/.gradle/init.d; do "
    f'mkdir -p "$d" 2>/dev/null && cp {GRADLE_INIT_PATH} "$d/" 2>/dev/null; done; true'
)


def legacy_container_files() -> dict[str, str]:
    """Files every legacy container got: the Maven/Gradle mirror and the Chrome wrapper."""
    mirror_dir = Path(__file__).resolve().parents[2] / "responses_api_agents" / "swe_agents" / "maven_mirror"
    return {
        MAVEN_SETTINGS_PATH: (mirror_dir / "settings.xml").read_text(),
        GRADLE_INIT_PATH: (mirror_dir / "init.gradle").read_text(),
        CHROME_WRAPPER_PATH: _CHROME_WRAPPER,
    }


async def apply_legacy_container_setup(sandbox: AsyncSandbox) -> None:
    """Finish the setup ``legacy_container_files`` started; pass to ``AsyncSandbox.start_with_setup``."""
    result = await sandbox.exec(LEGACY_CONTAINER_SETUP_COMMAND, timeout_s=120)
    if result.return_code != 0 or result.error_type:
        raise RuntimeError(
            f"legacy container setup failed: return_code={result.return_code} "
            f"error_type={result.error_type} stderr={(result.stderr or '')[-500:]!r}"
        )
