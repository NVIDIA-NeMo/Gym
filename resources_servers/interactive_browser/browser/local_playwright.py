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
"""Local Chromium backend — the default, and the one CI runs."""

from __future__ import annotations

from typing import Optional

from .page import PlaywrightConnectedBackend


class LocalPlaywrightBackend(PlaywrightConnectedBackend):
    """Launches a Chromium on this machine, one browser process per rollout.

    Each instance owns its own browser and context (cookie/session isolation),
    so N rollouts do not interfere. No quota, no network dependency, released
    with the rollout — which is why it stays the default and why the offline
    `site/` tasks are graded against it in CI.
    """

    def __init__(self, headless: bool = True, *, session_metadata: Optional[dict[str, str]] = None):
        super().__init__(session_metadata=session_metadata)
        self._headless = headless

    async def _connect(self, playwright):
        # One browser process can host many isolated contexts; here we keep it
        # simple and own a context per backend instance.
        browser = await playwright.chromium.launch(headless=self._headless)
        try:
            context = await browser.new_context()
        except Exception:
            await browser.close()
            raise
        return browser, context, True
