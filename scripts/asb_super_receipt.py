# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded authenticated receipt probe for the shared Super VL FDR service.

The function deliberately returns no URL, secret name, header, or token value. It
only establishes whether an existing service credential authenticates the pinned
model identity. This is a preflight probe, not a benchmark run.
"""

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

import modal


# ``modal run --env=FDR`` supplies the explicit creation environment. A local
# App declaration is required by this Modal SDK for function registration;
# ``App.lookup`` returns a deployed reference that cannot register one.
APP = modal.App("asb-super-receipt")
SUPER_SECRET = modal.Secret.from_name("nemotron-super-vl-service", environment_name="FDR")
BASE_URL = os.environ.get("ASB_SUPER_BASE_URL")
EXPECTED_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"


@APP.function(secrets=[SUPER_SECRET], timeout=60)
def receipt() -> dict[str, object]:
    """Return a redacted model receipt using any injected bearer-like value."""
    if not BASE_URL:
        return {"receipt": "missing_base_url", "expected_model_present": False, "model_count": 0}
    candidates = [
        value
        for key, value in os.environ.items()
        if value and any(fragment in key.upper() for fragment in ("API_KEY", "TOKEN", "BEARER", "AUTH"))
    ]
    for candidate in candidates:
        try:
            request = Request(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {candidate}"})
            with urlopen(request, timeout=30) as response:
                if response.status != 200:
                    continue
                payload = json.loads(response.read())
            ids = {item.get("id") for item in payload.get("data", [])}
            return {
                "receipt": "200",
                "expected_model_present": EXPECTED_MODEL in ids,
                "model_count": len(ids),
            }
        except (OSError, URLError, ValueError):
            continue
    return {"receipt": "unauthenticated", "expected_model_present": False, "model_count": 0}


@APP.local_entrypoint()
def main() -> None:
    print(receipt.remote())
