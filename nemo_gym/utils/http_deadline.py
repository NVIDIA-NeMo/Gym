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
"""An HTTP GET that ends at a total time limit."""

import threading
from typing import Any

import requests


def get_within_deadline(
    url: str, *, timeout_s: float, read_timeout_s: float | None = None, read_body: bool = True
) -> tuple[int, bytes]:
    """GET ``url`` and return its status code and body, raising ``TimeoutError`` when ``timeout_s`` seconds have passed.

    A requests timeout bounds the wait for each read, not the whole call: a peer that keeps sending a byte inside it,
    in the headers or in the body, keeps the call going. The request therefore runs on a daemon thread that the caller
    stops waiting for at the limit. That thread ends with its next read timeout (``read_timeout_s``, default
    ``timeout_s``) or when the peer closes, and a body is not read after the limit. With ``read_body=False`` the body
    is not read and ``b""`` is returned: for a readiness check, the answer is the headers.

    Exceptions of the request, such as ``requests.ConnectionError``, are raised unchanged.
    """
    outcome: dict[str, Any] = {}
    expired = threading.Event()

    def get() -> None:
        try:
            with requests.get(url, timeout=timeout_s if read_timeout_s is None else read_timeout_s, stream=True) as r:
                body = b""
                if read_body:
                    chunks = []
                    for chunk in r.iter_content(chunk_size=8192):
                        if expired.is_set():
                            return
                        chunks.append(chunk)
                    body = b"".join(chunks)
                outcome["result"] = (r.status_code, body)
        except BaseException as error:  # Handed to the caller, which raises it.
            outcome["error"] = error

    worker = threading.Thread(target=get, name="get-within-deadline", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        expired.set()
        raise TimeoutError(f"GET {url} did not finish within {timeout_s} s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]
