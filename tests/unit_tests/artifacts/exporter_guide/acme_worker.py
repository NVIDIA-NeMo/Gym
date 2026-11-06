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
def run(payload: dict) -> dict:
    from acme_sdk import Client, UploadRefused

    try:
        upload = Client().upload(payload["staged_dir"], project=payload["project"])
    except UploadRefused as exc:
        return {"state": "failed", "error": f"Acme refused the upload: {exc}", "exit_code": 4}

    if upload.duplicate:
        return {"state": "skipped", "detail": {"upload_id": upload.id}}
    return {"detail": {"upload_id": upload.id}}
