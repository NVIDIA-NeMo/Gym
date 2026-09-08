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
from typing import Callable, Protocol


class LanguageConsistencyBackend(Protocol):
    def __call__(self, text: str, expected_language_code: str) -> float:
        """Return a language-consistency score between 0.0 and 1.0."""


def _load_wmt24pp_cld2() -> LanguageConsistencyBackend:
    # Imported lazily so `pycld2` (a compiled extension without wheels on every
    # Python version) is only required when this backend is actually selected —
    # not merely by importing the wmt_translation server with the feature off.
    from wmt24pp_cld2_language_consistency import wmt24pp_cld2_language_consistency_score

    return wmt24pp_cld2_language_consistency_score


def _load_flores_glotlid() -> LanguageConsistencyBackend:
    # GlotLID loads a large model at module import, so defer importing it until
    # this backend is explicitly selected.
    from flores_glotlid_language_consistency import flores_glotlid_language_consistency_score

    return flores_glotlid_language_consistency_score


# name -> zero-arg loader that imports and returns the backend on demand.
_LANGUAGE_CONSISTENCY_BACKEND_LOADERS: dict[str, Callable[[], LanguageConsistencyBackend]] = {
    "flores_glotlid": _load_flores_glotlid,
    "wmt24pp_cld2": _load_wmt24pp_cld2,
}


def get_language_consistency_backend(name: str) -> LanguageConsistencyBackend:
    try:
        loader = _LANGUAGE_CONSISTENCY_BACKEND_LOADERS[name]
    except KeyError as exc:
        available_backends = ", ".join(sorted(_LANGUAGE_CONSISTENCY_BACKEND_LOADERS))
        raise ValueError(
            f"Unknown language-consistency backend {name!r}; available backends: {available_backends}"
        ) from exc
    return loader()
