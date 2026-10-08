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
import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from language_consistency import get_language_consistency_backend


def test_wmt24pp_cld2_backend_scores_wmt24pp_language_codes() -> None:
    backend = get_language_consistency_backend("wmt24pp_cld2")

    german = "Der schnelle braune Fuchs springt über den faulen Hund."
    french = "Le renard brun rapide saute par-dessus le chien paresseux."

    assert backend(german, "de_DE") > 0.5
    assert backend(french, "fr_FR") > 0.5
    assert backend(german, "fr_FR") < 0.5


def test_unknown_backend_name_raises_with_available_backends_listed() -> None:
    with pytest.raises(ValueError) as exc_info:
        get_language_consistency_backend("does_not_exist")

    message = str(exc_info.value)
    assert "flores_glotlid" in message
    assert "wmt24pp_cld2" in message


def test_unknown_backend_name_does_not_invoke_any_loader() -> None:
    mock_loaders = {
        "flores_glotlid": MagicMock(),
        "wmt24pp_cld2": MagicMock(),
    }
    with patch.dict(
        "language_consistency._LANGUAGE_CONSISTENCY_BACKEND_LOADERS",
        mock_loaders,
        clear=True,
    ):
        with pytest.raises(ValueError):
            get_language_consistency_backend("does_not_exist")

    for mock_loader in mock_loaders.values():
        mock_loader.assert_not_called()


def test_known_backend_name_invokes_its_lazy_loader_once() -> None:
    mock_loader = MagicMock()

    with patch.dict(
        "language_consistency._LANGUAGE_CONSISTENCY_BACKEND_LOADERS",
        {"wmt24pp_cld2": mock_loader},
    ):
        backend = get_language_consistency_backend("wmt24pp_cld2")

    mock_loader.assert_called_once_with()
    assert backend is mock_loader.return_value


def test_flores_glotlid_backend_is_imported_lazily() -> None:
    expected_backend = MagicMock()
    fake_module = ModuleType("flores_glotlid_language_consistency")
    fake_module.flores_glotlid_language_consistency_score = expected_backend

    with patch.dict(sys.modules, {"flores_glotlid_language_consistency": fake_module}):
        backend = get_language_consistency_backend("flores_glotlid")

    assert backend is expected_backend


def test_importing_registry_does_not_import_glotlid_dependencies() -> None:
    module_path = Path(__file__).parents[1] / "language_consistency.py"
    spec = importlib.util.spec_from_file_location("_language_consistency_import_safety_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name not in {
            "fasttext",
            "flores_glotlid_language_consistency",
            "huggingface_hub",
        }
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", new=guarded_import):
        spec.loader.exec_module(module)
