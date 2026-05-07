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
"""Multilingual LiveCodeBench-X constants — mirrors NeMo Skills' versions in
`nemo_skills/dataset/livecodebench-x/livecodebench_x_utils.py`."""

SUPPORTED_LANGUAGES = ["de", "es", "fr", "ja"]
SUPPORTED_VERSIONS = ["v5", "v6"]

# English instruction (matches Skills' EN_INSTRUCTION).
EN_INSTRUCTION = "Here is a problem for which you need to generate executable code in the Python programming language."

# Language-specific instructions copied verbatim from Skills'
# `livecodebench_x_utils.CODEGEN_INSTRUCTIONS`. Translations match Skills'
# `eval/livecodebench/python_codegen_{lang}.yaml`.
CODEGEN_INSTRUCTIONS = {
    "de": ("Hier ist ein Problem, für das Sie ausführbaren Code in der Programmiersprache Python generieren müssen."),
    "es": (
        "Aquí tienes un problema para el cual necesitas generar un código "
        "ejecutable en el lenguaje de programación Python."
    ),
    "fr": ("Voici un problème pour lequel vous devez générer du code exécutable en langage de programmation Python."),
    "ja": ("以下は、Pythonプログラミング言語で実行可能なコードを生成する必要がある問題です。"),
}
