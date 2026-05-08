# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SUPPORTED_LANGUAGES = ["de", "es", "fr", "ja"]

# Regex to extract MCQ answer from \boxed{} output.
EXTRACT_REGEX = r"\\boxed\{([A-D])\}"

# English instruction from Skills' generic/general-boxed prompt.
EN_INSTRUCTION = r"Solve the following problem. Make sure to put the answer (and only answer) inside \boxed{}."

# Language-specific instructions mirrored from Skills' multilingual general-boxed prompts.
BOXED_INSTRUCTIONS = {
    "de": r"Lösen Sie das folgende Problem. Stellen Sie sicher, dass Sie die Antwort (und nur die Antwort) in \boxed{} setzen.",
    "es": r"Resuelve el siguiente problema. Asegúrate de poner la respuesta (y solo la respuesta) dentro de \boxed{}.",
    "fr": r"Résolvez le problème suivant. Assurez-vous de mettre la réponse (et seulement la réponse) dans \boxed{}.",
    "ja": r"以下の問題を解いてください。答え（答えのみ）を\boxed{}の中に入れてください。",
}
