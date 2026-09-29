# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SUPPORTED_LANGUAGES = ["de", "es", "fr", "ja"]

# English instruction from Skills' generic/math prompt.
EN_INSTRUCTION = r"Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}."

# Language-specific instructions mirrored from Skills' multilingual math prompts.
MATH_INSTRUCTIONS = {
    "de": r"Lösen Sie das folgende Mathematikproblem. Stellen Sie sicher, dass Sie die Antwort (und nur die Antwort) in \boxed{} setzen.",
    "es": r"Resuelve el siguiente problema matemático. Asegúrate de poner la respuesta (y solo la respuesta) dentro de \boxed{}.",
    "fr": r"Résolvez le problème mathématique suivant. Assurez-vous de mettre la réponse (et seulement la réponse) dans \boxed{}.",
    "ja": r"以下の数学問題を解いてください。答え（答えのみ）を\boxed{}の中に入れてください。",
}
