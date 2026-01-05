# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from enum import Enum


class LMStyle(Enum):
    OpenAIChat = "OpenAIChat"
    OpenAIReasonPreview = "OpenAIReasonPreview"
    OpenAIReason = "OpenAIReason"

    Claude = "Claude"  # Claude 1 and Claude 2
    Claude3 = "Claude3"
    Claude3Thinking = "Claude3Thinking"

    Gemini = "Gemini"
    GeminiThinking = "GeminiThinking"
    Grok = "Grok"

    MistralWeb = "MistralWeb"
    CohereCommand = "CohereCommand"

    DataBricks = "DataBricks"
    DeepSeekAPI = "DeepSeekAPI"

    GenericBase = "GenericBase"

    DeepSeekCodeInstruct = "DeepSeekCodeInstruct"
    CodeLLaMaInstruct = "CodeLLaMaInstruct"
    StarCoderInstruct = "StarCoderInstruct"
    CodeQwenInstruct = "CodeQwenInstruct"
    QwQ = "QwQ"
    LLaMa3 = "LLaMa3"
    DeepSeekR1 = "DeepSeekR1"

    TogetherAI = "TogetherAI"
