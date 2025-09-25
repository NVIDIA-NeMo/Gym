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
