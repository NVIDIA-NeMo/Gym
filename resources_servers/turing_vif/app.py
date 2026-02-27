"""
Turing VIF Resource Server for NeMo Gym.

This resource server integrates the Turing VIF (Verifiable Instruction Following)
validators into NeMo Gym's reinforcement learning framework. It supports both
fast rule-based validators and async LLM-based judge validators.
"""

import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, Field, ValidationError

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI


# Handle imports for both direct execution and module import
try:
    from .vif_validators.data_loader import (
        DEFINITION_GENERATOR_SYSTEM_PROMPT,
        EXPECTED_ARGUMENTS,
        JUDGE_SYSTEM_PROMPT,
        LLM_INSTRUCTIONS,
        LLM_JUDGE_QUESTION_PROMPT,
        eval_modes,
        inst_def,
        subinst_def,
    )
    from .vif_validators.validator import (
        SUPPORTED_LANGS,
        is_instruction_supported,
        validate_instruction,
    )
except ImportError:
    # When run directly (not as a module), add parent to path
    sys.path.insert(0, str(Path(__file__).parent))
    from vif_validators.data_loader import (
        DEFINITION_GENERATOR_SYSTEM_PROMPT,
        EXPECTED_ARGUMENTS,
        JUDGE_SYSTEM_PROMPT,
        LLM_INSTRUCTIONS,
        LLM_JUDGE_QUESTION_PROMPT,
        eval_modes,
        inst_def,
        subinst_def,
    )
    from vif_validators.validator import (
        SUPPORTED_LANGS,
        is_instruction_supported,
        validate_instruction,
    )


logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


class TuringVIFResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the Turing VIF Resource Server."""

    judge_base_url: Optional[str] = Field(
        default=None, description="Base URL for the LLM judge API. If not set, uses policy_base_url."
    )
    judge_api_key: Optional[str] = Field(
        default=None, description="API key for the LLM judge. If not set, uses policy_api_key."
    )
    judge_model: str = Field(default="gpt-4.1-2025-04-14", description="Model to use for LLM judge evaluations.")
    # Security limits for judge LLM calls (input/output length and error handling)
    judge_max_system_chars: Optional[int] = Field(
        default=200_000, description="Max character count for judge system prompt. None to disable."
    )
    judge_max_user_chars: Optional[int] = Field(
        default=100_000, description="Max character count for judge user content. None to disable."
    )
    judge_max_output_chars: Optional[int] = Field(
        default=50_000, description="Max character count for judge response. None to disable."
    )
    judge_error_fallback: str = Field(
        default="Judge unavailable.",
        description="Safe string returned when the judge LLM call fails (security: no error leak).",
    )


# ============================================================================
# Request/Response Models
# ============================================================================


class InstructionItem(BaseModel):
    """A single instruction with its parameters."""

    instruction_id: str
    # Additional kwargs are captured via model_extra
    model_config = {"extra": "allow"}


class LLMJudgeItem(BaseModel):
    """A custom LLM judge question."""

    uid: int
    content: str
    pass_criteria: Literal["YES", "NO"] = Field(
        default="YES",
        description="Expected verdict from judge for the response to pass. 'YES' means judge must say YES for pass, 'NO' means judge must say NO for pass.",
    )
    source: Literal["user", "system"]
    is_misalignment_check: bool


class TuringVIFRunRequest(BaseRunRequest):
    """Request model for the Turing VIF resource server."""

    id: int = Field(default=0, description="Request identifier")
    instructions: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of instruction objects with instruction_id and kwargs"
    )
    llm_judge: List[LLMJudgeItem] = Field(default_factory=list, description="List of custom LLM judge questions")
    prompt: Optional[str] = Field(default=None, description="The original user prompt")
    language: str = Field(
        default="en",
        description="Language code for multi-language validation (e.g., 'en', 'es', 'ja', 'zh', 'hi', 'ar')",
    )


class TuringVIFVerifyRequest(TuringVIFRunRequest, BaseVerifyRequest):
    """Verify request combining run request with response."""

    pass


class ValidationResult(BaseModel):
    """Result of a single validation check."""

    instruction: str
    status: Literal["Passed", "Failed", "Skipped"]
    message: str


class TuringVIFVerifyResponse(BaseVerifyResponse):
    """Response from the verify endpoint."""

    follow_all_instructions: bool
    follow_instruction_list: List[bool]
    validation_results: List[ValidationResult] = Field(default_factory=list)


class ValidationError(BaseModel):
    """Error in a single validation check."""

    errors: List[str]


# ============================================================================
# Pydantic Models for LLM Judge Response Parsing
# ============================================================================


class JudgeResponse(BaseModel):
    """Expected JSON structure for LLM Judge responses."""

    verdict: Literal["YES", "NO"]
    reasoning: str


class DefinitionResponse(BaseModel):
    """Expected JSON structure for definition generator responses."""

    status: Literal["PASS", "FAIL"]
    definition: str


# ============================================================================
# Resource Server Implementation
# ============================================================================


class TuringVIFResourcesServer(SimpleResourcesServer):
    """
    Turing VIF Resource Server for NeMo Gym.

    Validates LLM responses against instruction-following criteria using both
    fast rule-based validators and async LLM-as-a-judge validators.
    """

    config: TuringVIFResourcesServerConfig
    _judge_client: Optional[NeMoGymAsyncOpenAI] = None
    _definition_cache: Dict[Tuple[str, str], Tuple[str, bool]] = {}

    # GPT-5 and other reasoning models that require the Responses API
    REASONING_MODELS: ClassVar[List[str]] = ["gpt-5", "o1", "o3", "o4-mini"]

    @staticmethod
    def analyze_misalignment_check(is_valid: bool, message: str) -> Tuple[bool, str]:
        """
        Inverts validation result for misalignment checks.

        When source="user" and is_misalignment_check=True, a passing validation
        means the response followed the user's misaligned instruction (bad),
        so we invert the result.
        """
        if is_valid:
            return (False, "Response misaligns with system instruction.")
        else:
            return (True, "No Error")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if the model is a reasoning model that requires Responses API."""
        model_lower = model_name.lower()
        return any(rm in model_lower for rm in self.REASONING_MODELS)

    def _get_judge_client(self) -> NeMoGymAsyncOpenAI:
        """Get or create the LLM judge client."""
        if self._judge_client is None:
            # Use judge-specific config if available, otherwise fall back to policy config
            base_url = self.config.judge_base_url or getattr(
                self.config, "policy_base_url", "https://api.openai.com/v1"
            )
            api_key = self.config.judge_api_key or getattr(self.config, "policy_api_key", "")

            self._judge_client = NeMoGymAsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        return self._judge_client

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    # ========================================================================
    # Async LLM Judge Functions (with security wrapper)
    # ========================================================================

    async def _judge_llm_api_call_async(
        self,
        client: NeMoGymAsyncOpenAI,
        user_content: str,
        system_content: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Internal: perform the actual judge LLM API call (Responses or Chat Completions).
        Call only via _judge_llm_api_async, which applies security checks.
        """
        model = self.config.judge_model
        if self._is_reasoning_model(model):
            result = await client.create_response(
                model=model,
                input=[{"role": "developer", "content": system_content}, {"role": "user", "content": user_content}],
                max_output_tokens=max_tokens,
            )
            output_text = ""
            for output_item in result.get("output", []):
                if output_item.get("type") == "message":
                    for content_item in output_item.get("content", []):
                        if content_item.get("type") == "output_text":
                            output_text += content_item.get("text", "")
            return output_text
        else:
            result = await client.create_chat_completion(
                model=model,
                messages=[{"role": "system", "content": system_content}, {"role": "user", "content": user_content}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return result["choices"][0]["message"]["content"]

    async def _judge_llm_api_async(
        self, user_content: str, system_content: str, temperature: float = 1.0, max_tokens: int = 10000
    ) -> str:
        """
        Async wrapper for LLM judge API calls with security safeguards.

        - Validates and caps input lengths (system/user) to prevent abuse.
        - Catches all exceptions from the judge API and returns a safe fallback
          (no error details leaked).
        - Caps output length before returning.

        Uses Responses API for reasoning models and Chat Completions for others.
        """
        cfg = self.config
        max_sys = getattr(cfg, "judge_max_system_chars", None)
        max_usr = getattr(cfg, "judge_max_user_chars", None)
        max_out = getattr(cfg, "judge_max_output_chars", None)
        fallback = getattr(cfg, "judge_error_fallback", "Judge unavailable.")

        if not isinstance(system_content, str):
            system_content = str(system_content)
        if not isinstance(user_content, str):
            user_content = str(user_content)

        if max_sys is not None and len(system_content) > max_sys:
            system_content = system_content[:max_sys]
        if max_usr is not None and len(user_content) > max_usr:
            user_content = user_content[:max_usr]

        try:
            client = self._get_judge_client()
            out = await self._judge_llm_api_call_async(client, user_content, system_content, temperature, max_tokens)
            if not isinstance(out, str):
                out = str(out) if out is not None else ""
            if max_out is not None and len(out) > max_out:
                out = out[:max_out]
            return out
        except Exception as e:
            logger.warning("Judge LLM call failed: %s", type(e).__name__, exc_info=False)
            return fallback

    async def _validate_custom_llm_judge_async(self, response: str, question_text: str) -> Tuple[bool, str]:
        """
        Validates a response against a free-form LLM Judge question.

        Args:
            response: The model response to evaluate
            question_text: The question to evaluate against

        Returns:
            Tuple of (is_valid, reasoning)
        """
        try:
            judge_prompt = LLM_JUDGE_QUESTION_PROMPT.format(question=question_text, model_response=response)

            evaluation = await self._judge_llm_api_async(
                user_content="Evaluate the response.", system_content=judge_prompt
            )

            # Parse response
            evaluation = evaluation.strip()

            # Handle Markdown code blocks
            if evaluation.startswith("```"):
                evaluation = re.sub(r"^```(?:\w+)?\s*", "", evaluation, flags=re.DOTALL)
                evaluation = re.sub(r"\s*```$", "", evaluation, flags=re.DOTALL)

            # Extract JSON
            json_match = re.search(r"(\{.*\})", evaluation, re.DOTALL)
            if json_match:
                evaluation = json_match.group(1)

            json_data = json.loads(evaluation)

            # Check if judge returned wrong format
            if "model_response" in json_data or "question" in json_data:
                return False, "Judge returned input format instead of output format."

            judge_response = JudgeResponse(**json_data)
            flag = judge_response.verdict == "YES"
            return flag, judge_response.reasoning

        except (json.JSONDecodeError, ValidationError) as e:
            return False, f"Error parsing Judge response: {e}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    async def _get_dynamic_definition_async(self, inst_type: str, term: str) -> Tuple[str, bool]:
        """
        Calls an LLM to dynamically define a sub-instruction term.

        Args:
            inst_type: The instruction type
            term: The term to define

        Returns:
            Tuple of (definition, is_valid)
        """
        cache_key = (inst_type, term)
        if cache_key in self._definition_cache:
            return self._definition_cache[cache_key]

        try:
            instruction_name = inst_def.get(inst_type, {}).get("instruction_name", inst_type)
            context_terms_list = list(subinst_def.get(inst_type, {}).keys())
            context_terms_str = ", ".join(context_terms_list) if context_terms_list else "none"

            system_prompt = DEFINITION_GENERATOR_SYSTEM_PROMPT.format(
                instruction=instruction_name, inst_label=inst_type, term=term, context_related_terms=context_terms_str
            )

            response_str = await self._judge_llm_api_async(
                user_content=f"Define the term: {term}", system_content=system_prompt
            )

            # Parse the response
            evaluation = response_str.strip()
            if evaluation.startswith("```"):
                evaluation = re.sub(r"^```(?:\w+)?\s*", "", evaluation, flags=re.DOTALL)
                evaluation = re.sub(r"\s*```$", "", evaluation, flags=re.DOTALL)

            json_match = re.search(r"(\{.*\})", evaluation, re.DOTALL)
            if json_match:
                evaluation = json_match.group(1)

            json_data = json.loads(evaluation)
            definition = json_data.get("definition", "definition not found")
            status = json_data.get("status", "FAIL")

            if status == "PASS":
                result = (definition, True)
            else:
                result = (definition, False)

            self._definition_cache[cache_key] = result
            return result

        except (json.JSONDecodeError, KeyError) as e:
            return (f"Error parsing definition response: {e}", False)
        except Exception as e:
            return (f"Error in definition generation: {e}", False)

    async def _validate_llm_instruction_async(
        self, response: str, inst_type: str, kwargs: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validates a response using the LLM judge for stylistic/linguistic instructions.

        Args:
            response: The model response to evaluate
            inst_type: The instruction type (e.g., "stylistic:tone_formality")
            kwargs: The instruction arguments

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            argument_strings = []
            instruction_type = inst_def.get(inst_type, {}).get("instruction_type", "")
            type_definition = eval_modes.get(instruction_type, {}).get("definition", "")
            evaluation_mode_str = f"{instruction_type} - {type_definition}"

            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    arg_value_str = str(arg_value)
                    definition = ""

                    try:
                        if arg_value_str in subinst_def.get(inst_type, {}):
                            definition = subinst_def[inst_type][arg_value_str]
                        elif "num_" in arg_name or arg_name == "relation":
                            pass  # No definition needed for numeric args
                        else:
                            definition, is_valid = await self._get_dynamic_definition_async(inst_type, arg_value_str)

                            if not is_valid:
                                return (False, f"Invalid argument: '{arg_value_str}' is not valid for '{inst_type}'")

                        argument_strings.append(f"- {arg_name} ({arg_value_str}): {definition}")
                    except KeyError:
                        argument_strings.append(f"- {arg_name}: {arg_value_str}")

                instruction_arguments = "\n".join(argument_strings)
            else:
                instruction_arguments = "N/A"

            # Format the judge prompt
            judge_prompt = JUDGE_SYSTEM_PROMPT.format(
                model_response=response,
                instruction_name=inst_def.get(inst_type, {}).get("instruction_name", inst_type),
                instruction_definition=inst_def.get(inst_type, {}).get("definition", ""),
                instruction_arguments=instruction_arguments,
                evaluation_mode=evaluation_mode_str,
            )

            evaluation = await self._judge_llm_api_async(response, judge_prompt)

            # Parse response
            evaluation = evaluation.strip()
            if evaluation.startswith("```"):
                evaluation = re.sub(r"^```(?:\w+)?\s*", "", evaluation, flags=re.DOTALL)
                evaluation = re.sub(r"\s*```$", "", evaluation, flags=re.DOTALL)

            json_match = re.search(r"(\{.*\})", evaluation, re.DOTALL)
            if json_match:
                evaluation = json_match.group(1)

            json_data = json.loads(evaluation)

            if "model_response" in json_data or "question" in json_data:
                return (False, "Judge returned input format instead of output format.")

            judge_response = JudgeResponse(**json_data)
            flag = judge_response.verdict == "YES"
            return (flag, judge_response.reasoning)

        except (json.JSONDecodeError, ValidationError) as e:
            return (False, f"Error parsing LLM Judge response: {e}")
        except Exception as e:
            return (False, f"Validation error: {str(e)}")

    # ========================================================================
    # Main Verify Endpoint
    # ========================================================================

    async def validate_instructions_schema(self, instructions: List[Dict[str, Any]]) -> List[ValidationResult]:
        errors = []
        all_instructions = instructions.get("instructions", [])
        llm_judge = instructions.get("llm_judge", [])

        seen_uids: Dict[str, str] = {}
        for idx, instruction in enumerate(all_instructions):
            if not isinstance(instruction, dict):
                errors.append(f"Instruction at index {idx}: must be an object")
                continue

            # Validate instruction_id is present
            inst_id = instruction.get("instruction_id")
            if not inst_id:
                errors.append(f"Instruction at index {idx}: must have an 'instruction_id' field")
                continue

            # Validate expected arguments for the instruction
            if inst_id in EXPECTED_ARGUMENTS:
                expected_args = EXPECTED_ARGUMENTS[inst_id]
                actual_args = set(
                    k
                    for k in instruction.keys()
                    if k not in ("instruction_id", "uid", "source", "is_misalignment_check")
                )
                missing_args = set(expected_args) - actual_args
                if missing_args:
                    errors.append(
                        f"Instruction '{inst_id}' at index {idx}: missing required argument(s): {sorted(missing_args)}"
                    )
            else:
                errors.append(f"Instruction '{inst_id}' at index {idx}: unknown instruction_id")

            # Validate uid is present and unique
            uid = instruction.get("uid")
            if uid is None:
                errors.append(f"Instruction '{inst_id}' at index {idx}: must have a 'uid' field")
            else:
                # Check for duplicate uid
                if uid in seen_uids:
                    errors.append(
                        f"Instruction '{inst_id}' at index {idx}: duplicate 'uid' value '{uid}' (first seen at {seen_uids[uid]})"
                    )
                else:
                    seen_uids[uid] = f"instruction '{inst_id}' at index {idx}"

            # Validate required fields
            if "source" not in instruction:
                errors.append(f"Instruction '{inst_id}': must have a 'source' field")
            elif instruction["source"] not in ("user", "system"):
                errors.append(
                    f"Instruction '{inst_id}': invalid 'source' value '{instruction['source']}'. Must be 'user' or 'system'."
                )

            if "is_misalignment_check" not in instruction:
                errors.append(f"Instruction '{inst_id}': must have an 'is_misalignment_check' field")
            elif not isinstance(instruction["is_misalignment_check"], bool):
                errors.append(
                    f"Instruction '{inst_id}': 'is_misalignment_check' must be a boolean, got '{instruction['is_misalignment_check']}'."
                )

        for idx, item in enumerate(llm_judge):
            if not isinstance(item, dict):
                errors.append(f"llm_judge at index {idx}: must be an object")
                continue

            # Validate uid is present and unique
            uid = item.get("uid")
            if uid is None:
                errors.append(f"llm_judge at index {idx}: must have a 'uid' field")
            else:
                # Check for duplicate uid (across both instructions and llm_judge)
                if uid in seen_uids:
                    errors.append(
                        f"llm_judge at index {idx}: duplicate 'uid' value '{uid}' (first seen at {seen_uids[uid]})"
                    )
                else:
                    seen_uids[uid] = f"llm_judge at index {idx}"

            if "content" not in item:
                errors.append(f"llm_judge '{uid or idx}': must have a 'content' field")

            if "source" not in item:
                errors.append(f"llm_judge '{uid or idx}': must have a 'source' field")
            elif item.get("source") not in ("user", "system"):
                errors.append(
                    f"llm_judge '{uid or idx}': invalid 'source' value '{item.get('source')}'. Must be 'user' or 'system'."
                )

            if "is_misalignment_check" not in item:
                errors.append(f"llm_judge '{uid or idx}': must have an 'is_misalignment_check' field")
            elif not isinstance(item.get("is_misalignment_check"), bool):
                errors.append(
                    f"llm_judge '{uid or idx}': 'is_misalignment_check' must be a boolean, got '{item.get('is_misalignment_check')}'."
                )

        return errors

    async def verify(self, body: TuringVIFVerifyRequest) -> TuringVIFVerifyResponse:
        """
        Verify a response against all instructions.

        Runs fast validators synchronously and LLM validators in parallel
        using asyncio.gather for efficiency.

        Args:
            body: The verify request containing the response and instructions

        Returns:
            TuringVIFVerifyResponse with reward and validation details
        """
        # Extract the response text from the NeMoGymResponse
        final_response_text = ""
        if body.response.output:
            last_output = body.response.output[-1]
            if hasattr(last_output, "content") and last_output.content:
                final_response_text = last_output.content[0].text

        is_following_list: List[bool] = []
        validation_results: List[ValidationResult] = []

        # Validate schema first - if errors, skip this rollout
        all_instructions = {"instructions": [], "llm_judge": []}
        if body.instructions:
            all_instructions["instructions"] = body.instructions
        if body.llm_judge:
            # Convert LLMJudgeItem models to dicts for schema validation
            all_instructions["llm_judge"] = [item.model_dump() for item in body.llm_judge]

        schema_errors = await self.validate_instructions_schema(all_instructions)
        if schema_errors:
            # Return early with schema errors - rollout will be skipped
            for err in schema_errors:
                validation_results.append(
                    ValidationResult(
                        instruction="schema_validation",
                        status="Failed",
                        message=str(err),
                    )
                )
                is_following_list.append(False)

            return TuringVIFVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                follow_all_instructions=False,
                follow_instruction_list=is_following_list,
                validation_results=validation_results,
            )

        # Get language from request (defaults to "en")
        language = body.language if body.language in SUPPORTED_LANGS else "en"

        # Pre-validate: Check if all instructions are supported for this language
        unsupported_instructions = []
        for instruction in body.instructions:
            inst_id = instruction.get("instruction_id", "")
            if not is_instruction_supported(inst_id, language):
                unsupported_instructions.append(inst_id)

        if unsupported_instructions:
            # Skip rollout: record error for errors.json and do not add to processed.
            # The rollout collector treats this like schema_validation (skip + errors.json).
            validation_results.append(
                ValidationResult(
                    instruction="language_compatibility",
                    status="Failed",
                    message=f"Instructions not supported for language '{language}': {unsupported_instructions}",
                )
            )
            is_following_list.append(False)

            return TuringVIFVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                follow_all_instructions=False,
                follow_instruction_list=is_following_list,
                validation_results=validation_results,
            )

        # Separate fast validators from LLM validators
        fast_instructions = []
        llm_instructions = []

        for instruction in body.instructions:
            inst_id = instruction.get("instruction_id", "")
            if inst_id in LLM_INSTRUCTIONS:
                llm_instructions.append(instruction)
            else:
                fast_instructions.append(instruction)

        # Run fast validators synchronously (they're CPU-bound)
        for instruction in fast_instructions:
            inst_id = instruction.get("instruction_id", "")
            kwargs = {
                k: v
                for k, v in instruction.items()
                if k not in ("instruction_id", "uid", "source", "is_misalignment_check")
            }

            try:
                is_valid, message = validate_instruction(final_response_text, inst_id, kwargs, language=language)
            except Exception as e:
                is_valid, message = False, f"Validator error: {str(e)}"

            # Apply misalignment check if source="user" and is_misalignment_check=True
            if instruction.get("source") == "user" and instruction.get("is_misalignment_check") is True:
                is_valid, message = self.analyze_misalignment_check(is_valid, message)

            is_following_list.append(is_valid)
            validation_results.append(
                ValidationResult(instruction=inst_id, status="Passed" if is_valid else "Failed", message=message)
            )

        # Run LLM validators in parallel using asyncio.gather
        if llm_instructions:

            async def validate_llm_instruction(instruction: Dict[str, Any]) -> Tuple[str, bool, str, str, bool]:
                inst_id = instruction.get("instruction_id", "")
                source = instruction.get("source", "")
                is_misalignment = instruction.get("is_misalignment_check", False)
                kwargs = {
                    k: v
                    for k, v in instruction.items()
                    if k not in ("instruction_id", "uid", "source", "is_misalignment_check")
                }

                try:
                    is_valid, message = await self._validate_llm_instruction_async(
                        final_response_text, inst_id, kwargs
                    )
                except Exception as e:
                    is_valid, message = False, f"LLM validator error: {str(e)}"

                return inst_id, is_valid, message, source, is_misalignment

            llm_results = await asyncio.gather(*[validate_llm_instruction(inst) for inst in llm_instructions])

            for inst_id, is_valid, message, source, is_misalignment in llm_results:
                # Apply misalignment check if source="user" and is_misalignment_check=True
                if source == "user" and is_misalignment is True:
                    is_valid, message = self.analyze_misalignment_check(is_valid, message)

                is_following_list.append(is_valid)
                validation_results.append(
                    ValidationResult(instruction=inst_id, status="Passed" if is_valid else "Failed", message=message)
                )

        # Process custom LLM judge questions
        if body.llm_judge:

            async def validate_llm_judge_question(item: LLMJudgeItem) -> Tuple[str, bool, str, str, bool]:
                try:
                    judge_said_yes, message = await self._validate_custom_llm_judge_async(
                        final_response_text, item.content
                    )
                    # Compare judge verdict against expected pass_criteria
                    # If pass_criteria is "YES", judge must say YES for pass
                    # If pass_criteria is "NO", judge must say NO for pass (negate result)
                    if item.pass_criteria == "NO":
                        is_valid = not judge_said_yes
                    else:
                        is_valid = judge_said_yes
                except Exception as e:
                    is_valid, message = False, f"LLM judge error: {str(e)}"

                return f"llm_judge_{item.uid}", is_valid, message, item.source, item.is_misalignment_check

            judge_results = await asyncio.gather(*[validate_llm_judge_question(item) for item in body.llm_judge])

            for inst_id, is_valid, message, source, is_misalignment in judge_results:
                # Apply misalignment check if source="user" and is_misalignment_check=True
                if source == "user" and is_misalignment is True:
                    is_valid, message = self.analyze_misalignment_check(is_valid, message)

                is_following_list.append(is_valid)
                validation_results.append(
                    ValidationResult(instruction=inst_id, status="Passed" if is_valid else "Failed", message=message)
                )

        # Calculate overall success
        follow_all_instructions = all(is_following_list) if is_following_list else True

        return TuringVIFVerifyResponse(
            **body.model_dump(),
            reward=float(follow_all_instructions),
            follow_all_instructions=follow_all_instructions,
            follow_instruction_list=is_following_list,
            validation_results=validation_results,
        )


if __name__ == "__main__":
    TuringVIFResourcesServer.run_webserver()
