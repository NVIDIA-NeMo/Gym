"""
Turing VIF Resource Server for NeMo Gym.

This resource server integrates the Turing VIF (Verifiable Instruction Following)
validators into NeMo Gym's reinforcement learning framework. It supports both
fast rule-based validators and async LLM-based judge validators.
"""

import asyncio
import json
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
    from .vif_validators.validator import (
        validate_instruction,
        validate_instruction_schema,
        check_contradicting_instructions,
    )
    from .vif_validators.data_loader import (
        LLM_INSTRUCTIONS,
        JUDGE_SYSTEM_PROMPT,
        DEFINITION_GENERATOR_SYSTEM_PROMPT,
        LLM_JUDGE_QUESTION_PROMPT,
        eval_modes,
        inst_def,
        subinst_def,
    )
except ImportError:
    # When run directly (not as a module), add parent to path
    sys.path.insert(0, str(Path(__file__).parent))
    from vif_validators.validator import (
        validate_instruction,
        validate_instruction_schema,
        check_contradicting_instructions,
    )
    from vif_validators.data_loader import (
        LLM_INSTRUCTIONS,
        JUDGE_SYSTEM_PROMPT,
        DEFINITION_GENERATOR_SYSTEM_PROMPT,
        LLM_JUDGE_QUESTION_PROMPT,
        eval_modes,
        inst_def,
        subinst_def,
    )


# ============================================================================
# Configuration
# ============================================================================

class TuringVIFResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the Turing VIF Resource Server."""
    judge_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the LLM judge API. If not set, uses policy_base_url."
    )
    judge_api_key: Optional[str] = Field(
        default=None,
        description="API key for the LLM judge. If not set, uses policy_api_key."
    )
    judge_model: str = Field(
        default="gpt-4.1-2025-04-14",
        description="Model to use for LLM judge evaluations."
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


class TuringVIFRunRequest(BaseRunRequest):
    """Request model for the Turing VIF resource server."""
    id: int = Field(default=0, description="Request identifier")
    instructions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of instruction objects with instruction_id and kwargs"
    )
    llm_judge: List[LLMJudgeItem] = Field(
        default_factory=list,
        description="List of custom LLM judge questions"
    )
    prompt: Optional[str] = Field(
        default=None,
        description="The original user prompt"
    )


class TuringVIFVerifyRequest(TuringVIFRunRequest, BaseVerifyRequest):
    """Verify request combining run request with response."""
    pass


class ValidationResult(BaseModel):
    """Result of a single validation check."""
    instruction: str
    status: Literal["Passed", "Failed"]
    message: str


class TuringVIFVerifyResponse(BaseVerifyResponse):
    """Response from the verify endpoint."""
    follow_all_instructions: bool
    follow_instruction_list: List[bool]
    validation_results: List[ValidationResult] = Field(default_factory=list)


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
            base_url = self.config.judge_base_url or getattr(self.config, 'policy_base_url', 'https://api.openai.com/v1')
            api_key = self.config.judge_api_key or getattr(self.config, 'policy_api_key', '')
            
            self._judge_client = NeMoGymAsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        return self._judge_client

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    # ========================================================================
    # Async LLM Judge Functions
    # ========================================================================

    async def _judge_llm_api_async(
        self,
        user_content: str,
        system_content: str,
        temperature: float = 1.0,
        max_tokens: int = 10000
    ) -> str:
        """
        Async wrapper for LLM judge API calls using NeMoGymAsyncOpenAI.
        
        Automatically uses the Responses API for GPT-5/reasoning models,
        and Chat Completions API for standard chat models.
        
        Args:
            user_content: The user message content
            system_content: The system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            The LLM's response content as a string
        """
        client = self._get_judge_client()
        model = self.config.judge_model
        
        if self._is_reasoning_model(model):
            # Use Responses API for GPT-5 and other reasoning models
            result = await client.create_response(
                model=model,
                input=[
                    {"role": "developer", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                max_output_tokens=max_tokens,
                # Note: temperature is not supported for reasoning models
            )
            
            # Extract text from Responses API output format
            output_text = ""
            for output_item in result.get("output", []):
                if output_item.get("type") == "message":
                    for content_item in output_item.get("content", []):
                        if content_item.get("type") == "output_text":
                            output_text += content_item.get("text", "")
            return output_text
        else:
            # Use Chat Completions API for standard models
            result = await client.create_chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            return result["choices"][0]["message"]["content"]

    async def _validate_custom_llm_judge_async(
        self,
        response: str,
        question_text: str
    ) -> Tuple[bool, str]:
        """
        Validates a response against a free-form LLM Judge question.
        
        Args:
            response: The model response to evaluate
            question_text: The question to evaluate against
            
        Returns:
            Tuple of (is_valid, reasoning)
        """
        try:
            judge_prompt = LLM_JUDGE_QUESTION_PROMPT.format(
                question=question_text,
                model_response=response
            )

            evaluation = await self._judge_llm_api_async(
                user_content="Evaluate the response.",
                system_content=judge_prompt
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
                return False, f"Judge returned input format instead of output format."
            
            judge_response = JudgeResponse(**json_data)
            flag = (judge_response.verdict == "YES")
            return flag, judge_response.reasoning

        except (json.JSONDecodeError, ValidationError) as e:
            return False, f"Error parsing Judge response: {e}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    async def _get_dynamic_definition_async(
        self,
        inst_type: str,
        term: str
    ) -> Tuple[str, bool]:
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
                instruction=instruction_name,
                inst_label=inst_type,
                term=term,
                context_related_terms=context_terms_str
            )

            response_str = await self._judge_llm_api_async(
                user_content=f"Define the term: {term}",
                system_content=system_prompt
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
        self,
        response: str,
        inst_type: str,
        kwargs: Dict[str, Any]
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
                            definition, is_valid = await self._get_dynamic_definition_async(
                                inst_type, arg_value_str
                            )

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
            flag = (judge_response.verdict == "YES")
            return (flag, judge_response.reasoning)

        except (json.JSONDecodeError, ValidationError) as e:
            return (False, f"Error parsing LLM Judge response: {e}")
        except Exception as e:
            return (False, f"Validation error: {str(e)}")

    # ========================================================================
    # Main Verify Endpoint
    # ========================================================================

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
            kwargs = {k: v for k, v in instruction.items() if k != "instruction_id"}
            
            try:
                is_valid, message = validate_instruction(final_response_text, inst_id, kwargs)
            except Exception as e:
                is_valid, message = False, f"Validator error: {str(e)}"

            is_following_list.append(is_valid)
            validation_results.append(ValidationResult(
                instruction=inst_id,
                status="Passed" if is_valid else "Failed",
                message=message
            ))

        # Run LLM validators in parallel using asyncio.gather
        if llm_instructions:
            async def validate_llm_instruction(instruction: Dict[str, Any]) -> Tuple[str, bool, str]:
                inst_id = instruction.get("instruction_id", "")
                kwargs = {k: v for k, v in instruction.items() if k != "instruction_id"}
                
                try:
                    is_valid, message = await self._validate_llm_instruction_async(
                        final_response_text, inst_id, kwargs
                    )
                except Exception as e:
                    is_valid, message = False, f"LLM validator error: {str(e)}"
                
                return inst_id, is_valid, message

            llm_results = await asyncio.gather(
                *[validate_llm_instruction(inst) for inst in llm_instructions]
            )

            for inst_id, is_valid, message in llm_results:
                is_following_list.append(is_valid)
                validation_results.append(ValidationResult(
                    instruction=inst_id,
                    status="Passed" if is_valid else "Failed",
                    message=message
                ))

        # Process custom LLM judge questions
        if body.llm_judge:
            async def validate_llm_judge_question(item: LLMJudgeItem) -> Tuple[str, bool, str]:
                try:
                    is_valid, message = await self._validate_custom_llm_judge_async(
                        final_response_text, item.content
                    )
                except Exception as e:
                    is_valid, message = False, f"LLM judge error: {str(e)}"
                
                return f"llm_judge_{item.uid}", is_valid, message

            judge_results = await asyncio.gather(
                *[validate_llm_judge_question(item) for item in body.llm_judge]
            )

            for inst_id, is_valid, message in judge_results:
                is_following_list.append(is_valid)
                validation_results.append(ValidationResult(
                    instruction=inst_id,
                    status="Passed" if is_valid else "Failed",
                    message=message
                ))

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

