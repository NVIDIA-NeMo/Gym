from . import validator as base_validator
from . import validator_en as english_validator

# Wrapper to keep explicit English usage
def validate_instruction(response, inst_type, kwargs, all_instructions=None):
    return english_validator.validate_instruction(response, inst_type, kwargs, all_instructions)


def validate_llm_instruction(response, inst_type, kwargs, all_instructions=None, definition_cache=None):
    return english_validator.validate_llm_instruction(response, inst_type, kwargs, all_instructions, definition_cache)


def validate_prompt_against_instructions(user_prompt, turn_instructions):
    return english_validator.validate_prompt_against_instructions(user_prompt, turn_instructions)
