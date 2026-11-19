# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BIRD-Interact ADK's c-Interact texts that the agent reads, as the bird_interact Environment Server sends them.

Copies of ``environment_servers/bird_interact/adk_prompts.py``, which this agent's tests cannot import: the agent does
not depend on that server. ``tests/unit_tests/benchmarks/test_bird_interact_auto_ontology.py`` checks that the
agent reads the server's own texts.
"""

C_INTERACT_INSTRUCTION = 'You are a data scientist with great PostgreSQL writing ability.\nYou have a DB called "{db_name}".\n\n# DB Schema Info:\n{db_schema}\n\n# External Knowledge:\n{external_kg}\n\n# Instructions:\nYou are tasked with generating PostgreSQL to solve the user\'s query. However, the query may be ambiguous. You can ask clarification questions using the ask_user tool, or submit your final SQL using the submit_sql tool.\n\nYou have at most {max_turn} clarification turns. After that you must submit.\n\nStrategy:\n- Ask ONE clarification question at a time using ask_user.\n- When you have enough clarity, call submit_sql with your PostgreSQL query.\n- If a submission fails, analyze the error and try again.\n- After a successful Phase 1, you may receive a follow-up question for Phase 2.\n'

PHASE_1_MESSAGE = "User Query:\n{amb_user_query}\n\nYou have {max_turn} clarification turns. Ask questions with ask_user to resolve ambiguities, then call submit_sql with your final PostgreSQL query."

DEBUG_EXECUTION_ERROR_MESSAGE = "Your SQL is not executable: {error}\nPlease fix and call submit_sql."

DEBUG_WRONG_RESULT_MESSAGE = "Your SQL is not correct. You have one more chance. Please fix and call submit_sql."

FOLLOW_UP_MESSAGE = "Phase 1 is complete. Here is a follow-up question:\n\n{follow_up_query}\n\nGenerate the PostgreSQL query and call submit_sql."

CLARIFICATIONS_LEFT_NOTE = "\n\n[SYSTEM NOTE: Clarification turns remaining: {remaining}/{max_turn}]"
