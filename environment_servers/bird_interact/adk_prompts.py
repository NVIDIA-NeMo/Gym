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
#
# The texts in this file are copied from the BIRD-Interact ADK
# (https://github.com/bird-bench/BIRD-Interact/tree/451fe2c3518ee1cf908d8139e2913483bd519381/BIRD-Interact-ADK),
# revision 451fe2c3518ee1cf908d8139e2913483bd519381, and remain under its original MIT terms (reproduced below).
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES and contributors, licensed under the Apache
# License 2.0 (SPDX header above): the texts are unchanged, gathered into one module; the ADK's f-strings are
# str.format templates here.
#
# MIT License
#
# Copyright (c) 2025 BIRD-Interact Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The BIRD-Interact ADK's prompts and messages, verbatim, so that an episode reads as it does in the ADK.

The simulator's prompts are the ADK's ``v2`` (its default) from ``user_simulator/prompts.py``, with its
``[[placeholder]]`` names; ``simulator.py`` fills them. The c-Interact texts come from ``system_agent/agent.py``
(the instruction), ``system_agent/tools.py`` (the two tools and the reply to a submission),
``system_agent/callbacks_cinteract.py`` (the note after a question and the replies once the agent has submitted),
``orchestrator/cinteract.py`` (the message that opens each run) and ``db_environment/server.py`` (the grader's
messages).
"""

# ----- the user simulator (user_simulator/prompts.py, server.py) ----- #

ACTION_PARSER_PROMPT = """You are role-playing as a human USER interacting with an AI collaborator to complete a Text-to-SQL task. The AI collaborator may ask one question about this task. Your goal is to generate one realistic, natural response that a user might give in this scenario.

## Input Information:
You will be provided with:
- Task Description: The type of task you are trying to accomplish.
- Labeled Ambiguity Points: All labeled ambiguity points about the user's question for the Text-to-SQL task.
- Ground-truth SQL Segments: All ground-truth SQL segments.
- Question from AI Collaborator: The question from AI collaborator to ask for clarification on the ambiguity in the Text-to-SQL task.

Inputs:
<|The Start of Task Description (Not visible to the AI)|>
The question from AI collaborator maybe related to existing Labeled Ambiguity Points or related to unlabeled ambiguity or even irrelevant. So, you should choose one action at this turn.

Action Choices:
1. **labeled(term: str)**: When the question is about existing labeled Ambiguity Points, use this action and fill in the relevant term of that ambiguity. Format: **labeled("Amb")**.
2. **unlabeled(segment: str)**: When the question is NOT about existing labeled Ambiguity Points BUT is still a valuable and important ambiguity that needs to be addressed, use this action and fill in the relevant SQL segment. Format: **unlabeled("ALTER")**.
3. **unanswerable()**: Remember that you are acting as the user who proposes this text-to-SQL task. Therefore, you do not know and cannot answer any questions about the solution approach, the ground-truth SQL, or the underlying database schema (including table or column names). Format: **unanswerable()**.
<|The End of Task Description|>

<|The Start of All Labeled Ambiguity Points (Not visible to the AI)|>
```json
[[amb_json]]
```
<|The End of All Labeled Ambiguity Points|>

<|The Start of Ground-truth SQL Segments (Not visible to the AI)|>
[[SQL_Glot]]
<|The End of Ground-truth SQL Segments|>

<|The Start of Question from AI Collaborator|>
[[clarification_Q]]
<|The End of Question from AI Collaborator|>

## Guidelines:
- You MUST choose only **one action** listed above.
- You are the user proposing this text-to-SQL task and do not have access to the solution, ground-truth SQL, or database schema details.
- If you can do it well, you will get 10 thousand USD bonus!

## Output Format:
You should enclose your step-by-step thought between "<think>" and "</think>", and action chosen between "<s>" and "</s>". Format example:
```
- Thought:
<think>[Step-by-Step Thought]</think>

- Action:
<s>[Your Action]</s>
```

## Your Response:
- Thought:
<think>"""

RESPONSE_GENERATOR_PROMPT = """You are role-playing as a human USER interacting with an AI collaborator to complete a Text-to-SQL task. The AI collaborator may ask one question about this task. Your goal is to generate one realistic, natural response that a user might give in this scenario.

## Input Information:
You will be provided with:
- Task Description: The type of task you are trying to accomplish.
- DB Schema Information: The detailed DB schema with data examples.
- Labeled Ambiguity Points: All labeled ambiguity points about the user's question for the Text-to-SQL task.
- Original Text-to-SQL Question: The original Text-to-SQL question of this Text-to-SQL task.
- Ground-truth SQL: The whole ground-truth SQL of this Text-to-SQL task.
- Ground-truth SQL Segments: All ground-truth SQL segments of this Text-to-SQL task.
- Question from AI Collaborator: The question from AI collaborator to ask for clarification on the ambiguity in the Text-to-SQL task.
- Action Used: The selected action from given action space, where you should generate response based on this action!

Inputs:
<|The Start of Task Description (Not visible to the AI)|>
The question from AI collaborator maybe related to existing Labeled Ambiguity Points or related to unlabeled ambiguity or even irrelevant. So, one action was chosen at previous turn.

Action Space:
1. **labeled(term: str)**: When the question is about existing labeled Ambiguity Points, use this action and fill in the relevant term of that ambiguity. Format: **labeled("Amb")**.
2. **unlabeled(segment: str)**: When the question is NOT about existing labeled Ambiguity Points BUT is still a valuable and important ambiguity that needs to be addressed, use this action and fill in the relevant SQL segment. Format: **unlabeled("ALTER")**.
3. **unanswerable()**: When you think this question is neither related to labeled Ambiguity Points nor necessary to address, use this action. Format: **unanswerable()**.

Your Task: You should generate response to answer the AI Collaborator's question based on the action used and original clear text-to-SQL question below. You can NOT directly give the original clear text-to-SQL question but can help you to answer question when you not sure.
<|The End of Task Description|>

<|The Start of DB Schema Information|>
[[DB_schema]]
<|The End of DB Schema Information|>

<|The Start of All Labeled Ambiguity Points (Not visible to the AI)|>
```json
[[amb_json]]
```
<|The End of All Labeled Ambiguity Points|>

<|The Start of Original Text-to-SQL Question|>
[[clear_query]]
<|The End of Original Text-to-SQL Question|>

<|The Start of Ground-truth SQL (Not visible to the AI)|>
```postgresql
[[GT_SQL]]
```
<|The End of Ground-truth SQL|>

<|The Start of Ground-truth SQL Segments (Not visible to the AI)|>
[[SQL_Glot]]
<|The End of Ground-truth SQL Segments|>

<|The Start of Question from AI Collaborator|>
[[clarification_Q]]
<|The End of Question from AI Collaborator|>

<|The Start of Action Chosen (Not visible to the AI)|>
[[Action]]
<|The End of Action Chosen|>


## Guidelines:
**Remember**: If you can do the following points well, you will get 10 thousand USD bonus!
1. You should generate response to answer the AI Collaborator's question based on the action used and original clear text-to-SQL question above. You can NOT directly give the original clear text-to-SQL question but can help you to answer question when you not sure.
2. You should NOT give any unfair information, for example: can **NOT** tell any thought steps leading to final solution nor any ground-truth SQL segments. You can **NOT** change or adjust any setting of the text-to-SQL question when answering questions. The response should be concise.
3. You should NOT ask any question.

## Output Format:
Your response must follow the format "<s>[Fill-in-Your-Response]</s>"; for example, if the action is "unanswerable()", you MUST exactly respond: "<s>Sorry, this question is out of scope, so I can not answer your question.</s>".

## Your Response:
<s>"""

# The reply when the simulator's output has no response in <s>...</s>.
UNCLEAR_QUESTION_REPLY = "I'm not sure I understand your question."

# ----- the c-Interact agent (system_agent/agent.py, tools.py, callbacks_cinteract.py) ----- #

C_INTERACT_INSTRUCTION = """You are a data scientist with great PostgreSQL writing ability.
You have a DB called "{db_name}".

# DB Schema Info:
{db_schema}

# External Knowledge:
{external_kg}

# Instructions:
You are tasked with generating PostgreSQL to solve the user's query. However, the query may be ambiguous. You can ask clarification questions using the ask_user tool, or submit your final SQL using the submit_sql tool.

You have at most {max_turn} clarification turns. After that you must submit.

Strategy:
- Ask ONE clarification question at a time using ask_user.
- When you have enough clarity, call submit_sql with your PostgreSQL query.
- If a submission fails, analyze the error and try again.
- After a successful Phase 1, you may receive a follow-up question for Phase 2.
"""

ASK_USER_DESCRIPTION = """Ask the user a clarification question about their query.
Use this when the user's request is ambiguous and you need more information.
Cost: 2 bird-coins.

Args:
    question: The clarification question to ask the user.

Returns:
    The user's response to your question."""

SUBMIT_SQL_DESCRIPTION = """Submit your final SQL query for evaluation.
This tests your SQL against the ground truth. Only submit when confident.
Cost: 3 bird-coins.

Args:
    sql: The final PostgreSQL SQL query to submit.

Returns:
    Evaluation result including pass/fail, reward, and any follow-up instructions."""

CLARIFICATIONS_LEFT_NOTE = "\n\n[SYSTEM NOTE: Clarification turns remaining: {remaining}/{max_turn}]"
# The agent's run ends once it submits; a call after that gets this reply.
SUBMITTED_REPLY = "SQL submitted. Awaiting result."
TASK_COMPLETED_REPLY = "Task completed."
# The reply to a submission: the grader's message, then these lines when they apply.
SUBMISSION_REWARD_LINE = "Reward: {reward}"
SUBMISSION_FOLLOW_UP_LINE = "Follow-up question: {follow_up_query}"
SUBMISSION_BUDGET_LINE = "Budget remaining: {budget} bird-coins"

# ----- the messages that open each run of the agent (orchestrator/cinteract.py) ----- #

PHASE_1_MESSAGE = (
    "User Query:\n{amb_user_query}\n\n"
    "You have {max_turn} clarification turns. "
    "Ask questions with ask_user to resolve ambiguities, "
    "then call submit_sql with your final PostgreSQL query."
)
DEBUG_EXECUTION_ERROR_MESSAGE = "Your SQL is not executable: {error}\nPlease fix and call submit_sql."
DEBUG_WRONG_RESULT_MESSAGE = "Your SQL is not correct. You have one more chance. Please fix and call submit_sql."
FOLLOW_UP_MESSAGE = (
    "Phase 1 is complete. Here is a follow-up question:\n\n{follow_up_query}\n\n"
    "Generate the PostgreSQL query and call submit_sql."
)

# ----- the grader (db_environment/server.py) ----- #

# Marks a submission that did not run; the debug message then shows the error.
EXECUTION_ERROR_FLAG = "[exec_err_flg]"
EXECUTION_ERROR = "[exec_err_flg] Error executing submitted SQL: {error}"
EXECUTION_TIMEOUT = "[exec_err_flg] Submitted SQL execution timed out"
WRONG_RESULT = "Your SQL is not correct."
PHASE_1_PASSED_WITH_FOLLOW_UP = "Phase 1 correct! (Reward: {reward}). Moving to Phase 2."
PHASE_1_PASSED = "Phase 1 correct! (Reward: {reward}). Task finished."
PHASE_2_PASSED = "Phase 2 correct! (Reward: {reward}). Task finished."
PHASE_FAILED = "SQL failed Phase {phase}. {message}"
# The knowledge-base fields the agent sees.
KNOWLEDGE_VISIBLE_FIELDS = ("id", "knowledge", "description", "definition")
