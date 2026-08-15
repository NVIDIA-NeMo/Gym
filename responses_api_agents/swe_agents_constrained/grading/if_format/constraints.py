from enum import Enum
from pydantic import BaseModel, Field


class ConstraintScope(str, Enum):
    ALL_STEPS = "all_steps"
    REASONING_STEPS = "reasoning_steps"
    CODE_STEPS = "code_steps"
    AFTER_TOOL_CALL = "after_tool_call"
    # Assistant message immediately PRECEDING a tool call. Distinct from
    # ALL_STEPS: the synthesized "Action: <name>" step and the environment's
    # observation can never carry model-authored preamble text.
    BEFORE_TOOL_CALL = "before_tool_call"
    FIRST_STEP_ONLY = "first_step_only"
    FINAL_OUTPUT = "final_output"


class FormatRegime(str, Enum):
    """Interaction format a constraint's compliance semantics assume.

    A constraint is only injectable into rollouts whose harness produces the
    format it governs — e.g. 'Action Input:' rules are meaningless (vacuously
    satisfied or impossible) under native tool calling, and constraints that
    require a user's mid-episode reply cannot fire in single-agent rollouts.
    Screening (G2) must reject pairs whose task rollout regime is not covered.

    ANY               — format-agnostic (plain text conventions, final output).
    TEXT_RESPONSE     — single completion / final text answer.
    TEXT_REACT        — textual ReAct loops ('Thought:'/'Action:'/'Action Input:').
    NATIVE_TOOL_CALL  — structured tool_calls via the chat/responses API.
    BASH_SCAFFOLD     — mini-swe-style bash-loop agents.
    USER_INTERACTIVE  — requires real/simulated user turns mid-episode.
    ENV_REWARD_VISIBLE— requires the environment to reveal per-step rewards.
    """
    ANY = "any"
    TEXT_RESPONSE = "text_response"
    TEXT_REACT = "text_react"
    NATIVE_TOOL_CALL = "native_tool_call"
    BASH_SCAFFOLD = "bash_scaffold"
    USER_INTERACTIVE = "user_interactive"
    ENV_REWARD_VISIBLE = "env_reward_visible"


class InjectionMode(str, Enum):
    """Where the constraint instruction is placed in the conversation.

    SYSTEM_PROMPT    — constraint in the system prompt; model sees it from step 0.
    FIRST_USER_TURN  — constraint appended/prepended to the first user message.
    MID_CONVERSATION — constraint injected at conversation turn N (via injection_turn
                       in verifier_metadata); model compliance is only evaluated for
                       steps that occur AFTER that turn. This tests the harder capability
                       of updating behavior mid-task based on a late instruction.

    Interaction with ConstraintScope:
        scope filters WHICH steps are relevant to a constraint.
        injection_mode + injection_turn filter WHICH steps the model was
        actually aware of the constraint — steps before the injection turn
        are never penalised regardless of scope.

    Constraints with scope=FIRST_STEP_ONLY are incompatible with MID_CONVERSATION
    because the first step has already occurred before the instruction arrives.
    """
    SYSTEM_PROMPT = "system_prompt"
    FIRST_USER_TURN = "first_user_turn"
    MID_CONVERSATION = "mid_conversation"


_ALL_INJECTION_MODES: list[InjectionMode] = list(InjectionMode)
_EARLY_ONLY_MODES: list[InjectionMode] = [
    InjectionMode.SYSTEM_PROMPT,
    InjectionMode.FIRST_USER_TURN,
]


class AgenticConstraintType(str, Enum):
    """Format constraints that govern how the model operates through an agentic trajectory.

    These apply to intermediate reasoning blocks, tool calls, and structured outputs
    — not just the final answer. Organised by domain origin.
    """
    # ── Core agentic (original) ──────────────────────────────────────────────
    # NOTE: thinking_tags was removed 2026-07-30. Reasoning placement is an API
    # transport concern, not a user-requested output format; see agent.md
    # "Constraints must be things a user would actually ask for".
    UNIFIED_DIFF = "unified_diff"
    ACTION_LOG_JSON = "action_log_json"
    NUMBERED_PLAN = "numbered_plan"
    FILE_PATH_BEFORE_CODE = "file_path_before_code"
    STEP_SUMMARY_PREFIX = "step_summary_prefix"
    SCOPE_CONSTRAINT = "scope_constraint"
    JSON_ERROR_REPORTING = "json_error_reporting"
    HANDOFF_SCHEMA = "handoff_schema"
    OUTPUT_SECTIONS = "output_sections"

    # ── Software engineering agent ────────────────────────────────────────────
    NO_FORCE_GIT_COMMANDS = "no_force_git_commands"
    PR_DESCRIPTION_SECTIONS = "pr_description_sections"

    # ── RAG / synthesis agent ─────────────────────────────────────────────────
    CITATION_AFTER_CLAIM = "citation_after_claim"
    RETRIEVAL_IDS_BEFORE_SYNTHESIS = "retrieval_ids_before_synthesis"
    UNCERTAINTY_FLAG = "uncertainty_flag"

    # ── Data pipeline agent ───────────────────────────────────────────────────
    SQL_EXPLAIN_BEFORE_DML = "sql_explain_before_dml"
    DRY_RUN_BEFORE_EXECUTE = "dry_run_before_execute"

    # ── Security audit agent ──────────────────────────────────────────────────
    SEVERITY_ENUM = "severity_enum"
    CVE_FIELDS_REQUIRED = "cve_fields_required"

    # ── Multi-agent orchestrator ──────────────────────────────────────────────
    SUBTASK_ID_ASSIGNED = "subtask_id_assigned"

    # ── DevOps agent ──────────────────────────────────────────────────────────
    INCIDENT_PRIORITY_TAGGED = "incident_priority_tagged"
    IMPACT_BEFORE_REMEDIATION = "impact_before_remediation"
    TIMESTAMP_ISO8601 = "timestamp_iso8601"

    # ── Document processing agent ─────────────────────────────────────────────
    PAGE_REF_IN_EXTRACTION = "page_ref_in_extraction"

    # ── Customer support agent ────────────────────────────────────────────────
    TICKET_ID_IN_ALL_STEPS = "ticket_id_in_all_steps"

    # ── Tool use discipline (discovered) ─────────────────────────────────────
    TOOL_CALL_INTENT_TAG = "tool_call_intent_tag"
    DIFF_STAT_AFTER_EDIT = "diff_stat_after_edit"
    COMMAND_EXIT_CODE_REPORTED = "command_exit_code_reported"

    # ── Safety and compliance (discovered) ───────────────────────────────────
    ROLLBACK_COMMAND_BEFORE_DEPLOY = "rollback_command_before_deploy"
    NO_SECRET_LITERALS_IN_CODE = "no_secret_literals_in_code"
    PII_MASKED_IN_TRANSCRIPT = "pii_masked_in_transcript"
    ENV_TAG_ON_EVERY_COMMAND = "env_tag_on_every_command"
    KUBECTL_NAMESPACE_EXPLICIT = "kubectl_namespace_explicit"

    # ── Software engineering hygiene (discovered) ─────────────────────────────
    TEST_COMMAND_BEFORE_PATCH = "test_command_before_patch"
    BRANCH_NAME_CONVENTION = "branch_name_convention"

    # ── SWE-bench multi-turn (generated, screened against swe_agents) ─────────
    # Trajectory-level constraints: most read prior_steps, so they only score
    # meaningfully once the grader receives the full trajectory rather than a patch.
    EXPECTED_ACTUAL_ERROR_BLOCK = "expected_actual_error_block"
    EXPLICIT_TEST_TARGET_REQUIRED = "explicit_test_target_required"
    EXPLICIT_TEST_SELECTION_ARGS = "explicit_test_selection_args"
    PYTEST_TARGET_SCOPED = "pytest_target_scoped"
    EDITS_VIA_EDIT_TOOL_ONLY = "edits_via_edit_tool_only"
    EXPECTATION_BEFORE_RUN_CHECK_AFTER = "expectation_before_run_check_after"
    FAILING_TEST_ID_ENUMERATION = "failing_test_id_enumeration"
    REREAD_BEFORE_EDIT_RETRY = "reread_before_edit_retry"
    STRAY_FILE_AUDIT_LINE = "stray_file_audit_line"
    SCRATCH_FILE_LEDGER = "scratch_file_ledger"

    # ── Multi-agent orchestrator extended (discovered) ────────────────────────
    DELEGATION_BUDGET_FIELD = "delegation_budget_field"
    RETRY_ATTEMPT_COUNTER = "retry_attempt_counter"

    # ── ReAct / API orchestration format (from ToolBench benchmark) ───────────
    REACT_STEP_INDEX_MONOTONIC = "react_step_index_monotonic"
    ACTION_INPUT_STRICT_JSON = "action_input_strict_json"
    API_CATEGORY_TAG_PER_ACTION = "api_category_tag_per_action"

    # ── Function calling protocol (from BFCL / NexusRaven / AgentInstruct) ───
    IRRELEVANCE_SENTINEL_LINE = "irrelevance_sentinel_line"
    UNAVAILABLE_TOOL_DECLARATION = "unavailable_tool_declaration"
    EXTRA_PARAM_REJECTION_LINE = "extra_param_rejection_line"
    MISSING_PARAM_QUESTION_BLOCK = "missing_param_question_block"
    NESTED_CALL_INLINE_SYNTAX = "nested_call_inline_syntax"
    PARALLEL_GROUP_FANOUT_DECLARATION = "parallel_group_fanout_declaration"
    ARG_PROVENANCE_MAP = "arg_provenance_map"
    FORBIDDEN_TOOL_ABSTENTION = "forbidden_tool_abstention"

    # ── Repository repair discipline (from SWE-bench benchmark) ──────────────
    ANCHOR_COMMIT_DECLARED_FIRST = "anchor_commit_declared_first"
    READ_BEFORE_EDIT_CITATION = "read_before_edit_citation"
    HYPOTHESIS_BEFORE_EDIT_TAG = "hypothesis_before_edit_tag"

    # ── Customer service compliance (from τ-bench benchmark) ──────────────────
    CONFIRMATION_GATE_TOKEN = "confirmation_gate_token"
    POLICY_CLAUSE_CITE_ON_REFUSAL = "policy_clause_cite_on_refusal"

    # ── Code documentation format (from FOFO / AgentInstruct) ─────────────────
    DOCSTRING_SECTION_ORDER_FIXED = "docstring_section_order_fixed"

    # ── Progress reporting format ──────────────────────────────────────────────
    MONOTONIC_STEP_INDEX_HEADER = "monotonic_step_index_header"

    # ── RL environment interaction (from EnvFactory-RL) ───────────────────────
    RL_REWARD_REPORTED = "rl_reward_reported"

    # ── Agentic coding discipline (from customer_cursor / customer_coderabbit) ──
    # Grounded in real customer traces; see feedbacks/customer_cursor/mapping.md
    # and feedbacks/customer_coderabbit/mapping.md.
    CODE_CITE_LINE_RANGE_FORMAT = "code_cite_line_range_format"
    NO_TRAILING_COLON_BEFORE_TOOL = "no_trailing_colon_before_tool"
    APPROVAL_BODY_EXACT_LITERAL = "approval_body_exact_literal"
    REVIEW_FILE_WRAP_MARKERS = "review_file_wrap_markers"

    # ── SWE-bench batch 2 (curated 2026-08-11: rescreen + repair + verifier-first;
    #    see rubrics/audits/swebench_candidate_curation.json) ──────────────────
    # First-message discipline
    OPENING_TRIAGE_ENUM_LINE = "opening_triage_enum_line"
    NO_SIMULATED_TOOL_OUTPUT_IN_OPENING = "no_simulated_tool_output_in_opening"
    NO_OUTCOME_CLAIMS_IN_OPENING = "no_outcome_claims_in_opening"
    ORIENTATION_MESSAGE_PRECEDES_FIRST_TOOL_CALL = "orientation_message_precedes_first_tool_call"
    SEARCH_BEFORE_FIRST_READ = "search_before_first_read"
    # Tool-call argument discipline
    TIMEOUT_WRAPPED_EXECUTION = "timeout_wrapped_execution"
    GREP_SCOPED_AND_NUMBERED = "grep_scoped_and_numbered"
    NONINTERACTIVE_COMMAND_DISCIPLINE = "noninteractive_command_discipline"
    REPRO_SCRIPT_SANDBOX_PATH = "repro_script_sandbox_path"
    REMOVAL_INTENT_TAG = "removal_intent_tag"
    GIT_SUBCOMMAND_MODE_DECLARATION = "git_subcommand_mode_declaration"
    CONFIG_FILE_EDIT_DECLARATION_TAG = "config_file_edit_declaration_tag"
    OUT_OF_REPO_PATH_ACCESS_TAG = "out_of_repo_path_access_tag"
    # Post-observation reporting
    TEST_TALLY_LINE_AFTER_RUN = "test_tally_line_after_run"
    FAILURE_CLASS_ENUM_TAG = "failure_class_enum_tag"
    LARGE_OBSERVATION_FOCUS_LINE = "large_observation_focus_line"
    # Final-message deliverables
    FINAL_TEST_LEDGER_JSON_BLOCK = "final_test_ledger_json_block"
    CHANGED_FILES_MANIFEST_FINAL = "changed_files_manifest_final"
    IMPACT_ASSESSMENT_FINAL_LINE = "impact_assessment_final_line"
    EDGE_CASE_CHECKLIST_BLOCK = "edge_case_checklist_block"
    # Cross-turn state
    ISSUE_SUMMARY_VERBATIM_ECHO = "issue_summary_verbatim_echo"
    STATE_LEDGER_MONOTONIC_CARRYOVER = "state_ledger_monotonic_carryover"
    CUMULATIVE_TOUCHED_FILES_MANIFEST = "cumulative_touched_files_manifest"
    PATCH_REVISION_COUNTER_PER_FILE = "patch_revision_counter_per_file"
    CHECKPOINT_EVERY_NTH_TOOL_CALL = "checkpoint_every_nth_tool_call"
    DUPLICATE_COMMAND_RERUN_TAG = "duplicate_command_rerun_tag"
    SINGLE_TOOL_CALL_PER_MESSAGE = "single_tool_call_per_message"
    PHASE_TAG_ORDERED_LIFECYCLE = "phase_tag_ordered_lifecycle"
    VERIFICATION_CALL_AFTER_EACH_EDIT = "verification_call_after_each_edit"
    PRE_FIRST_EDIT_CALL_TALLY_ONCE = "pre_first_edit_call_tally_once"
    # Narration grounding
    SUCCESS_CLAIM_OBSERVATION_QUOTE = "success_claim_observation_quote"
    PREEXISTING_FAILURE_BASELINE_CITATION = "preexisting_failure_baseline_citation"
    NO_USER_QUESTIONS_ASSUMPTION_TAG = "no_user_questions_assumption_tag"
    REPO_RELATIVE_PATHS_IN_NARRATION = "repo_relative_paths_in_narration"

    # ── SWE-bench batch 3 (2026-08-12: repairs of 3-criteria calibration
    #    failures; see artifacts/calibration/applicability_2026-08-12_seed42.json.
    #    Each replaces a batch-2 constraint whose trigger was too rare (match)
    #    or that no unconstrained agent could violate (activates)) ────────────
    NO_OUTCOME_CLAIMS_BEFORE_EXECUTION = "no_outcome_claims_before_execution"
    EDIT_CLASS_DECLARATION_TAG = "edit_class_declaration_tag"
    ABS_PATH_SCOPE_TAG = "abs_path_scope_tag"
    RAW_OUTPUT_QUARANTINE = "raw_output_quarantine"

    # ── Real-traffic coverage batch (2026-08-14: Fay Wang's fc.1.1 dataset,
    #    364 audited failures, + kernelbench NVBug case; per-pattern grounding in
    #    reports/real_traffic_if_format_coverage.md — trace IDs cited per entry) ──
    EXACT_SENTINEL_REPLY = "exact_sentinel_reply"
    CLOSED_TAG_VERDICT_REPLY = "closed_tag_verdict_reply"
    TAGGED_SECTIONS_WELL_FORMED = "tagged_sections_well_formed"
    OUTPUT_ONLY_PASSTHROUGH = "output_only_passthrough"
    CONTINUATION_NO_RESTART = "continuation_no_restart"
    CONDITIONAL_REQUIRED_SENTENCE = "conditional_required_sentence"
    ABS_PATHS_IN_FINAL_RESPONSE = "abs_paths_in_final_response"


class ConversationalConstraintType(str, Enum):
    """Format constraints on text responses within conversational turns.

    Original set sourced from IFEval (29 types) and IFBench training set.
    Extended set discovered via domain sweep — all rule-based.
    """
    # ── Original (IFEval / IFBench) ───────────────────────────────────────────
    WORD_COUNT_MAX = "word_count_max"
    WORD_COUNT_MIN = "word_count_min"
    JSON_FORMAT = "json_format"
    BULLET_LIST = "bullet_list"
    NUMBERED_LIST = "numbered_list"
    LANGUAGE = "language"
    KEYWORD_INCLUDE = "keyword_include"
    KEYWORD_FORBIDDEN = "keyword_forbidden"
    SECTION_HEADERS = "section_headers"
    SENTENCE_COUNT = "sentence_count"
    RESPONSE_PREFIX = "response_prefix"
    RESPONSE_SUFFIX = "response_suffix"

    # ── Extended (domain sweep) ───────────────────────────────────────────────
    TABLE_FORMAT_REQUIRED = "table_format_required"
    CODE_BLOCK_LANGUAGE_TAG = "code_block_language_tag"
    MAX_LIST_NESTING_DEPTH = "max_list_nesting_depth"
    NO_CONTRACTIONS = "no_contractions"
    ACTION_ITEMS_CHECKBOX = "action_items_checkbox"
    TLDR_PREFIX = "tldr_prefix"
    CONFIDENCE_LEVEL_SUFFIX = "confidence_level_suffix"
    MAX_SENTENCE_LENGTH = "max_sentence_length"

    # ── Real-traffic coverage batch (2026-08-14: Fay Wang's fc.1.1 dataset +
    #    kernelbench; see reports/real_traffic_if_format_coverage.md) ────────────
    RESPONSE_LINE_LIMIT = "response_line_limit"
    NO_PREAMBLE_POSTAMBLE = "no_preamble_postamble"
    JSON_REQUIRED_FIELDS = "json_required_fields"
    FENCED_FINAL_ANSWER = "fenced_final_answer"
    MARKDOWN_PROHIBITED = "markdown_prohibited"
    QUOTE_MAX_LENGTH = "quote_max_length"
    PROHIBITED_CHARACTERS = "prohibited_characters"
    KEYWORD_POSITION = "keyword_position"
    NO_EMOJI = "no_emoji"
    ALLOWED_TAG_VOCABULARY = "allowed_tag_vocabulary"


class AgenticConstraint(BaseModel):
    constraint_type: AgenticConstraintType
    scope: ConstraintScope
    # description is the INJECTABLE instruction (model-facing). Verifier
    # implementation notes belong in infrastructure/verifiers, never here —
    # this text goes into prompts verbatim (after .format(**parameters)).
    description: str
    parameters: dict = {}
    verifier_approach: str = ""
    conflict_with: list[str] = []
    # For BEFORE/AFTER_TOOL_CALL scopes only: True means the constraint
    # obligates model text at EVERY anchor, so a silent tool call is itself a
    # violation (grading emits an uncovered-anchor verdict). False (default)
    # means the constraint's trigger is a subset of anchors — an anchor with
    # no model text is vacuous, never a violation. 2026-08-12 trace-QA audit:
    # applying anchor coverage to conditional constraints mass-false-failed
    # trajectories with silent chained tool calls (~150 false FAILs / 10
    # constraints on the 45-trace smoke batch).
    anchor_text_required: bool = False
    compatible_injection_modes: list[InjectionMode] = Field(
        default_factory=lambda: list(_ALL_INJECTION_MODES)
    )
    format_regimes: list[FormatRegime] = Field(
        default_factory=lambda: [FormatRegime.ANY]
    )


class ConversationalConstraint(BaseModel):
    constraint_type: ConversationalConstraintType
    description: str
    parameters: dict = {}
    verifier_approach: str = ""
    conflict_with: list[str] = []
    # Deliverable finals (a JSON object, a fenced answer) are OWED — a
    # trajectory that never writes a final message violates them; generic
    # formatting rules are conditional on a final existing. Mirrors the
    # agentic anchor_text_required semantics for the synthetic-final check.
    anchor_text_required: bool = False
    compatible_injection_modes: list[InjectionMode] = Field(
        default_factory=lambda: list(_ALL_INJECTION_MODES)
    )
    format_regimes: list[FormatRegime] = Field(
        default_factory=lambda: [FormatRegime.ANY]
    )


# ── Authoritative registry ────────────────────────────────────────────────────
# Each entry is the spec the verifier implements against.
# conflict_with: constraint types that must not appear in the same training example.

AGENTIC_CONSTRAINT_REGISTRY: dict[AgenticConstraintType, AgenticConstraint] = {

    # ── Core agentic ─────────────────────────────────────────────────────────

    AgenticConstraintType.UNIFIED_DIFF: AgenticConstraint(
        constraint_type=AgenticConstraintType.UNIFIED_DIFF,
        scope=ConstraintScope.CODE_STEPS,
        description="Output all file edits as unified diffs (--- a/, +++ b/, @@ headers). Never output full file rewrites.",
        verifier_approach="regex",
        format_regimes=[FormatRegime.TEXT_RESPONSE, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.ACTION_LOG_JSON: AgenticConstraint(
        constraint_type=AgenticConstraintType.ACTION_LOG_JSON,
        scope=ConstraintScope.AFTER_TOOL_CALL,
        description='After each tool call, output a one-line JSON log: {"tool": ..., "input_summary": ..., "result_summary": ...}.',
        verifier_approach="json_parse",
        anchor_text_required=True,
    ),
    AgenticConstraintType.NUMBERED_PLAN: AgenticConstraint(
        constraint_type=AgenticConstraintType.NUMBERED_PLAN,
        scope=ConstraintScope.FIRST_STEP_ONLY,
        description="Begin every multi-step task with a numbered plan (minimum 3 items) before executing any action.",
        verifier_approach="structural",
        parameters={"min_items": 3},
        conflict_with=["sql_explain_before_dml", "anchor_commit_declared_first"],
        compatible_injection_modes=_EARLY_ONLY_MODES,
    ),
    AgenticConstraintType.FILE_PATH_BEFORE_CODE: AgenticConstraint(
        constraint_type=AgenticConstraintType.FILE_PATH_BEFORE_CODE,
        scope=ConstraintScope.CODE_STEPS,
        description="State the full file path before showing any code block.",
        verifier_approach="structural",
    ),
    AgenticConstraintType.STEP_SUMMARY_PREFIX: AgenticConstraint(
        constraint_type=AgenticConstraintType.STEP_SUMMARY_PREFIX,
        scope=ConstraintScope.AFTER_TOOL_CALL,
        description='After each tool observation, write one sentence beginning with "Found:".',
        verifier_approach="string_prefix",
        parameters={"prefix": "Found:"},
        anchor_text_required=True,
    ),
    AgenticConstraintType.SCOPE_CONSTRAINT: AgenticConstraint(
        constraint_type=AgenticConstraintType.SCOPE_CONSTRAINT,
        scope=ConstraintScope.ALL_STEPS,
        description="Only modify files or resources explicitly listed in the user's request.",
        verifier_approach="structural",
    ),
    AgenticConstraintType.JSON_ERROR_REPORTING: AgenticConstraint(
        constraint_type=AgenticConstraintType.JSON_ERROR_REPORTING,
        scope=ConstraintScope.AFTER_TOOL_CALL,
        description='Report all errors as JSON with fields: {"type": ..., "file": ..., "line": ..., "message": ...}.',
        verifier_approach="json_parse",
        parameters={"required_fields": ["type", "file", "line", "message"]},
    ),
    AgenticConstraintType.HANDOFF_SCHEMA: AgenticConstraint(
        constraint_type=AgenticConstraintType.HANDOFF_SCHEMA,
        scope=ConstraintScope.ALL_STEPS,
        description='When delegating to a subagent, use: {"task": ..., "context": ..., "constraints": ...}.',
        verifier_approach="json_parse",
        parameters={"required_fields": ["task", "context", "constraints"]},
    ),
    AgenticConstraintType.OUTPUT_SECTIONS: AgenticConstraint(
        constraint_type=AgenticConstraintType.OUTPUT_SECTIONS,
        scope=ConstraintScope.FINAL_OUTPUT,
        description="Final report must contain sections: Summary, Findings, Recommendations — in that order.",
        verifier_approach="structural",
        parameters={"sections": ["Summary", "Findings", "Recommendations"]},
    ),

    # ── Software engineering ──────────────────────────────────────────────────

    AgenticConstraintType.NO_FORCE_GIT_COMMANDS: AgenticConstraint(
        constraint_type=AgenticConstraintType.NO_FORCE_GIT_COMMANDS,
        scope=ConstraintScope.CODE_STEPS,
        description="Never use destructive git operations: no --force, --hard reset, or branch -D.",
        verifier_approach="regex",
    ),
    AgenticConstraintType.PR_DESCRIPTION_SECTIONS: AgenticConstraint(
        constraint_type=AgenticConstraintType.PR_DESCRIPTION_SECTIONS,
        scope=ConstraintScope.FINAL_OUTPUT,
        description="Pull request descriptions must contain sections: Problem, Solution, Testing — in that order.",
        verifier_approach="structural",
        parameters={"sections": ["Problem", "Solution", "Testing"]},
    ),

    # ── RAG / synthesis ───────────────────────────────────────────────────────

    AgenticConstraintType.CITATION_AFTER_CLAIM: AgenticConstraint(
        constraint_type=AgenticConstraintType.CITATION_AFTER_CLAIM,
        scope=ConstraintScope.ALL_STEPS,
        description="Every factual claim must be followed immediately by a citation tag: <cite>sourceId:chunkIdx</cite>.",
        verifier_approach="regex",
    ),
    AgenticConstraintType.RETRIEVAL_IDS_BEFORE_SYNTHESIS: AgenticConstraint(
        constraint_type=AgenticConstraintType.RETRIEVAL_IDS_BEFORE_SYNTHESIS,
        scope=ConstraintScope.AFTER_TOOL_CALL,
        description="After each retrieval tool call, list the retrieved document IDs before writing any synthesis.",
        verifier_approach="structural",
    ),
    AgenticConstraintType.UNCERTAINTY_FLAG: AgenticConstraint(
        constraint_type=AgenticConstraintType.UNCERTAINTY_FLAG,
        scope=ConstraintScope.ALL_STEPS,
        description="Prefix any claim with low source support with [UNCERTAIN].",
        verifier_approach="string_prefix",
        parameters={"prefix": "[UNCERTAIN]"},
    ),

    # ── Data pipeline ─────────────────────────────────────────────────────────

    AgenticConstraintType.SQL_EXPLAIN_BEFORE_DML: AgenticConstraint(
        constraint_type=AgenticConstraintType.SQL_EXPLAIN_BEFORE_DML,
        scope=ConstraintScope.CODE_STEPS,
        description="Before any INSERT, UPDATE, DELETE, or DROP statement, output a -- comment explaining its purpose and affected rows.",
        verifier_approach="structural",
        conflict_with=["numbered_plan"],
    ),
    AgenticConstraintType.DRY_RUN_BEFORE_EXECUTE: AgenticConstraint(
        constraint_type=AgenticConstraintType.DRY_RUN_BEFORE_EXECUTE,
        scope=ConstraintScope.ALL_STEPS,
        description="Propose every destructive operation as a dry-run first; only execute after confirming the dry-run output.",
        verifier_approach="structural",
    ),

    # ── Security audit ────────────────────────────────────────────────────────

    AgenticConstraintType.SEVERITY_ENUM: AgenticConstraint(
        constraint_type=AgenticConstraintType.SEVERITY_ENUM,
        scope=ConstraintScope.ALL_STEPS,
        description="Every vulnerability finding must be tagged with exactly one severity: [CRITICAL], [HIGH], [MEDIUM], or [LOW].",
        verifier_approach="set_membership",
        parameters={"allowed_values": ["[CRITICAL]", "[HIGH]", "[MEDIUM]", "[LOW]"]},
    ),
    AgenticConstraintType.CVE_FIELDS_REQUIRED: AgenticConstraint(
        constraint_type=AgenticConstraintType.CVE_FIELDS_REQUIRED,
        scope=ConstraintScope.ALL_STEPS,
        description='Report each vulnerability as JSON with fields: {"id": ..., "severity": ..., "file": ..., "line": ..., "description": ..., "remediation": ...}.',
        verifier_approach="json_parse",
        parameters={"required_fields": ["id", "severity", "file", "line", "description", "remediation"]},
    ),

    # ── Multi-agent orchestrator ──────────────────────────────────────────────

    AgenticConstraintType.SUBTASK_ID_ASSIGNED: AgenticConstraint(
        constraint_type=AgenticConstraintType.SUBTASK_ID_ASSIGNED,
        scope=ConstraintScope.ALL_STEPS,
        description="Assign a unique subtask_id (format: ST-NNN) to every delegated subtask before dispatching it.",
        verifier_approach="regex",
        parameters={"pattern": r"ST-\d{3}"},
    ),

    # ── DevOps ────────────────────────────────────────────────────────────────

    AgenticConstraintType.INCIDENT_PRIORITY_TAGGED: AgenticConstraint(
        constraint_type=AgenticConstraintType.INCIDENT_PRIORITY_TAGGED,
        scope=ConstraintScope.ALL_STEPS,
        description="Tag every incident action with its priority: [P0], [P1], [P2], or [P3].",
        verifier_approach="regex",
        parameters={"pattern": r"\[P[0-3]\]"},
    ),
    AgenticConstraintType.IMPACT_BEFORE_REMEDIATION: AgenticConstraint(
        constraint_type=AgenticConstraintType.IMPACT_BEFORE_REMEDIATION,
        scope=ConstraintScope.ALL_STEPS,
        description="State the blast radius / affected systems before proposing any remediation action.",
        verifier_approach="structural",
    ),
    AgenticConstraintType.TIMESTAMP_ISO8601: AgenticConstraint(
        constraint_type=AgenticConstraintType.TIMESTAMP_ISO8601,
        scope=ConstraintScope.ALL_STEPS,
        description="All timestamps in logs and reports must use ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ).",
        verifier_approach="regex",
        parameters={"pattern": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"},
    ),

    # ── Document processing ───────────────────────────────────────────────────

    AgenticConstraintType.PAGE_REF_IN_EXTRACTION: AgenticConstraint(
        constraint_type=AgenticConstraintType.PAGE_REF_IN_EXTRACTION,
        scope=ConstraintScope.ALL_STEPS,
        description='Every extracted field must include a page reference: {"value": ..., "page": N}.',
        verifier_approach="json_parse",
        parameters={"required_fields": ["value", "page"]},
    ),

    # ── Customer support ──────────────────────────────────────────────────────

    AgenticConstraintType.TICKET_ID_IN_ALL_STEPS: AgenticConstraint(
        constraint_type=AgenticConstraintType.TICKET_ID_IN_ALL_STEPS,
        scope=ConstraintScope.ALL_STEPS,
        description="Every action or message must reference the ticket ID (format: TKT-NNNNNN).",
        verifier_approach="regex",
        parameters={"pattern": r"TKT-\d{6}"},
    ),

    # ── Tool use discipline ───────────────────────────────────────────────────

    AgenticConstraintType.TOOL_CALL_INTENT_TAG: AgenticConstraint(
        constraint_type=AgenticConstraintType.TOOL_CALL_INTENT_TAG,
        scope=ConstraintScope.BEFORE_TOOL_CALL,
        description="Every tool invocation must be immediately preceded by [INTENT:<VERB>] <one sentence> where VERB ∈ {READ, SEARCH, EDIT, RUN, GIT, NET}.",
        verifier_approach="regex",
        parameters={"pattern": r"\[INTENT:(READ|SEARCH|EDIT|RUN|GIT|NET)\] .+"},
        anchor_text_required=True,
    ),
    AgenticConstraintType.DIFF_STAT_AFTER_EDIT: AgenticConstraint(
        constraint_type=AgenticConstraintType.DIFF_STAT_AFTER_EDIT,
        scope=ConstraintScope.AFTER_TOOL_CALL,
        description="After each file-edit tool call, emit one line: CHANGED: <path> (+<added>/-<removed>) with counts matching the diff applied.",
        verifier_approach="regex",
        parameters={"pattern": r"CHANGED: .+ \(\+\d+/-\d+\)"},
    ),
    AgenticConstraintType.COMMAND_EXIT_CODE_REPORTED: AgenticConstraint(
        constraint_type=AgenticConstraintType.COMMAND_EXIT_CODE_REPORTED,
        scope=ConstraintScope.AFTER_TOOL_CALL,
        # 2026-08-12 trace-QA repair: the old wording ("the observation must
        # include...") was satisfied by harnesses that append [exit_code=N] to
        # every observation themselves, making compliance undecidable — the
        # judge credited harness output while the verifier graded model text.
        # The obligation is explicitly on the model's own message now.
        description=(
            "After every shell/test/build command, your own next message must restate the "
            "command's result on a line matching exit_code=<int>, followed by FAILED: on "
            "non-zero exit. Exit-code text appearing only inside the tool observation "
            "itself (e.g. a harness-appended [exit_code=N] tail) does not count."
        ),
        verifier_approach="regex",
        parameters={"pattern": r"exit_code=\d+"},
        anchor_text_required=True,
    ),

    # ── Safety and compliance ─────────────────────────────────────────────────

    AgenticConstraintType.ROLLBACK_COMMAND_BEFORE_DEPLOY: AgenticConstraint(
        constraint_type=AgenticConstraintType.ROLLBACK_COMMAND_BEFORE_DEPLOY,
        scope=ConstraintScope.CODE_STEPS,
        description="Before any deploy/apply/upgrade command, emit a line: ROLLBACK: <exact revert command>.",
        verifier_approach="regex",
        parameters={"pattern": r"ROLLBACK: .+"},
    ),
    AgenticConstraintType.NO_SECRET_LITERALS_IN_CODE: AgenticConstraint(
        constraint_type=AgenticConstraintType.NO_SECRET_LITERALS_IN_CODE,
        scope=ConstraintScope.CODE_STEPS,
        description="No code block may contain hardcoded credentials. AWS keys, bearer/JWT tokens, password= or api_key= literals, PEM headers are all violations.",
        verifier_approach="regex",
        parameters={"forbidden_patterns": ["AKIA[0-9A-Z]{16}", r"password\s*=\s*['\"][^${\(]", "-----BEGIN.*PRIVATE KEY-----"]},
    ),
    AgenticConstraintType.PII_MASKED_IN_TRANSCRIPT: AgenticConstraint(
        constraint_type=AgenticConstraintType.PII_MASKED_IN_TRANSCRIPT,
        scope=ConstraintScope.ALL_STEPS,
        description="Customer PII in agent output must be masked: emails → first-char***@domain, card numbers → ****last4, phones → ***-***-NNNN.",
        verifier_approach="regex",
    ),
    AgenticConstraintType.ENV_TAG_ON_EVERY_COMMAND: AgenticConstraint(
        constraint_type=AgenticConstraintType.ENV_TAG_ON_EVERY_COMMAND,
        scope=ConstraintScope.ALL_STEPS,
        description="Every tool call and command block must begin with exactly one environment tag: [ENV:prod], [ENV:staging], or [ENV:dev].",
        verifier_approach="set_membership",
        parameters={"allowed_values": ["[ENV:prod]", "[ENV:staging]", "[ENV:dev]"]},
    ),
    AgenticConstraintType.KUBECTL_NAMESPACE_EXPLICIT: AgenticConstraint(
        constraint_type=AgenticConstraintType.KUBECTL_NAMESPACE_EXPLICIT,
        scope=ConstraintScope.CODE_STEPS,
        description="Every kubectl/helm invocation must include -n <ns> or --namespace=<ns> (or --all-namespaces for read-only listing). Implicit namespace is a violation.",
        verifier_approach="regex",
        parameters={"pattern": r"kubectl .+(-n |--namespace=|--all-namespaces)"},
    ),

    # ── Software engineering hygiene ──────────────────────────────────────────

    AgenticConstraintType.TEST_COMMAND_BEFORE_PATCH: AgenticConstraint(
        constraint_type=AgenticConstraintType.TEST_COMMAND_BEFORE_PATCH,
        scope=ConstraintScope.CODE_STEPS,
        description="Before any source-file edit, show the output of at least one test/reproduction command, and precede the edit block with Repro: <command>.",
        verifier_approach="regex",
        parameters={"pattern": r"Repro: .+"},
    ),
    AgenticConstraintType.BRANCH_NAME_CONVENTION: AgenticConstraint(
        constraint_type=AgenticConstraintType.BRANCH_NAME_CONVENTION,
        scope=ConstraintScope.ALL_STEPS,
        description="Any git branch the agent creates must match ^(fix|feat|chore|refactor)/[a-z0-9._-]+-[0-9]+$ (type-prefix/slug-issuenum).",
        verifier_approach="regex",
        parameters={"pattern": r"^(fix|feat|chore|refactor)/[a-z0-9._-]+-\d+$"},
    ),

    # ── SWE-bench multi-turn ──────────────────────────────────────────────────

    AgenticConstraintType.EXPECTED_ACTUAL_ERROR_BLOCK: AgenticConstraint(
        constraint_type=AgenticConstraintType.EXPECTED_ACTUAL_ERROR_BLOCK,
        scope=ConstraintScope.AFTER_TOOL_CALL,
        description=(
            "The first time a reproduction command is observed to fail, emit a three-line block, "
            "each on its own line and in this order: 'EXPECTED: <one line>', 'ACTUAL: <one line>', "
            "'ERROR_TYPE: <CamelCase exception or symbol name>'."
        ),
        verifier_approach="structural",
        # 2026-08-12 coherence audit: conflict restored from constraint_metadata.json,
        # which retained it after it was dropped from the registry.
        conflict_with=["json_error_reporting"],
    ),
    AgenticConstraintType.EXPLICIT_TEST_TARGET_REQUIRED: AgenticConstraint(
        constraint_type=AgenticConstraintType.EXPLICIT_TEST_TARGET_REQUIRED,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "Every test-runner invocation (pytest/unittest/tox/nose) must name an explicit target — "
            "a file path, a directory, or a node id containing '::' — or use an explicit selector "
            "flag (-k or -m). A bare 'pytest' that runs the whole suite is a violation."
        ),
        verifier_approach="regex",
    ),
    AgenticConstraintType.EXPLICIT_TEST_SELECTION_ARGS: AgenticConstraint(
        constraint_type=AgenticConstraintType.EXPLICIT_TEST_SELECTION_ARGS,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "Every test-execution command must state its selection explicitly: the command line must "
            "contain a path or node id (e.g. tests/test_x.py, tests/test_x.py::TestC::test_m) or a "
            "selector flag (-k <expr>, -m <marker>). Unscoped full-suite runs are violations."
        ),
        verifier_approach="regex",
    ),
    AgenticConstraintType.PYTEST_TARGET_SCOPED: AgenticConstraint(
        constraint_type=AgenticConstraintType.PYTEST_TARGET_SCOPED,
        # 2026-08-12 trace-QA repair: BEFORE_TOOL_CALL graded the narration
        # preceding a call — which never contains the command — so bare
        # `python -m pytest` invocations were structurally invisible. The
        # constraint governs the command itself: CODE_STEPS (includes tool
        # calls since the same audit).
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "Every test-runner invocation must name at least one explicit target: a file path, a "
            "directory, or a node id of the form path::Name (optionally path::Class::test_name). "
            "A bare 'pytest', 'python -m pytest', 'tox', or 'make test' is a violation."
        ),
        verifier_approach="regex",
    ),
    AgenticConstraintType.EDITS_VIA_EDIT_TOOL_ONLY: AgenticConstraint(
        constraint_type=AgenticConstraintType.EDITS_VIA_EDIT_TOOL_ONLY,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "All source-file mutations must go through the structured file edit/write tool. Mutating a "
            "repository file from the shell is a violation: redirects ('>', '>>', 'tee') into a repo "
            "path, heredocs writing to a path, 'sed -i', 'perl -i', 'truncate', or 'mv'/'cp' over a "
            "tracked file."
        ),
        verifier_approach="structural",
        # 2026-08-12 coherence audit: conflict restored from constraint_metadata.json,
        # which retained it after it was dropped from the registry.
        conflict_with=["unified_diff"],
    ),
    AgenticConstraintType.EXPECTATION_BEFORE_RUN_CHECK_AFTER: AgenticConstraint(
        constraint_type=AgenticConstraintType.EXPECTATION_BEFORE_RUN_CHECK_AFTER,
        scope=ConstraintScope.AFTER_TOOL_CALL,
        description=(
            "Every command-execution tool call must be preceded by 'EXPECT: <one sentence prediction>' "
            "and, after the observation, followed by 'ACTUAL: <one sentence>' and a line reading "
            "'MATCH: yes' or 'MATCH: no'."
        ),
        verifier_approach="structural",
        # 2026-08-12 coherence audit: conflict restored from constraint_metadata.json,
        # which retained it after it was dropped from the registry.
        conflict_with=["step_summary_prefix"],
        anchor_text_required=True,
    ),
    AgenticConstraintType.FAILING_TEST_ID_ENUMERATION: AgenticConstraint(
        constraint_type=AgenticConstraintType.FAILING_TEST_ID_ENUMERATION,
        scope=ConstraintScope.AFTER_TOOL_CALL,
        description=(
            "After any test command that reports failures, emit a line 'FAILING: <id>[, <id>...]' "
            "listing the failing test node ids exactly as they appear in the tool output. The number "
            "of ids listed must equal the number of failures reported."
        ),
        verifier_approach="structural",
    ),
    AgenticConstraintType.REREAD_BEFORE_EDIT_RETRY: AgenticConstraint(
        constraint_type=AgenticConstraintType.REREAD_BEFORE_EDIT_RETRY,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "When an edit tool call fails (no match for the search string, patch rejected, file not "
            "found), the next tool call must read or grep the same path. Only after that observation "
            "may an edit on that path be re-attempted."
        ),
        verifier_approach="structural",
        # 2026-08-12 coherence audit: conflict restored from constraint_metadata.json,
        # which retained it after it was dropped from the registry.
        conflict_with=["read_before_edit_citation"],
    ),
    AgenticConstraintType.STRAY_FILE_AUDIT_LINE: AgenticConstraint(
        constraint_type=AgenticConstraintType.STRAY_FILE_AUDIT_LINE,
        scope=ConstraintScope.AFTER_TOOL_CALL,
        description=(
            "After a 'git status --porcelain' tool call, emit 'STRAY_UNTRACKED: none' or "
            "'STRAY_UNTRACKED: <path>[, <path>...]', and the listed paths must exactly match the '??' "
            "entries in that git status output."
        ),
        verifier_approach="structural",
    ),
    AgenticConstraintType.SCRATCH_FILE_LEDGER: AgenticConstraint(
        constraint_type=AgenticConstraintType.SCRATCH_FILE_LEDGER,
        scope=ConstraintScope.FINAL_OUTPUT,
        description=(
            "Every file created that is not part of the fix (repro scripts, logs, scratch output) must "
            "be announced at creation with 'SCRATCH: <path>', and before the final message each "
            "announced path must appear in a removal line 'REMOVED: <path>'."
        ),
        verifier_approach="structural",
        # 2026-08-12: final-message deliverable owed even when the trajectory
        # ends silently (see FINAL_OUTPUT handling in verifiers/trajectory.py).
        anchor_text_required=True,
    ),

    # ── Multi-agent orchestrator extended ─────────────────────────────────────

    AgenticConstraintType.DELEGATION_BUDGET_FIELD: AgenticConstraint(
        constraint_type=AgenticConstraintType.DELEGATION_BUDGET_FIELD,
        scope=ConstraintScope.ALL_STEPS,
        description='Every delegation payload must include a "budget" object with integer "max_tool_calls" and "max_tokens" keys.',
        verifier_approach="json_parse",
        parameters={"required_fields": ["budget"], "budget_fields": ["max_tool_calls", "max_tokens"]},
        conflict_with=["handoff_schema"],
    ),
    AgenticConstraintType.RETRY_ATTEMPT_COUNTER: AgenticConstraint(
        constraint_type=AgenticConstraintType.RETRY_ATTEMPT_COUNTER,
        scope=ConstraintScope.ALL_STEPS,
        description='Any retry of a previously dispatched subtask must carry "attempt <k>/<max>" where k increments per retry and never exceeds max.',
        verifier_approach="regex",
        parameters={"pattern": r"attempt \d+/\d+"},
    ),

    # ── ReAct / API orchestration format ──────────────────────────────────────

    AgenticConstraintType.REACT_STEP_INDEX_MONOTONIC: AgenticConstraint(
        constraint_type=AgenticConstraintType.REACT_STEP_INDEX_MONOTONIC,
        scope=ConstraintScope.ALL_STEPS,
        description="Every ReAct cycle must be numbered with a step header 'Step <N>:' immediately before the Thought line, starting at 1 and incrementing by exactly 1 with no gaps, repeats, or resets — including across backtracked branches.",
        verifier_approach="structural",
        parameters={"pattern": r"^Step \d+:"},
        format_regimes=[FormatRegime.TEXT_REACT],
    ),
    AgenticConstraintType.ACTION_INPUT_STRICT_JSON: AgenticConstraint(
        constraint_type=AgenticConstraintType.ACTION_INPUT_STRICT_JSON,
        scope=ConstraintScope.ALL_STEPS,
        description="Every 'Action Input:' payload must be a single-line parseable JSON object with double-quoted keys, no trailing commas, no Python literals (None/True/False), and no markdown fences. Empty calls must be '{}'.",
        verifier_approach="json_parse",
        format_regimes=[FormatRegime.TEXT_REACT],
    ),
    AgenticConstraintType.API_CATEGORY_TAG_PER_ACTION: AgenticConstraint(
        constraint_type=AgenticConstraintType.API_CATEGORY_TAG_PER_ACTION,
        scope=ConstraintScope.ALL_STEPS,
        description="Every non-Finish Action line must be immediately preceded by a category tag of the form '[CATEGORY:<Name>]' naming the API category (e.g. Location, Weather, Finance, Sports).",
        verifier_approach="regex",
        parameters={"pattern": r"\[CATEGORY:[A-Za-z][A-Za-z0-9 _-]*\]"},
        format_regimes=[FormatRegime.TEXT_REACT],
    ),

    # ── Function calling protocol ──────────────────────────────────────────────

    AgenticConstraintType.IRRELEVANCE_SENTINEL_LINE: AgenticConstraint(
        constraint_type=AgenticConstraintType.IRRELEVANCE_SENTINEL_LINE,
        scope=ConstraintScope.ALL_STEPS,
        description="When no function in the provided schema can satisfy the request, the agent must output 'NO_FUNCTION_APPLICABLE: <reason>' (single line, reason non-empty) and make zero tool calls in that turn.",
        verifier_approach="regex",
        parameters={"pattern": r"^NO_FUNCTION_APPLICABLE: .+"},
    ),
    AgenticConstraintType.UNAVAILABLE_TOOL_DECLARATION: AgenticConstraint(
        constraint_type=AgenticConstraintType.UNAVAILABLE_TOOL_DECLARATION,
        scope=ConstraintScope.ALL_STEPS,
        description="When a required tool is absent from the tool list, the agent must emit 'UNAVAILABLE: <capability>' followed by 'NO_CALL_MADE' on the next line, and make no tool call in that turn.",
        verifier_approach="regex",
        parameters={"pattern": r"^UNAVAILABLE: .+\nNO_CALL_MADE"},
    ),
    AgenticConstraintType.EXTRA_PARAM_REJECTION_LINE: AgenticConstraint(
        constraint_type=AgenticConstraintType.EXTRA_PARAM_REJECTION_LINE,
        scope=ConstraintScope.ALL_STEPS,
        description="When the user supplies arguments not in the function schema, the agent must emit 'IGNORED_PARAMS: [<name>, ...]' before the tool call, and the subsequent call must contain none of those keys.",
        verifier_approach="regex",
        parameters={"pattern": r"^IGNORED_PARAMS: \[.+\]"},
    ),
    AgenticConstraintType.MISSING_PARAM_QUESTION_BLOCK: AgenticConstraint(
        constraint_type=AgenticConstraintType.MISSING_PARAM_QUESTION_BLOCK,
        scope=ConstraintScope.ALL_STEPS,
        description="When required parameters are missing, the agent must emit a MISSING_PARAMS block where each line matches '- <param_name> (<type>): <question ending in ?>'. Parameter names must be the exact schema identifiers.",
        verifier_approach="regex",
        parameters={"pattern": r"^MISSING_PARAMS\n(- \w+ \([^)]+\): .+\?)+"},
        format_regimes=[FormatRegime.USER_INTERACTIVE],
    ),
    AgenticConstraintType.NESTED_CALL_INLINE_SYNTAX: AgenticConstraint(
        constraint_type=AgenticConstraintType.NESTED_CALL_INLINE_SYNTAX,
        scope=ConstraintScope.ALL_STEPS,
        description="Nested function composition must be expressed as a single inline expression 'outer(param=inner(param=value))'. Placeholder-based indirection ($result, {{step1}}) or split-message composition is a violation.",
        verifier_approach="regex",
        parameters={"pattern": r"\w+\([^)]*\w+\([^)]+\)[^)]*\)"},
    ),
    AgenticConstraintType.PARALLEL_GROUP_FANOUT_DECLARATION: AgenticConstraint(
        constraint_type=AgenticConstraintType.PARALLEL_GROUP_FANOUT_DECLARATION,
        scope=ConstraintScope.ALL_STEPS,
        description="Any batch of concurrently dispatched subtasks must be preceded by 'FANOUT <PG-N> count=<k> max=<m>' (k ≤ m), and every delegation payload in that batch must carry \"parallel_group\": \"PG-N\" matching the declared group.",
        verifier_approach="structural",
        parameters={"pattern": r"^FANOUT PG-\d+ count=\d+ max=\d+"},
        conflict_with=["delegation_budget_field"],
    ),
    AgenticConstraintType.ARG_PROVENANCE_MAP: AgenticConstraint(
        constraint_type=AgenticConstraintType.ARG_PROVENANCE_MAP,
        scope=ConstraintScope.CODE_STEPS,
        description='Before every tool call, emit one line: ARG_SOURCES: {"<arg>": "user"|"tool_result"|"schema_default", ...} with keys exactly matching the call\'s argument object.',
        verifier_approach="json_parse",
        parameters={"prefix": "ARG_SOURCES: ", "value_enum": ["user", "tool_result", "schema_default"]},
    ),
    AgenticConstraintType.FORBIDDEN_TOOL_ABSTENTION: AgenticConstraint(
        constraint_type=AgenticConstraintType.FORBIDDEN_TOOL_ABSTENTION,
        scope=ConstraintScope.ALL_STEPS,
        # Model-facing template; the pair generator fills {forbidden_tool}.
        # Verification: the named tool must not appear in any function call
        # (see infrastructure/verifiers — _ForbiddenToolAbstentionVerifier).
        description=(
            "The tool `{forbidden_tool}` in your tool list is strictly forbidden: "
            "never invoke it under any circumstances, even if it appears useful. "
            "Complete the task without it."
        ),
        parameters={"forbidden_tool": "restricted_tool"},
        verifier_approach="regex",
        compatible_injection_modes=_EARLY_ONLY_MODES,
    ),

    # ── Repository repair discipline ───────────────────────────────────────────

    AgenticConstraintType.ANCHOR_COMMIT_DECLARED_FIRST: AgenticConstraint(
        constraint_type=AgenticConstraintType.ANCHOR_COMMIT_DECLARED_FIRST,
        scope=ConstraintScope.FIRST_STEP_ONLY,
        description="The very first output must declare the base commit: 'BASE_COMMIT: <40-hex-sha>'. Every subsequent diff block must be prefixed with 'APPLIES_TO: <same-sha>'.",
        verifier_approach="regex",
        parameters={"pattern": r"^BASE_COMMIT: [0-9a-f]{40}"},
        conflict_with=["numbered_plan"],
        compatible_injection_modes=_EARLY_ONLY_MODES,
    ),
    AgenticConstraintType.READ_BEFORE_EDIT_CITATION: AgenticConstraint(
        constraint_type=AgenticConstraintType.READ_BEFORE_EDIT_CITATION,
        scope=ConstraintScope.CODE_STEPS,
        description="Every edit tool call (str_replace/sed/patch) must be immediately preceded by 'READ: <path>:<start>-<end>' naming the file and line range already observed in a prior tool output. Editing without a prior read is a violation.",
        verifier_approach="regex",
        parameters={"pattern": r"^READ: .+:\d+-\d+"},
    ),
    AgenticConstraintType.HYPOTHESIS_BEFORE_EDIT_TAG: AgenticConstraint(
        constraint_type=AgenticConstraintType.HYPOTHESIS_BEFORE_EDIT_TAG,
        scope=ConstraintScope.CODE_STEPS,
        description="Before the first edit to any file, the agent must output 'HYPOTHESIS[<path>]: <cause> -> <expected test effect>' with an arrow '->' and reference to at least one test name mentioned earlier in the trajectory.",
        verifier_approach="regex",
        parameters={"pattern": r"^HYPOTHESIS\[.+\]: .+ -> .+"},
    ),

    # ── Customer service compliance ────────────────────────────────────────────

    AgenticConstraintType.CONFIRMATION_GATE_TOKEN: AgenticConstraint(
        constraint_type=AgenticConstraintType.CONFIRMATION_GATE_TOKEN,
        scope=ConstraintScope.ALL_STEPS,
        description="Every state-modifying tool call must be preceded by 'CONFIRM: <tool_name> | <key=value>[; ...] | reply YES to proceed', then by a user message containing an affirmative token (yes/y/confirm/confirmed/go ahead). Executing before the CONFIRM line or without user affirmative is a violation.",
        verifier_approach="structural",
        parameters={"pattern": r"^CONFIRM: \w+ \|"},
        conflict_with=["dry_run_before_execute"],
        format_regimes=[FormatRegime.USER_INTERACTIVE],
    ),
    AgenticConstraintType.POLICY_CLAUSE_CITE_ON_REFUSAL: AgenticConstraint(
        constraint_type=AgenticConstraintType.POLICY_CLAUSE_CITE_ON_REFUSAL,
        scope=ConstraintScope.ALL_STEPS,
        description="Any refusal or decline must contain '[POLICY:<SECTION_ID>] \"<quoted clause>\"' where SECTION_ID matches ^[A-Z]{2,10}-[0-9]{1,3}(\\.[0-9]{1,2})?$ and the clause text is in double quotes on the same line.",
        verifier_approach="regex",
        parameters={"pattern": r"\[POLICY:[A-Z]{2,10}-\d{1,3}(\.\d{1,2})?\] \".+\""},
        conflict_with=["citation_after_claim"],
        format_regimes=[FormatRegime.USER_INTERACTIVE],
    ),

    # ── Code documentation format ──────────────────────────────────────────────

    AgenticConstraintType.DOCSTRING_SECTION_ORDER_FIXED: AgenticConstraint(
        constraint_type=AgenticConstraintType.DOCSTRING_SECTION_ORDER_FIXED,
        scope=ConstraintScope.CODE_STEPS,
        description="Every docstring must contain Google-style section headers 'Args:', 'Returns:', 'Raises:', 'Example:' — all four present, in that exact order, no other section names allowed (not Parameters:, Output:, Throws:, Usage:).",
        verifier_approach="structural",
        parameters={"required_sections_in_order": ["Args:", "Returns:", "Raises:", "Example:"]},
        conflict_with=["section_headers"],
    ),

    # ── Progress reporting format ──────────────────────────────────────────────

    AgenticConstraintType.MONOTONIC_STEP_INDEX_HEADER: AgenticConstraint(
        constraint_type=AgenticConstraintType.MONOTONIC_STEP_INDEX_HEADER,
        scope=ConstraintScope.ALL_STEPS,
        # 2026-08-11 repair: the original 'STEP <nnn>/<NNN>' form demanded a
        # constant denominator predicted upfront, which turns into nonsense
        # (STEP 013/010) whenever the trajectory outruns the guess.
        description="Every status update must open with 'STEP <nnn>' (zero-padded 3 digits, starting at 001 and increasing by exactly 1 per update). The header must be the first line with no leading whitespace or markdown.",
        verifier_approach="structural",
        parameters={"pattern": r"^STEP \d{3}\b"},
        conflict_with=["response_prefix", "tldr_prefix", "react_step_index_monotonic"],
        format_regimes=[FormatRegime.TEXT_RESPONSE, FormatRegime.TEXT_REACT],
    ),

    # ── RL environment interaction ─────────────────────────────────────────────

    AgenticConstraintType.RL_REWARD_REPORTED: AgenticConstraint(
        constraint_type=AgenticConstraintType.RL_REWARD_REPORTED,
        scope=ConstraintScope.AFTER_TOOL_CALL,
        description="After every environment step (action/tool call), emit a line 'REWARD: <float>' reporting the scalar reward received. Missing or non-numeric reward lines are violations.",
        verifier_approach="regex",
        parameters={"pattern": r"^REWARD: -?\d+(?:\.\d+)?$"},
        format_regimes=[FormatRegime.ENV_REWARD_VISIBLE],
        anchor_text_required=True,
    ),

    # ── Agentic coding discipline ─────────────────────────────────────────────
    # Grounded in customer traces; see feedbacks/customer_cursor/mapping.md
    # (code_cite_line_range_format, no_trailing_colon_before_tool) and
    # feedbacks/customer_coderabbit/mapping.md (approval_body_exact_literal,
    # review_file_wrap_markers).

    AgenticConstraintType.CODE_CITE_LINE_RANGE_FORMAT: AgenticConstraint(
        # customer_cursor → code-reference-format-never-used: 22/23 code fences
        # in the lombok trace used language tags instead of the mandated
        # startLine:endLine:filepath info-string.  Violation was total
        # non-adoption from turn 4 onward, not late-run decay.
        constraint_type=AgenticConstraintType.CODE_CITE_LINE_RANGE_FORMAT,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "When citing existing code from the codebase, the code fence info-string must be "
            "`startLine:endLine:filepath` (e.g., `42:58:src/utils/auth.ts`). "
            "Language-tagged fences (```python, ```typescript, etc.) are only permitted for "
            "new or proposed code that does not already exist in the repository."
        ),
        verifier_approach="regex",
        parameters={"pattern": r"^```\d+:\d+:.+"},
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
        compatible_injection_modes=_EARLY_ONLY_MODES,
    ),

    AgenticConstraintType.NO_TRAILING_COLON_BEFORE_TOOL: AgenticConstraint(
        # customer_cursor → colon-before-tool-calls: 82/89 text-bearing
        # tool-call turns violated the prohibition despite it sitting in the
        # system prompt.  Pattern is a first-turn cliff: complies turn 1, then
        # essentially never applies the rule again.
        constraint_type=AgenticConstraintType.NO_TRAILING_COLON_BEFORE_TOOL,
        scope=ConstraintScope.BEFORE_TOOL_CALL,
        description=(
            "Do not end the sentence immediately before a tool call with a colon. "
            "Use a period: write 'Let me read the file.' not 'Let me read the file:' "
            "Your tool calls may not be shown directly in the output, so trailing colons "
            "leave the narration incomplete."
        ),
        verifier_approach="regex",
        # Matches a line ending in ':' immediately before a tool call marker.
        parameters={"forbidden_pattern": r":\s*$"},
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL],
    ),

    AgenticConstraintType.APPROVAL_BODY_EXACT_LITERAL: AgenticConstraint(
        # customer_coderabbit → lgtm-exact-string: 134 strict violations over
        # 30 calls; model adds prose ("LGTM! Good refactor…") despite the prompt
        # stating the body must be exactly `LGTM!` with nothing before or after.
        constraint_type=AgenticConstraintType.APPROVAL_BODY_EXACT_LITERAL,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "When posting an approval comment, the comment body must be exactly "
            "`{approval_literal}` — no additional text, explanation, or punctuation "
            "before or after it."
        ),
        verifier_approach="string_match",
        parameters={"approval_literal": "LGTM!"},
    ),

    AgenticConstraintType.REVIEW_FILE_WRAP_MARKERS: AgenticConstraint(
        # customer_coderabbit → missing-file-markers: 9/30 calls omitted
        # file_start markers for at least one provided file; worst case dropped
        # 4 of 5 files while still emitting review comments.
        constraint_type=AgenticConstraintType.REVIEW_FILE_WRAP_MARKERS,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "For every file provided in a review request, emit `{start_marker} <path>` "
            "before its review section and `{end_marker} <path>` after it, even if the "
            "file has no review comments. Files without wrap markers are violations."
        ),
        verifier_approach="structural",
        parameters={"start_marker": "file_start", "end_marker": "file_end"},
    ),

    # ── SWE-bench batch 2 (curated 2026-08-11) ────────────────────────────────

    AgenticConstraintType.OPENING_TRIAGE_ENUM_LINE: AgenticConstraint(
        constraint_type=AgenticConstraintType.OPENING_TRIAGE_ENUM_LINE,
        scope=ConstraintScope.FIRST_STEP_ONLY,
        description=(
            "Your first message must contain exactly one triage line at column 0 of the exact form "
            "'TRIAGE: type=<BUG|REGRESSION|FEATURE|PERF|DOC|TEST> | surface=<python|js|ts|go|rust|c|cpp|other> "
            "| entry=<traceback|failing-test|described-behavior>', with values drawn exactly from those sets. "
            "Do not emit a TRIAGE line in any later message."
        ),
        verifier_approach="regex",
        compatible_injection_modes=_EARLY_ONLY_MODES,
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.NO_SIMULATED_TOOL_OUTPUT_IN_OPENING: AgenticConstraint(
        constraint_type=AgenticConstraintType.NO_SIMULATED_TOOL_OUTPUT_IN_OPENING,
        scope=ConstraintScope.FIRST_STEP_ONLY,
        description=(
            "Your first message, written before any command has run, must not contain text that looks like "
            "command, test, or git output: no 'Traceback (most recent call last)', no 'File \"...\", line N', "
            "no 'test session starts', no 'N passed/failed/skipped' counts, no 'exit_code=', no '$ cmd' lines, "
            "and no git status phrases. Quoting the issue report is allowed only in '>' blockquote lines."
        ),
        verifier_approach="regex",
        compatible_injection_modes=_EARLY_ONLY_MODES,
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.NO_OUTCOME_CLAIMS_IN_OPENING: AgenticConstraint(
        constraint_type=AgenticConstraintType.NO_OUTCOME_CLAIMS_IN_OPENING,
        scope=ConstraintScope.FIRST_STEP_ONLY,
        description=(
            "Your first message must not claim any work is already done or verified: no 'I have "
            "fixed/patched/resolved/verified', no 'all tests pass', no 'the bug is fixed', no 'the fix works', "
            "and never the word 'successfully'. Future-tense planning language ('I will fix', 'the goal is "
            "to make the failing test pass') is unrestricted."
        ),
        verifier_approach="regex",
        compatible_injection_modes=_EARLY_ONLY_MODES,
    ),
    AgenticConstraintType.ORIENTATION_MESSAGE_PRECEDES_FIRST_TOOL_CALL: AgenticConstraint(
        constraint_type=AgenticConstraintType.ORIENTATION_MESSAGE_PRECEDES_FIRST_TOOL_CALL,
        scope=ConstraintScope.FIRST_STEP_ONLY,
        description=(
            "Your first message must issue zero tool calls — it is a pure orientation message. Its last "
            "non-empty line must be exactly one line of the form 'NEXT: <SEARCH|READ|RUN> | <what and why>', "
            "with nothing after it. Issue your first tool call only in a later message."
        ),
        verifier_approach="structural",
        compatible_injection_modes=_EARLY_ONLY_MODES,
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.SEARCH_BEFORE_FIRST_READ: AgenticConstraint(
        constraint_type=AgenticConstraintType.SEARCH_BEFORE_FIRST_READ,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "Your first tool call of the trajectory must be a search or listing operation "
            "(grep/rg/find/ls/glob), not a file read, a file edit, or a test run. Locate candidate "
            "files before opening anything."
        ),
        verifier_approach="structural",
        compatible_injection_modes=_EARLY_ONLY_MODES,
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.TIMEOUT_WRAPPED_EXECUTION: AgenticConstraint(
        constraint_type=AgenticConstraintType.TIMEOUT_WRAPPED_EXECUTION,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "Prefix every shell command that executes code (python, pytest, tox, nose, make, node, npm, "
            "bash/sh on a script, or ./script) with 'timeout <positive integer>[s|m]', optionally preceded "
            "only by VAR=value environment assignments. Pure inspection commands (ls, cat, grep, find, git, "
            "sed without -i, head, wc) are exempt."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.GREP_SCOPED_AND_NUMBERED: AgenticConstraint(
        constraint_type=AgenticConstraintType.GREP_SCOPED_AND_NUMBERED,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "Every grep/rg search must include a line-number flag (-n or --line-number) AND an explicit "
            "path or glob argument scoping it to a directory inside the repository. Bare recursive searches "
            "with no path, or searches rooted at '/', '~', or '/usr', are violations."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.NONINTERACTIVE_COMMAND_DISCIPLINE: AgenticConstraint(
        constraint_type=AgenticConstraintType.NONINTERACTIVE_COMMAND_DISCIPLINE,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "Every bash command must be non-interactive and non-paging: use 'git --no-pager' (or pipe into "
            "cat/head) for git log/show/diff/blame; never invoke less, more, man, vim, vi, nano, emacs, top, "
            "or htop; never start a bare python/ipython/node REPL without -c, -m, or a script argument."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.REPRO_SCRIPT_SANDBOX_PATH: AgenticConstraint(
        constraint_type=AgenticConstraintType.REPRO_SCRIPT_SANDBOX_PATH,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "Write every reproduction or debug artifact you create to an absolute path under /tmp/ with a "
            "basename matching repro_*, debug_*, or scratch_* (e.g. /tmp/repro_issue.py). Never create repro "
            "scripts inside the repository working tree."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.REMOVAL_INTENT_TAG: AgenticConstraint(
        constraint_type=AgenticConstraintType.REMOVAL_INTENT_TAG,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "Immediately before any deletion or truncation command (rm, unlink, find -delete, truncate), "
            "emit one line per targeted path of the form 'REMOVE: <path> | reason=<repro|debug-output|"
            "build-artifact|editor-backup>'. The tagged path must appear in the command that follows."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.GIT_SUBCOMMAND_MODE_DECLARATION: AgenticConstraint(
        constraint_type=AgenticConstraintType.GIT_SUBCOMMAND_MODE_DECLARATION,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "Immediately before every shell command that invokes git, emit one line 'GIT_OP: <subcommand> | "
            "mode=<read|write>' where <subcommand> is the first non-flag token after 'git'. Declare mode=read "
            "only for status, diff, log, show, ls-files, rev-parse, blame, cat-file, or describe; declare "
            "every other subcommand mode=write. The tag never forbids any command."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.CONFIG_FILE_EDIT_DECLARATION_TAG: AgenticConstraint(
        constraint_type=AgenticConstraintType.CONFIG_FILE_EDIT_DECLARATION_TAG,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "Immediately before editing any packaging/build/CI/lint configuration file (setup.py, setup.cfg, "
            "pyproject.toml, MANIFEST.in, tox.ini, pytest.ini, Dockerfile, Makefile, requirements*.txt, or "
            "anything under .github/workflows/), emit exactly one line 'CONFIG_EDIT: <path> | "
            "class=<packaging|build|ci|lint|runtime-config> | reason=<non-empty>'. Ordinary source or test "
            "edits must not carry this line. Editing config files is allowed; it only has to be announced."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.OUT_OF_REPO_PATH_ACCESS_TAG: AgenticConstraint(
        constraint_type=AgenticConstraintType.OUT_OF_REPO_PATH_ACCESS_TAG,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "Immediately before any tool call whose arguments reference a path starting with /usr/, /opt/, "
            "or /etc/, or containing site-packages/ or dist-packages/, emit exactly one line "
            "'OUT_OF_REPO: <path> | graded=no | purpose=<inspect|run|env-check>' naming that path verbatim. "
            "Calls touching only repo or /tmp/ paths must not carry this line. Inspecting installed packages "
            "is allowed; it only has to be announced."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.TEST_TALLY_LINE_AFTER_RUN: AgenticConstraint(
        constraint_type=AgenticConstraintType.TEST_TALLY_LINE_AFTER_RUN,
        scope=ConstraintScope.AFTER_TOOL_CALL,
        description=(
            "Immediately after every test-execution command (pytest/unittest/tox/nose/make test), and before "
            "any prose analysis of that run, emit a single line of the exact form "
            "'TESTS: passed=<int> failed=<int> errors=<int> skipped=<int>' transcribing the run output's counts."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.FAILURE_CLASS_ENUM_TAG: AgenticConstraint(
        constraint_type=AgenticConstraintType.FAILURE_CLASS_ENUM_TAG,
        scope=ConstraintScope.AFTER_TOOL_CALL,
        description=(
            "After every observation reporting a failure (non-zero exit code, failed test run, or a failed "
            "edit), emit a single line 'FAILURE_CLASS: <TOKEN>' where TOKEN is exactly one of ENV, DEPENDENCY, "
            "SYNTAX, IMPORT, ASSERTION, TIMEOUT, PATCH_NOMATCH, PERMISSION, UNKNOWN — one token, no free text."
        ),
        verifier_approach="set_membership",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.LARGE_OBSERVATION_FOCUS_LINE: AgenticConstraint(
        constraint_type=AgenticConstraintType.LARGE_OBSERVATION_FOCUS_LINE,
        scope=ConstraintScope.AFTER_TOOL_CALL,
        description=(
            "When a tool observation exceeds 200 lines, your next message must contain exactly one line "
            "'OBS_LARGE: <source> | focus=\"<8-200 char excerpt>\"' where <source> is the path read or the "
            "first token of the command that produced it, and the excerpt occurs verbatim in that observation. "
            "Never emit an OBS_LARGE line after an observation of 50 lines or fewer."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.FINAL_TEST_LEDGER_JSON_BLOCK: AgenticConstraint(
        constraint_type=AgenticConstraintType.FINAL_TEST_LEDGER_JSON_BLOCK,
        scope=ConstraintScope.FINAL_OUTPUT,
        description=(
            "End your final message with a single fenced ```json block parsing to an object with exactly the "
            "keys 'newly_passing', 'still_failing_preexisting', 'newly_failing' (each an array of test node "
            "ids like path/to/test.py::test_name) plus integer key 'commands_run'. List only node ids that "
            "appeared in earlier tool output."
        ),
        verifier_approach="json_parse",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
        # 2026-08-12: final-message deliverable owed even when the trajectory
        # ends silently (see FINAL_OUTPUT handling in verifiers/trajectory.py).
        anchor_text_required=True,
    ),
    AgenticConstraintType.CHANGED_FILES_MANIFEST_FINAL: AgenticConstraint(
        constraint_type=AgenticConstraintType.CHANGED_FILES_MANIFEST_FINAL,
        scope=ConstraintScope.FINAL_OUTPUT,
        description=(
            "Your final message must contain a block introduced by the line 'CHANGED FILES MANIFEST' where "
            "each following entry line matches '(M|A|D) <path> :: <one-line rationale>'. List exactly the "
            "repository files you mutated — no omissions, no scratch files, no duplicates."
        ),
        verifier_approach="structural",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
        # 2026-08-12: final-message deliverable owed even when the trajectory
        # ends silently (see FINAL_OUTPUT handling in verifiers/trajectory.py).
        anchor_text_required=True,
    ),
    AgenticConstraintType.IMPACT_ASSESSMENT_FINAL_LINE: AgenticConstraint(
        constraint_type=AgenticConstraintType.IMPACT_ASSESSMENT_FINAL_LINE,
        scope=ConstraintScope.FINAL_OUTPUT,
        description=(
            "Your final message must contain exactly one line at column 0 of the exact form "
            "'IMPACT: files=<int> | public_api=<yes|no> | behavior_change=<yes|no> | risk=<low|medium|high>', "
            "where files= equals the number of distinct repository files you edited (scratch files under "
            "/tmp/ excluded). Any declaration is permitted — the values are yours to assess."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
        # 2026-08-12: final-message deliverable owed even when the trajectory
        # ends silently (see FINAL_OUTPUT handling in verifiers/trajectory.py).
        anchor_text_required=True,
    ),
    AgenticConstraintType.EDGE_CASE_CHECKLIST_BLOCK: AgenticConstraint(
        constraint_type=AgenticConstraintType.EDGE_CASE_CHECKLIST_BLOCK,
        scope=ConstraintScope.FINAL_OUTPUT,
        description=(
            "Your final message must contain a block starting with the line 'EDGE_CASES:' followed "
            "immediately by two or more consecutive lines, each of the form "
            "'- [<covered|uncovered|n/a>] <description> :: <test|reasoning|manual-run>=<evidence>'. "
            "Descriptions must be unique. Marking every entry uncovered is fully compliant — enumerate "
            "honestly rather than adding code or tests."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
        # 2026-08-12: final-message deliverable owed even when the trajectory
        # ends silently (see FINAL_OUTPUT handling in verifiers/trajectory.py).
        anchor_text_required=True,
    ),
    AgenticConstraintType.ISSUE_SUMMARY_VERBATIM_ECHO: AgenticConstraint(
        constraint_type=AgenticConstraintType.ISSUE_SUMMARY_VERBATIM_ECHO,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "Your first message must contain a single line 'ISSUE: <summary>' of at most 120 characters. "
            "Every later 'ISSUE: ' line you write must repeat that summary character-for-character — no "
            "paraphrase, truncation, or capitalization changes — and you must re-emit it immediately before "
            "your first file edit and in your final message."
        ),
        verifier_approach="string_prefix",
        compatible_injection_modes=_EARLY_ONLY_MODES,
    ),
    AgenticConstraintType.STATE_LEDGER_MONOTONIC_CARRYOVER: AgenticConstraint(
        constraint_type=AgenticConstraintType.STATE_LEDGER_MONOTONIC_CARRYOVER,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "Every message that issues a tool call must end with a single ledger line "
            "'STATE: edits=<int> files=<int> repro=<FAIL|PASS|UNRUN>' where edits counts your file-edit tool "
            "calls so far, files counts the distinct files you have edited so far, and both never decrease."
        ),
        verifier_approach="structural",
        conflict_with=["checkpoint_every_nth_tool_call"],
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.CUMULATIVE_TOUCHED_FILES_MANIFEST: AgenticConstraint(
        constraint_type=AgenticConstraintType.CUMULATIVE_TOUCHED_FILES_MANIFEST,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "From the message containing your first repository-file edit onward, every assistant message "
            "(including the final one) must contain exactly one line 'TOUCHED: <path>[, <path>...]' listing, "
            "in sorted order without duplicates, every repository file you have edited so far, "
            "using repo-relative paths."
        ),
        verifier_approach="structural",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.PATCH_REVISION_COUNTER_PER_FILE: AgenticConstraint(
        constraint_type=AgenticConstraintType.PATCH_REVISION_COUNTER_PER_FILE,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "Immediately before every file-edit tool call, emit a line 'PATCH <path> rev=<k>' where k is 1 "
            "for your first edit to that path and increments by exactly 1 for each later edit to the same "
            "path, tracked per file and never reset. Spell the path identically across all its PATCH lines."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.CHECKPOINT_EVERY_NTH_TOOL_CALL: AgenticConstraint(
        constraint_type=AgenticConstraintType.CHECKPOINT_EVERY_NTH_TOOL_CALL,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "In the turn issuing your {n}th, {n}x2th, {n}x3th... cumulative tool call, the narration "
            "immediately preceding that call must contain exactly one line "
            "'CHECKPOINT <seq> | calls=<cumulative call count> | goal: <current goal>' where seq starts at 1 "
            "and increments by exactly 1 per checkpoint, with no gaps, repeats, or resets."
        ),
        parameters={"n": 5},
        verifier_approach="structural",
        conflict_with=["state_ledger_monotonic_carryover"],
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.DUPLICATE_COMMAND_RERUN_TAG: AgenticConstraint(
        constraint_type=AgenticConstraintType.DUPLICATE_COMMAND_RERUN_TAG,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "When you issue a shell command string identical (after whitespace normalization) to one you "
            "already executed, precede the call with a line 'RERUN #<occurrence> (same as call #<index>)' "
            "giving the occurrence number (2 for the second run) and the call index of the previous "
            "execution. First executions must not carry a RERUN tag."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.SINGLE_TOOL_CALL_PER_MESSAGE: AgenticConstraint(
        constraint_type=AgenticConstraintType.SINGLE_TOOL_CALL_PER_MESSAGE,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "Issue at most one tool call per assistant message, and accompany every tool call with at least "
            "one line of narration prose (5 or more words) outside code fences and tag lines. Never batch "
            "multiple tool calls into one message or emit a bare tool call with no prose."
        ),
        verifier_approach="structural",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.PHASE_TAG_ORDERED_LIFECYCLE: AgenticConstraint(
        constraint_type=AgenticConstraintType.PHASE_TAG_ORDERED_LIFECYCLE,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "Open every assistant message with exactly one phase tag from [PHASE:EXPLORE], [PHASE:REPRO], "
            "[PHASE:DIAGNOSE], [PHASE:PATCH], [PHASE:VERIFY], [PHASE:CLEANUP] as the very first characters. "
            "The first message must be [PHASE:EXPLORE]; the last must be [PHASE:CLEANUP]; and by the end the "
            "tag sequence must contain EXPLORE, REPRO, DIAGNOSE, PATCH, VERIFY, CLEANUP as an in-order "
            "subsequence (re-entering an earlier phase is allowed)."
        ),
        verifier_approach="regex",
        compatible_injection_modes=_EARLY_ONLY_MODES,
    ),
    AgenticConstraintType.VERIFICATION_CALL_AFTER_EACH_EDIT: AgenticConstraint(
        constraint_type=AgenticConstraintType.VERIFICATION_CALL_AFTER_EACH_EDIT,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "The tool call immediately following any file edit must verify it: either re-read the edited "
            "file or run a test/repro/compile command (pytest, python -m, python script, tox, npm test, "
            "go test). Never issue two edits back-to-back, and never follow an edit with an unrelated search."
        ),
        verifier_approach="structural",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.PRE_FIRST_EDIT_CALL_TALLY_ONCE: AgenticConstraint(
        constraint_type=AgenticConstraintType.PRE_FIRST_EDIT_CALL_TALLY_ONCE,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "In the message containing your first file-edit tool call, before that call, emit exactly one "
            "line 'PRE_EDIT_TOOL_CALLS: <n>' where n is the number of tool calls you issued strictly before "
            "that first edit (0 is legal and written as 'PRE_EDIT_TOOL_CALLS: 0'). Emit this line exactly "
            "once in the whole trajectory — never before other calls and never with a fuzzy value."
        ),
        verifier_approach="structural",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.SUCCESS_CLAIM_OBSERVATION_QUOTE: AgenticConstraint(
        constraint_type=AgenticConstraintType.SUCCESS_CLAIM_OBSERVATION_QUOTE,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "Any message claiming success ('is fixed', 'now passes', 'all tests pass', 'works now', 'the fix "
            "works') must also contain at least one line 'EVIDENCE: call#<k> :: \"<quote>\"' where call #k is "
            "an earlier command execution and the quote occurs verbatim in that call's output. Never claim "
            "success without citing observed output."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.PREEXISTING_FAILURE_BASELINE_CITATION: AgenticConstraint(
        constraint_type=AgenticConstraintType.PREEXISTING_FAILURE_BASELINE_CITATION,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "Any message calling a failure 'pre-existing' or 'unrelated to my/this change' must also contain "
            "a line 'PREEXISTING: <test node id> | baseline=call#<k> | status=FAIL' where call #k is a "
            "test run you executed before your first file edit and the node id appears verbatim in its "
            "output. Without such a pre-edit baseline run, do not use those phrases at all."
        ),
        verifier_approach="regex",
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.NO_USER_QUESTIONS_ASSUMPTION_TAG: AgenticConstraint(
        constraint_type=AgenticConstraintType.NO_USER_QUESTIONS_ASSUMPTION_TAG,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "This rollout is fully automated: never address a question to the user (no sentence ending in "
            "'?' outside code blocks, tool arguments, or quoted text). When information is missing, emit a "
            "line 'ASSUMPTION: <statement> | basis=<file|test|docs|convention>' and proceed."
        ),
        verifier_approach="regex",
    ),
    AgenticConstraintType.REPO_RELATIVE_PATHS_IN_NARRATION: AgenticConstraint(
        constraint_type=AgenticConstraintType.REPO_RELATIVE_PATHS_IN_NARRATION,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "Write every file path mentioned in your prose repo-relative: no leading '/', no '~', and no "
            "container prefixes such as '/testbed/' or '/workspace/'. Absolute paths are permitted only "
            "inside actual shell command strings."
        ),
        verifier_approach="regex",
    ),

    # ── SWE-bench batch 3 (2026-08-12 repairs) ────────────────────────────────

    AgenticConstraintType.NO_OUTCOME_CLAIMS_BEFORE_EXECUTION: AgenticConstraint(
        # Repair of no_outcome_claims_in_opening (activates=0%: no agent claims
        # outcomes in its opening). The real fork is claiming success after an
        # edit but before running anything — a common unconstrained behavior.
        constraint_type=AgenticConstraintType.NO_OUTCOME_CLAIMS_BEFORE_EXECUTION,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "Until at least one shell command has executed and returned an observation in this trajectory, "
            "no message may claim any outcome: no 'I have fixed/patched/resolved/verified', no 'all tests "
            "pass', no 'the bug is fixed', no 'the fix works', and never the word 'successfully'. After the "
            "first observed command execution these claims become unrestricted. Future-tense planning "
            "language ('I will fix', 'this should resolve') is always allowed."
        ),
        verifier_approach="structural",
        conflict_with=["success_claim_observation_quote"],
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.EDIT_CLASS_DECLARATION_TAG: AgenticConstraint(
        # Repair of config_file_edit_declaration_tag (match=3%: SWE-bench fixes
        # almost never touch packaging/CI config, so the tag never fired).
        # Trigger broadened from rare config edits to every edit.
        constraint_type=AgenticConstraintType.EDIT_CLASS_DECLARATION_TAG,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "Immediately before every file edit, emit exactly one line 'EDIT_CLASS: <path> | "
            "class=<source|test|config|docs>' naming the edit target. Determine class by this precedence: "
            "config if the basename is setup.py, setup.cfg, pyproject.toml, MANIFEST.in, tox.ini, "
            "pytest.ini, Dockerfile, Makefile, or requirements*.txt, or the path is under "
            ".github/workflows/; else test if the basename starts with 'test_' or ends with '_test.py', "
            "or the path contains a /tests/ or /test/ directory; else docs if the extension is .md, .rst, "
            "or .txt; else source. Non-edit tool calls must not carry this line."
        ),
        verifier_approach="regex",
        conflict_with=["config_file_edit_declaration_tag"],
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.ABS_PATH_SCOPE_TAG: AgenticConstraint(
        # Repair of out_of_repo_path_access_tag (activates=5%: agents almost
        # never touch /usr, /etc, or site-packages). Trigger broadened to any
        # absolute path, which SWE-bench agents use constantly (/testbed/...).
        constraint_type=AgenticConstraintType.ABS_PATH_SCOPE_TAG,
        scope=ConstraintScope.CODE_STEPS,
        description=(
            "Immediately before any tool call whose arguments contain an absolute path (a token starting "
            "with '/'), emit exactly one line 'PATH_SCOPE: <path> | zone=<repo|tmp|system>' naming the "
            "first absolute path in the call verbatim. zone=tmp if it starts with /tmp/; zone=system if it "
            "starts with /usr/, /opt/, /etc/, or /var/, or contains site-packages/ or dist-packages/; "
            "otherwise zone=repo. Calls whose arguments contain no absolute path must not carry this line."
        ),
        verifier_approach="regex",
        conflict_with=["out_of_repo_path_access_tag", "repo_relative_paths_in_narration"],
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),
    AgenticConstraintType.RAW_OUTPUT_QUARANTINE: AgenticConstraint(
        # Repair of no_simulated_tool_output_in_opening (activates=22%: only
        # binds when the issue text tempts the agent to echo output in its
        # opening). Broadened to all messages: agents routinely paste tally
        # lines and tracebacks into bare prose after real runs.
        constraint_type=AgenticConstraintType.RAW_OUTPUT_QUARANTINE,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "In every message, text that looks like command, test, or git output — 'Traceback (most recent "
            "call last)', 'File \"...\", line N', 'test session starts', 'N passed/failed/skipped' counts, "
            "'exit_code=', '$ cmd' lines, git status phrases — must appear only inside a fenced code "
            "block, an inline `code span`, or a '>' blockquote line. Never present output-like text as "
            "bare prose, whether quoted from the issue or from a real observation."
        ),
        verifier_approach="regex",
        conflict_with=["no_simulated_tool_output_in_opening"],
        format_regimes=[FormatRegime.NATIVE_TOOL_CALL, FormatRegime.BASH_SCAFFOLD],
    ),

    # ── Real-traffic coverage batch (2026-08-14) ──────────────────────────────
    # Grounded in Fay Wang's real-traffic fc.1.1 dataset (364 audited failures)
    # and the kernelbench NVBug case; per-constraint trace IDs live in
    # reports/real_traffic_if_format_coverage.md. Each entry names its pattern.

    AgenticConstraintType.EXACT_SENTINEL_REPLY: AgenticConstraint(
        # Pattern P1 (89 traces, mostly OpenClaw heartbeats): sentinel token
        # required alone, model appends status prose, analyses, or code dumps
        # ("Okay, HEARTBEAT_OK", report + "[SILENT]").
        constraint_type=AgenticConstraintType.EXACT_SENTINEL_REPLY,
        # 2026-08-14 SWE pick-in: rescoped from poll-turn replies (chat-only
        # phrasing, judged inapplicable on SWE tasks) to the final message.
        # ALL_STEPS scope: the verifier branches internally — owed-exact check
        # on the final message, no-early-emission trigger on earlier ones
        # (FINAL_OUTPUT scope hid mid-work token emissions the judge flags).
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "After completing the work, your final message must be exactly `{sentinel_token}` — "
            "no status prose, no explanation, no timestamps, no repeated token, and no other "
            "content of any kind. Only the final message is subject to this rule; do not send "
            "a message consisting solely of `{sentinel_token}` before the work is complete."
        ),
        verifier_approach="string_match",
        parameters={"sentinel_token": "HEARTBEAT_OK"},
    ),

    AgenticConstraintType.CLOSED_TAG_VERDICT_REPLY: AgenticConstraint(
        # Pattern P2 (56 traces, all Claude Code security-monitor calls):
        # '<severity>N</severity> ONLY' answered with unclosed '<severity>50'
        # or a prose analysis before/after the tag.
        constraint_type=AgenticConstraintType.CLOSED_TAG_VERDICT_REPLY,
        # 2026-08-14 SWE pick-in: rescoped to the final message (see
        # EXACT_SENTINEL_REPLY note).
        scope=ConstraintScope.FINAL_OUTPUT,
        description=(
            "Your final message must consist only of well-formed tags of the form "
            "`<{tag_name}>value</{tag_name}>` — every opened tag closed, no text before, "
            "between, or after the tags."
        ),
        verifier_approach="regex",
        parameters={"tag_name": "severity"},
        format_regimes=[FormatRegime.TEXT_RESPONSE, FormatRegime.NATIVE_TOOL_CALL],
    ),

    AgenticConstraintType.TAGGED_SECTIONS_WELL_FORMED: AgenticConstraint(
        # Pattern P6 (16 traces): required <analysis>/<summary>-style blocks
        # emitted malformed (mismatched tags), truncated, or skipped entirely.
        constraint_type=AgenticConstraintType.TAGGED_SECTIONS_WELL_FORMED,
        scope=ConstraintScope.FINAL_OUTPUT,
        description=(
            "Your final message must contain these tagged sections, in order, each with matching "
            "opening and closing tags: {required_tags}. A missing section, a missing closing tag, "
            "or sections out of order are violations."
        ),
        verifier_approach="structural",
        parameters={"required_tags": ["analysis", "summary"]},
        conflict_with=["output_sections"],
    ),

    AgenticConstraintType.OUTPUT_ONLY_PASSTHROUGH: AgenticConstraint(
        # Pattern P4 (31 traces): 'capture stdout and send it as your reply',
        # 'output only the report or NO_ALERT' answered with hand-written
        # summaries, debug narratives, or duplicate renderings.
        constraint_type=AgenticConstraintType.OUTPUT_ONLY_PASSTHROUGH,
        scope=ConstraintScope.FINAL_OUTPUT,
        description=(
            "Your final message must consist solely of the designated artifact — the captured tool "
            "output verbatim, or exactly `{allowed_sentinel}` when there is nothing to deliver. "
            "Do not add any summary, commentary, headers, or restatement around it."
        ),
        verifier_approach="structural",
        parameters={"allowed_sentinel": "NO_ALERT"},
    ),

    AgenticConstraintType.CONTINUATION_NO_RESTART: AgenticConstraint(
        # Pattern (4 traces): 'Resume directly — do not acknowledge the summary,
        # do not recap' answered with 'I'll continue with…' / a session recap;
        # truncation continuations restarting from the top.
        constraint_type=AgenticConstraintType.CONTINUATION_NO_RESTART,
        scope=ConstraintScope.ALL_STEPS,
        description=(
            "When resuming or continuing prior work, output only the continuation: do not open with "
            "acknowledgments or recaps ('I'll continue', 'Continuing from', 'To recap', 'The session was "
            "working on…'), and do not re-emit text you already produced."
        ),
        verifier_approach="regex",
        compatible_injection_modes=_ALL_INJECTION_MODES,
    ),

    AgenticConstraintType.CONDITIONAL_REQUIRED_SENTENCE: AgenticConstraint(
        # Pattern (1 trace + irrelevance_sentinel family): 'include the sentence
        # "No material changes since last scan" if there are zero changes' —
        # condition held, sentence omitted.
        constraint_type=AgenticConstraintType.CONDITIONAL_REQUIRED_SENTENCE,
        scope=ConstraintScope.FINAL_OUTPUT,
        description=(
            "When the observed results match the condition ({condition_description}), your final message "
            "must include this exact sentence verbatim: \"{required_sentence}\""
        ),
        verifier_approach="structural",
        # 2026-08-14 SWE pick-in: default condition retargeted from scan-report
        # traffic ("zero changes") — which never occurs on SWE tasks (judged
        # inapplicable on all 32 QA tasks) — to scratch-file usage, which most
        # SWE trajectories exhibit, so the condition actually activates.
        parameters={
            "condition_description": "you created or ran any file under /tmp during the work",
            "condition_pattern": r"/tmp/[\w.\-/]+",
            "required_sentence": "Scratch files under /tmp were used during this investigation.",
        },
    ),

    AgenticConstraintType.ABS_PATHS_IN_FINAL_RESPONSE: AgenticConstraint(
        # Pattern (3 traces, Claude Code subagents): system prompt mandates
        # absolute paths in the final response; model returns bare filenames
        # (app_3.js) or repo-relative paths (src/app/page.tsx).
        constraint_type=AgenticConstraintType.ABS_PATHS_IN_FINAL_RESPONSE,
        scope=ConstraintScope.FINAL_OUTPUT,
        description=(
            "Every file path mentioned in your final message must be an absolute path from the "
            "filesystem root (e.g. /repo/src/app.py) — never a bare filename or a relative path."
        ),
        verifier_approach="regex",
        conflict_with=["repo_relative_paths_in_narration"],
    ),
}

# ── Conversational registry ───────────────────────────────────────────────────
# Injectable instruction templates for every ConversationalConstraintType.
# description is the model-facing text after .format(**parameters); parameter
# keys match what the corresponding verifier reads from constraint_params
# (infrastructure/verifiers/if_format.py). Types whose full semantics need an
# LLM judge (language) note it in verifier_approach.

CONVERSATIONAL_CONSTRAINT_REGISTRY: dict[ConversationalConstraintType, ConversationalConstraint] = {
    ConversationalConstraintType.WORD_COUNT_MAX: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.WORD_COUNT_MAX,
        description="Your entire response must be at most {max_words} words.",
        parameters={"max_words": 150},
        verifier_approach="counter",
    ),
    ConversationalConstraintType.WORD_COUNT_MIN: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.WORD_COUNT_MIN,
        description="Your entire response must be at least {min_words} words.",
        parameters={"min_words": 50},
        verifier_approach="counter",
    ),
    ConversationalConstraintType.JSON_FORMAT: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.JSON_FORMAT,
        description="Your entire response must be a single valid JSON object (a ```json code fence around it is allowed, but no prose outside it).",
        verifier_approach="json_parse",
    ),
    ConversationalConstraintType.BULLET_LIST: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.BULLET_LIST,
        description="Present the key points as a bullet list (lines starting with '-', '*', or '•').",
        verifier_approach="regex",
    ),
    ConversationalConstraintType.NUMBERED_LIST: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.NUMBERED_LIST,
        description="Present the key points as a numbered list ('1.', '2.', ...).",
        verifier_approach="regex",
    ),
    ConversationalConstraintType.LANGUAGE: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.LANGUAGE,
        description="Write your entire response in {language}.",
        parameters={"language": "English"},
        verifier_approach="llm_judge",  # static check only verifies non-empty
    ),
    ConversationalConstraintType.KEYWORD_INCLUDE: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.KEYWORD_INCLUDE,
        description="Your response must include all of these terms verbatim: {keywords}.",
        parameters={"keywords": ["result", "because"]},
        verifier_approach="string_match",
    ),
    ConversationalConstraintType.KEYWORD_FORBIDDEN: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.KEYWORD_FORBIDDEN,
        description="Your response must not contain any of these terms: {keywords}.",
        parameters={"keywords": ["obviously", "simply"]},
        verifier_approach="string_match",
    ),
    ConversationalConstraintType.SECTION_HEADERS: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.SECTION_HEADERS,
        description="Organize your response under markdown section headers (lines starting with '#', '##', or '###').",
        verifier_approach="regex",
        conflict_with=["json_format"],
    ),
    ConversationalConstraintType.SENTENCE_COUNT: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.SENTENCE_COUNT,
        description="Your response must contain exactly {count} sentences.",
        parameters={"count": 5},
        verifier_approach="counter",
    ),
    ConversationalConstraintType.RESPONSE_PREFIX: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.RESPONSE_PREFIX,
        description="Your response must begin with the exact text: {prefix}",
        parameters={"prefix": "ANSWER:"},
        verifier_approach="string_prefix",
        conflict_with=["tldr_prefix", "monotonic_step_index_header"],
    ),
    ConversationalConstraintType.RESPONSE_SUFFIX: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.RESPONSE_SUFFIX,
        description="Your response must end with the exact text: {suffix}",
        parameters={"suffix": "END OF REPORT"},
        verifier_approach="string_suffix",
        conflict_with=["confidence_level_suffix"],
    ),
    ConversationalConstraintType.TABLE_FORMAT_REQUIRED: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.TABLE_FORMAT_REQUIRED,
        description="Present the main results as a markdown table (rows delimited with '|').",
        verifier_approach="regex",
        conflict_with=["json_format"],
    ),
    ConversationalConstraintType.CODE_BLOCK_LANGUAGE_TAG: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.CODE_BLOCK_LANGUAGE_TAG,
        description="Every fenced code block must declare its language (```python, ```bash, ...); bare ``` fences are not allowed.",
        verifier_approach="regex",
    ),
    ConversationalConstraintType.MAX_LIST_NESTING_DEPTH: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.MAX_LIST_NESTING_DEPTH,
        description="Lists may be nested at most {max_depth} levels deep.",
        parameters={"max_depth": 2},
        verifier_approach="structural",
    ),
    ConversationalConstraintType.NO_CONTRACTIONS: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.NO_CONTRACTIONS,
        description="Do not use contractions anywhere in your response (write 'do not', never \"don't\").",
        verifier_approach="regex",
    ),
    ConversationalConstraintType.ACTION_ITEMS_CHECKBOX: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.ACTION_ITEMS_CHECKBOX,
        description="List all action items as markdown checkboxes ('- [ ] item' or '- [x] item').",
        verifier_approach="regex",
    ),
    ConversationalConstraintType.TLDR_PREFIX: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.TLDR_PREFIX,
        description="Begin your response with a line starting 'TL;DR:' that summarizes the answer in one sentence.",
        verifier_approach="regex",
        conflict_with=["response_prefix", "monotonic_step_index_header"],
    ),
    ConversationalConstraintType.CONFIDENCE_LEVEL_SUFFIX: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.CONFIDENCE_LEVEL_SUFFIX,
        description="End your response with a line 'Confidence: High', 'Confidence: Medium', or 'Confidence: Low'.",
        verifier_approach="regex",
        conflict_with=["response_suffix"],
    ),
    ConversationalConstraintType.MAX_SENTENCE_LENGTH: ConversationalConstraint(
        constraint_type=ConversationalConstraintType.MAX_SENTENCE_LENGTH,
        description="No sentence in your response may exceed {max_words_per_sentence} words.",
        parameters={"max_words_per_sentence": 30},
        verifier_approach="counter",
    ),

    # ── Real-traffic coverage batch (2026-08-14) ──────────────────────────────
    # Grounded in Fay Wang's real-traffic fc.1.1 dataset and the kernelbench
    # NVBug case; see reports/real_traffic_if_format_coverage.md for trace IDs.

    ConversationalConstraintType.RESPONSE_LINE_LIMIT: ConversationalConstraint(
        # Pattern P3 (47 traces): 'fewer than 4 lines' (opencode/Claude Code
        # house rule), '한국어 4줄 이내' — answered with multi-paragraph essays.
        constraint_type=ConversationalConstraintType.RESPONSE_LINE_LIMIT,
        # 2026-08-14 trace-QA: "your entire response" is ambiguous on agentic
        # trajectories — rescoped to the final message (the graded step).
        description=(
            "Your final message must be at most {max_lines} non-empty lines of visible "
            "text (a fenced code block counts as one line)."
        ),
        parameters={"max_lines": 4},
        verifier_approach="counter",
        conflict_with=["word_count_min", "section_headers", "table_format_required"],
    ),
    ConversationalConstraintType.NO_PREAMBLE_POSTAMBLE: ConversationalConstraint(
        # Pattern P3 companion: 'no preamble/postamble', 'avoid introductions,
        # conclusions, and explanations'.
        constraint_type=ConversationalConstraintType.NO_PREAMBLE_POSTAMBLE,
        # 2026-08-14 trace-QA: judges read the old wording as banning narration
        # on every turn (unsatisfiable for agentic work) — rescoped to the
        # final message, matching the graded step and the customer intent.
        description=(
            "Your final message must not begin with a filler opener (any of: 'Great', 'Sure', "
            "'Certainly', 'Okay', 'Awesome', 'Absolutely', 'Of course', 'Sounds good', 'Happy to', "
            "'I'd be happy', 'I'll now', 'I will now', \"Let's begin/start/dive\", 'Perfect') and "
            "must not end with a follow-up offer (any of: 'Let me know…', 'Feel free…', 'Hope this "
            "helps', 'If you have/need any…', \"Don't hesitate…\", 'Happy to help', 'Anything "
            "else…'). Only these listed patterns, and only in the final message, are violations."
        ),
        verifier_approach="regex",
    ),
    ConversationalConstraintType.JSON_REQUIRED_FIELDS: ConversationalConstraint(
        # Pattern P5 (19 traces, all Hermes): strict JSON returned but required
        # top-level keys (promotion_summary, operator_review) omitted.
        constraint_type=ConversationalConstraintType.JSON_REQUIRED_FIELDS,
        anchor_text_required=True,
        # 2026-08-15 triage: judges penalized mid-trajectory JSON emissions;
        # the graded step is the final message — say so.
        description=(
            "Your final message must include a JSON object containing all of these top-level "
            "fields: {required_fields}. Omitting any of them is a violation even if the JSON "
            "parses."
        ),
        parameters={"required_fields": ["summary", "review"]},
        verifier_approach="json_parse",
    ),
    ConversationalConstraintType.FENCED_FINAL_ANSWER: ConversationalConstraint(
        # kernelbench → cpp-fence-missing (P0): answer must be delivered inside
        # a single ```{lang} fence for the harness extraction regex.
        constraint_type=ConversationalConstraintType.FENCED_FINAL_ANSWER,
        anchor_text_required=True,
        description=(
            "Deliver your entire answer as exactly one fenced code block tagged ```{lang} — all code "
            "inside that single fence, no other fences, no code outside it."
        ),
        parameters={"lang": "cpp"},
        verifier_approach="regex",
        conflict_with=["json_format", "markdown_prohibited"],
    ),
    ConversationalConstraintType.MARKDOWN_PROHIBITED: ConversationalConstraint(
        # Pattern (6 traces): 'Do not use markdown formatting', 'simple text
        # renderable inside a terminal' — answered with headings/lists/bold.
        constraint_type=ConversationalConstraintType.MARKDOWN_PROHIBITED,
        description=(
            "Write in plain text only: no markdown headings, no bullet or numbered list markers, "
            "no bold/italic markers, no tables, and no code fences."
        ),
        verifier_approach="regex",
        conflict_with=[
            "bullet_list", "numbered_list", "section_headers", "table_format_required",
            "action_items_checkbox", "code_block_language_tag", "json_format",
        ],
    ),
    ConversationalConstraintType.QUOTE_MAX_LENGTH: ConversationalConstraint(
        # Pattern (4 traces): 'strict 125-character maximum for quotes',
        # quotation marks required around verbatim source text.
        constraint_type=ConversationalConstraintType.QUOTE_MAX_LENGTH,
        # 2026-08-15 triage: judges read the old wording as covering source
        # code reproduced in fences (any code snippet = unquoted quotation),
        # and graded mid-trajectory turns. Scope to prose quotations in the
        # final message; code display is exempt.
        description=(
            "In your final message, enclose any verbatim prose quotation in double quotation "
            "marks and keep each quotation at most {max_quote_chars} characters. Code shown in "
            "code blocks or inline code spans is not a quotation and is not restricted."
        ),
        parameters={"max_quote_chars": 125},
        verifier_approach="counter",
    ),
    ConversationalConstraintType.PROHIBITED_CHARACTERS: ConversationalConstraint(
        # Pattern (2 traces): 'NO dash of ANY kind' (voice/house style);
        # hyphens and em dashes emitted anyway.
        constraint_type=ConversationalConstraintType.PROHIBITED_CHARACTERS,
        description="Do not use any of these characters anywhere in your response: {chars}",
        parameters={"chars": ["—", "–"]},
        verifier_approach="string_match",
    ),
    ConversationalConstraintType.KEYWORD_POSITION: ConversationalConstraint(
        # Pattern (1 trace): 'NAME the business naturally inside the first two
        # sentences' — name placed in the final sentence instead.
        constraint_type=ConversationalConstraintType.KEYWORD_POSITION,
        # 2026-08-14 SWE pick-in: "your response" -> "your final message" (works
        # in both chat and agentic contexts); default keyword retargeted from
        # chat traffic ("the business name") to one every SWE task can satisfy.
        description="Mention {keyword} within the first {within_sentences} sentences of your final message.",
        # "root cause" (no article): the article made the match over-strict —
        # a final opening "Root cause: ..." failed the substring check.
        parameters={"keyword": "root cause", "within_sentences": 2},
        verifier_approach="counter",
        conflict_with=["keyword_include"],
    ),
    ConversationalConstraintType.NO_EMOJI: ConversationalConstraint(
        # Persona/style pattern subset (opencode/Claude Code 'only use emojis
        # if the user explicitly requests it'; violated in 5+ traces).
        # 2026-08-14 SWE pick-in: scoped to every message so it grades on
        # agentic trajectories too. NOTE: zero natural violations on 23 SWE
        # probe traces — activates rarely on SWE; kept for eval coverage.
        constraint_type=ConversationalConstraintType.NO_EMOJI,
        description="Do not use emojis anywhere in any of your messages.",
        verifier_approach="regex",
    ),
    ConversationalConstraintType.ALLOWED_TAG_VOCABULARY: ConversationalConstraint(
        # Pattern (1 trace, TTS/voice-markup traffic): 'keep emotion tags,
        # remove noise tags' — [sigh]/[groan]/[chuckle] kept anyway.
        constraint_type=ConversationalConstraintType.ALLOWED_TAG_VOCABULARY,
        description=(
            "Only bracket tags from this list may appear in your response: {allowed_tags}. "
            "Any other [tag] is a violation."
        ),
        parameters={"allowed_tags": ["happy", "sad", "excited", "whisper"]},
        verifier_approach="regex",
    ),
}

assert len(CONVERSATIONAL_CONSTRAINT_REGISTRY) == len(ConversationalConstraintType), (
    f"CONVERSATIONAL_CONSTRAINT_REGISTRY has {len(CONVERSATIONAL_CONSTRAINT_REGISTRY)} entries "
    f"but ConversationalConstraintType has {len(ConversationalConstraintType)} members"
)


# ── Description rendering ─────────────────────────────────────────────────────

def render_description(entry, overrides: dict | None = None) -> str:
    """Render an injectable instruction from a registry entry.

    Substitutes only the entry's declared parameter placeholders — descriptions
    legitimately contain literal JSON braces, so str.format() is unusable.
    """
    text = entry.description
    params = {**entry.parameters, **(overrides or {})}
    for key, value in params.items():
        text = text.replace("{" + key + "}", str(value))
    return text


# ── Conflict symmetrization ───────────────────────────────────────────────────
# conflict_with entries are authored one-directional; make them symmetric so
# pair composition can check either side. Cross-registry references (agentic ↔
# conversational) are symmetrized across both registries.

def _symmetrize_conflicts() -> None:
    by_name: dict[str, object] = {}
    for entry in AGENTIC_CONSTRAINT_REGISTRY.values():
        by_name[entry.constraint_type.value] = entry
    for entry in CONVERSATIONAL_CONSTRAINT_REGISTRY.values():
        by_name[entry.constraint_type.value] = entry
    for name, entry in by_name.items():
        for other_name in entry.conflict_with:
            other = by_name.get(other_name)
            if other is None:
                raise ValueError(
                    f"conflict_with dangling reference: {name!r} -> {other_name!r}"
                )
            if name not in other.conflict_with:
                other.conflict_with.append(name)


_symmetrize_conflicts()


# ── Honeypot tool pool ────────────────────────────────────────────────────────
# Plausible-sounding dummy tools inserted into the tool list for
# FORBIDDEN_TOOL_ABSTENTION training examples. Names are chosen to be tempting
# (they sound useful) but clearly forbidden by the injected system prompt rule.

DUMMY_TOOL_POOL: list[str] = [
    "escalate_to_admin",
    "bypass_rate_limit",
    "force_execute",
    "get_internal_data",
    "debug_override",
    "cache_lookup_fast",
    "send_raw_request",
    "admin_query_all",
    "override_safety_check",
    "fetch_restricted_record",
    "bulk_delete",
    "impersonate_user",
]
