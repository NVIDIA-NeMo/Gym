from enum import Enum
from pydantic import BaseModel, Field

from ..if_format.constraints import InjectionMode, _ALL_INJECTION_MODES


class PreferenceTier(str, Enum):
    """The three structural tiers of tool preference instructions.

    Hard and avoidance are deterministic (rule-based trace check).
    Soft requires contextual applicability judgment (LLM judge).
    """
    HARD = "hard"
    SOFT = "soft"
    AVOIDANCE = "avoidance"


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict = {}


class ToolPreference(BaseModel):
    tier: PreferenceTier
    preferred_tool: str | None = None
    forbidden_tool: str | None = None
    applicability_condition: str | None = None
    instruction_text: str
    compatible_injection_modes: list[InjectionMode] = Field(
        default_factory=lambda: list(_ALL_INJECTION_MODES)
    )


# ── Discovered preference scenarios ──────────────────────────────────────────
# Organised by domain. Each scenario includes all three tier variants where
# applicable, and a non_triviality note explaining why the preference choice
# is not obvious from the query alone.

PREFERENCE_SCENARIOS: list[dict] = [

    # ── Enterprise knowledge management ──────────────────────────────────────
    {
        "scenario": "internal_search_vs_web_search",
        "domain": "enterprise_knowledge_management",
        "tool_relationship": "internal_vs_external",
        "non_triviality": "Many queries can be answered by either internal docs or web; the preference enforces data governance, not capability.",
        "applicable_tiers": [PreferenceTier.HARD, PreferenceTier.SOFT, PreferenceTier.AVOIDANCE],
        "hard": "Only use the internal_search tool. Do not query external sources.",
        "soft": "Prefer internal_search over web_search when the topic is covered in internal documentation.",
        "avoidance": "Do not use web_search for any query involving internal systems or processes.",
    },
    {
        "scenario": "vector_search_vs_keyword_search",
        "domain": "enterprise_knowledge_management",
        "tool_relationship": "precision_vs_recall",
        "non_triviality": "Semantic vs keyword search each have strengths; the query alone doesn't reveal which is better.",
        "applicable_tiers": [PreferenceTier.SOFT],
        "soft": "Prefer vector_search over keyword_search when the query is conceptual rather than an exact phrase lookup.",
    },
    {
        "scenario": "structured_db_vs_llm_generation",
        "domain": "enterprise_knowledge_management",
        "tool_relationship": "structured_vs_unstructured",
        "non_triviality": "Factual lookups suit the DB; open-ended synthesis suits the LLM — the query rarely makes this obvious.",
        "applicable_tiers": [PreferenceTier.HARD, PreferenceTier.SOFT],
        "hard": "Always use database_query for data retrieval. Do not use LLM generation for factual lookups.",
        "soft": "Prefer database_query over llm_knowledge when the answer is a specific fact or record.",
    },

    # ── Financial analysis ────────────────────────────────────────────────────
    {
        "scenario": "realtime_api_vs_cached_store",
        "domain": "financial_analysis",
        "tool_relationship": "realtime_vs_cached",
        "non_triviality": "Historical analysis works fine with cached data; the query often doesn't state whether recency is required.",
        "applicable_tiers": [PreferenceTier.HARD, PreferenceTier.SOFT, PreferenceTier.AVOIDANCE],
        "hard": "Only use realtime_market_api for all price lookups.",
        "soft": "Prefer cached_data_store over realtime_market_api when the query does not require live prices.",
        "avoidance": "Do not use realtime_market_api for historical analysis reports.",
    },
    {
        "scenario": "calculator_vs_llm_estimation",
        "domain": "financial_analysis",
        "tool_relationship": "precision_vs_recall",
        "non_triviality": "Some calculations require exact arithmetic; others are rough estimates. The query rarely signals which.",
        "applicable_tiers": [PreferenceTier.SOFT],
        "soft": "Use calculator_tool when numerical precision matters; use LLM estimation only for rough approximations.",
    },

    # ── Software development ──────────────────────────────────────────────────
    {
        "scenario": "sandbox_exec_vs_production_exec",
        "domain": "software_development",
        "tool_relationship": "internal_vs_external",
        "non_triviality": "Both can run code; the risk difference is not visible in the query.",
        "applicable_tiers": [PreferenceTier.HARD, PreferenceTier.AVOIDANCE],
        "hard": "Only use sandbox_bash for all code execution. Never execute in the production environment.",
        "avoidance": "Do not use production_bash under any circumstances during development.",
    },
    {
        "scenario": "code_search_vs_web_search",
        "domain": "software_development",
        "tool_relationship": "internal_vs_external",
        "non_triviality": "Both can answer implementation questions; internal code search respects proprietary code.",
        "applicable_tiers": [PreferenceTier.SOFT, PreferenceTier.AVOIDANCE],
        "soft": "Prefer code_search over web_search when the question involves this codebase's APIs or conventions.",
        "avoidance": "Do not use web_search when answering questions about internal APIs.",
    },

    # ── Customer data platform ────────────────────────────────────────────────
    {
        "scenario": "cache_lookup_vs_crm_api",
        "domain": "customer_data_platform",
        "tool_relationship": "realtime_vs_cached",
        "non_triviality": "Cache is fast and cheap; CRM has latest data. The query rarely specifies freshness requirements.",
        "applicable_tiers": [PreferenceTier.HARD, PreferenceTier.SOFT],
        "hard": "Always use crm_api for customer lookups. Do not rely on the cache.",
        "soft": "Prefer cache_lookup over crm_api when the customer record is unlikely to have changed in the last hour.",
    },
    {
        "scenario": "data_warehouse_vs_event_stream",
        "domain": "customer_data_platform",
        "tool_relationship": "realtime_vs_cached",
        "non_triviality": "Aggregated historical queries suit the warehouse; real-time activity suits the stream.",
        "applicable_tiers": [PreferenceTier.SOFT],
        "soft": "Prefer data_warehouse_query over event_stream for aggregated or historical customer data.",
    },

    # ── Content generation ────────────────────────────────────────────────────
    {
        "scenario": "internal_style_guide_vs_llm_generation",
        "domain": "content_generation",
        "tool_relationship": "internal_vs_external",
        "non_triviality": "Both produce text; internal style guide enforces brand tone that LLM may not follow.",
        "applicable_tiers": [PreferenceTier.HARD, PreferenceTier.SOFT],
        "hard": "Always look up brand guidelines via style_guide_search before generating any customer-facing content.",
        "soft": "Prefer style_guide_search over direct llm_generation for external communications.",
    },
    {
        "scenario": "internal_translation_vs_external_api",
        "domain": "content_generation",
        "tool_relationship": "internal_vs_external",
        "non_triviality": "Both translate; external APIs must not receive confidential internal content.",
        "applicable_tiers": [PreferenceTier.HARD, PreferenceTier.AVOIDANCE],
        "hard": "Only use internal_translation_model for translating internal documents.",
        "avoidance": "Do not use external_translate_api for documents containing internal or confidential content.",
    },

    # ── DevOps ────────────────────────────────────────────────────────────────
    {
        "scenario": "metrics_api_vs_log_search",
        "domain": "devops_and_infrastructure",
        "tool_relationship": "structured_vs_unstructured",
        "non_triviality": "Both can diagnose incidents; structured metrics suit quantitative thresholds, logs suit qualitative errors.",
        "applicable_tiers": [PreferenceTier.SOFT],
        "soft": "Prefer metrics_api for quantitative threshold checks; use log_search for error message investigation.",
    },
    {
        "scenario": "runbook_lookup_vs_llm_remediation",
        "domain": "devops_and_infrastructure",
        "tool_relationship": "internal_vs_external",
        "non_triviality": "LLM can suggest fixes; runbooks encode the approved procedures. The query rarely specifies which is required.",
        "applicable_tiers": [PreferenceTier.HARD, PreferenceTier.SOFT],
        "hard": "Always consult runbook_search before taking any remediation action.",
        "soft": "Prefer runbook_search over llm_remediation for known incident types.",
    },

    # ── Scientific research ───────────────────────────────────────────────────
    {
        "scenario": "paper_search_vs_llm_knowledge",
        "domain": "scientific_research",
        "tool_relationship": "structured_vs_unstructured",
        "non_triviality": "LLM knowledge may be outdated or hallucinated; paper search provides citable sources.",
        "applicable_tiers": [PreferenceTier.HARD, PreferenceTier.SOFT],
        "hard": "Always use paper_search to support factual claims. Do not cite LLM internal knowledge.",
        "soft": "Prefer paper_search over llm_knowledge for empirical claims that require citation.",
    },

    # ── Agentic coding discipline (from customer_cursor / customer_lovable / deepswe) ──
    # Grounded in real customer traces; see feedbacks/ mapping.md files.
    {
        # customer_cursor → shell-for-file-io: 3/4 output-bearing traces used
        # shell grep/cat/find/sed despite Read/Grep/Glob being available and the
        # Shell tool's own description prohibiting file-operation shell usage.
        # ATTRIBUTED prevalence 50% across 125 rollouts.
        "scenario": "dedicated_file_tools_vs_shell",
        "domain": "software_development",
        "tool_relationship": "dedicated_vs_generic",
        "non_triviality": "Shell can perform file operations but customers mandate dedicated tools for predictability, auditability, and tool-specific retry semantics. The model defaults to shell even after many successful dedicated-tool calls in the same trajectory.",
        "applicable_tiers": [PreferenceTier.HARD, PreferenceTier.AVOIDANCE, PreferenceTier.SOFT],
        "hard": "For all file reading, searching, and editing, only use the dedicated tools (Read, Grep, Glob, StrReplace, Write). Do not use shell commands such as cat, head, tail, grep, find, sed, or awk for file operations.",
        "avoidance": "Do not use shell/bash commands for file reading (cat, head, tail), code searching (grep, find), or file editing (sed, awk, echo redirection). Use the dedicated file tools instead.",
        "soft": "Prefer dedicated file tools (Read, Grep, Glob) over shell commands when reading or searching files. Reserve shell for operations the dedicated tools cannot perform.",
    },
    {
        # customer_cursor → post-edit-lint-skipped: immutablejs trace, 9
        # StrReplace edits and zero ReadLints calls across 175 messages. The
        # model substituted node --check but the mandate named the tool
        # explicitly. ATTRIBUTED prevalence 42% of tasks.
        "scenario": "lint_check_after_edit",
        "domain": "software_development",
        "tool_relationship": "sequential_mandatory",
        "non_triviality": "A lint pass catches errors the model's static analysis misses; without a mandatory post-edit call the model tends to declare edits correct and move on without verification.",
        "applicable_tiers": [PreferenceTier.HARD, PreferenceTier.AVOIDANCE],
        "hard": "After every substantive code edit (any StrReplace, Write, or patch application), call the lint tool on the edited files before claiming the task is done or moving to the next step.",
        "avoidance": "Do not declare a task complete or move to subsequent work after a substantive code edit without first running the lint tool on the modified files. Substitute verification tools (compiler, node --check) do not satisfy this requirement.",
    },
    {
        # customer_lovable → plan-create-in-build-mode: plan--create called on
        # 5/10 well-scoped prompts (header, hero, features… fully specified),
        # producing 0 code edits; GLM-5.1 implemented directly on 4 of those 5.
        "scenario": "build_mode_implement_vs_plan",
        "domain": "software_development",
        "tool_relationship": "context_dependent_avoidance",
        "non_triviality": "A plan step creates an approval gate that stalls execution; calling it on fully-specified requests is a hard failure for the customer. The model cannot distinguish plan-worthy ambiguity from surface complexity.",
        "applicable_tiers": [PreferenceTier.SOFT, PreferenceTier.AVOIDANCE],
        "soft": "Call the plan or propose tool only when the task is large or genuinely ambiguous. For small, fully-specified requests with explicit requirements, implement directly without a planning step.",
        "avoidance": "Do not call the plan or propose tool when all requirements are explicit and the task is clearly scoped. Implement directly.",
    },
    {
        # deepswe → commit-when-done-skipped: 5 grounded exemplars where the
        # USER-turn instruction ("commit everything when you are done") was
        # visible at the final decision point yet git commit was never called.
        # In pebble, the model explicitly cited the system-default ("NEVER
        # commit unless asked") over the user's standing explicit ask.
        "scenario": "git_commit_on_task_completion",
        "domain": "software_development",
        "tool_relationship": "sequential_mandatory",
        "non_triviality": "The system prompt default ('never commit unless explicitly asked') conflicts with an explicit USER task instruction to commit. The model must treat the explicit per-task ask as the governing rule; this is non-obvious when the default is stated emphatically.",
        "applicable_tiers": [PreferenceTier.HARD],
        "hard": "When the task instruction explicitly asks you to commit your work (e.g. 'commit everything when you are done'), create a new branch and run git commit with all changes before declaring the task complete. An explicit task-level commit instruction overrides any system-level 'do not commit' default.",
    },

    # ── Multi-modal processing ────────────────────────────────────────────────
    {
        "scenario": "vision_api_vs_ocr",
        "domain": "multi_modal_processing",
        "tool_relationship": "cost_vs_quality",
        "non_triviality": "Vision API is more capable but expensive; OCR suffices for plain text documents.",
        "applicable_tiers": [PreferenceTier.SOFT],
        "soft": "Prefer ocr_tool over vision_api for documents that are predominantly plain text without complex layouts.",
    },
    {
        "scenario": "high_cost_api_vs_low_cost_fallback",
        "domain": "multi_modal_processing",
        "tool_relationship": "cost_vs_quality",
        "non_triviality": "Budget constraints drive preference, not capability. The query cannot signal cost intent.",
        "applicable_tiers": [PreferenceTier.HARD, PreferenceTier.SOFT],
        "hard": "Only use low_cost_embedding for all embedding tasks.",
        "soft": "Prefer low_cost_embedding over premium_embedding when the downstream task tolerates minor quality reduction.",
    },
]


# ── Canonical instruction templates per tier ──────────────────────────────────
# The data pipeline generates diverse phrasings from these; verifiers check
# against the tier semantics (trace inspection), not the phrasing.

PREFERENCE_INSTRUCTION_TEMPLATES: dict[PreferenceTier, list[str]] = {
    PreferenceTier.HARD: [
        "Only use the {tool} tool.",
        "You must exclusively call {tool} for all {task_type}.",
        "Use only {tool}. Do not invoke any other tool.",
        "Always use {tool} for {task_type}. No exceptions.",
        "All {task_type} must go through {tool}.",
    ],
    PreferenceTier.SOFT: [
        "When answering questions that require {capability}, prefer {preferred_tool} over {alternative_tool}.",
        "Use {preferred_tool} when {condition}, but feel free to use {alternative_tool} otherwise.",
        "Prefer {preferred_tool} when both {preferred_tool} and {alternative_tool} could answer the query.",
        "Default to {preferred_tool} for {task_type}; only use {alternative_tool} when {preferred_tool} is insufficient.",
        "When {condition} applies, {preferred_tool} is the better choice over {alternative_tool}.",
    ],
    PreferenceTier.AVOIDANCE: [
        "Do not use the {tool} tool under any circumstances.",
        "Avoid calling {tool} directly; route all {task_type} through {alternative}.",
        "Never invoke {tool}.",
        "{tool} is prohibited. Use {alternative} instead.",
        "Do not call {tool} for any reason.",
    ],
}
