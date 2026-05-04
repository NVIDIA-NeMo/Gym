/**
 * Benchmarks data — source of truth for the Benchmark Catalog.
 *
 * One entry per shipped env in `resources_servers/`. The `verified` flag is
 * conservative-by-default (matches the add-verified-flag pre-commit hook): an
 * env is listed `verified: false` unless its YAML config says otherwise.
 * To flip an env to verified=true here, the env's
 * resources_servers/<name>/configs/<name>.yaml must already say `verified: true`.
 *
 * Fields are deliberately minimal; per-env READMEs are the deep source. The
 * catalog page filters on these fields so users can find the env that matches
 * their task without scrolling 67 rows.
 */

export type Domain =
  | "math"
  | "coding"
  | "agent"
  | "knowledge"
  | "instruction_following"
  | "long_context"
  | "safety"
  | "games"
  | "multi_modal"
  | "other";

export type ScorerPattern =
  | "exact-match"
  | "llm-judge"
  | "code-execution"
  | "structural"
  | "stateful"
  | "step-based"
  | "custom-domain";

export interface Benchmark {
  id: string;
  name: string;
  /** One-line description of what the env scores. */
  task: string;
  domain: Domain[];
  /** Verifier pattern. */
  pattern: ScorerPattern;
  /**
   * Whether the env's YAML config asserts verified: true. Defaults to false per
   * the add-verified-flag pre-commit hook.
   */
  verified: boolean;
  /** Source directory under resources_servers/. */
  source_path: string;
}

const REPO_BASE = "https://github.com/NVIDIA-NeMo/Gym/tree/main";

export function benchmarkSourceUrl(b: Benchmark): string {
  return `${REPO_BASE}/${b.source_path}`;
}

const ENV = (
  name: string,
  task: string,
  domain: Domain[],
  pattern: ScorerPattern,
  verified = false,
): Benchmark => ({
  id: name,
  name,
  task,
  domain,
  pattern,
  verified,
  source_path: `resources_servers/${name}`,
});

export const BENCHMARKS: Benchmark[] = [
  // Math
  ENV("math_with_judge", "Math problems graded by an LLM judge or math-verify library", ["math"], "llm-judge"),
  ENV("math_with_code", "Math problems where the model uses Python as a reasoning tool", ["math", "coding"], "exact-match"),
  ENV("math_advanced_calculations", "Advanced calculations with library equivalence", ["math"], "exact-match"),
  ENV("math_with_autograder", "Math problems graded by an autograder", ["math"], "code-execution"),
  ENV("math_proof_judgement", "Math proof judgement", ["math"], "llm-judge"),
  ENV("math_formal_lean", "Lean 4 formal-proof tasks (compiler-in-the-loop)", ["math"], "custom-domain"),
  ENV("proof_judge", "Proof grading via LLM judge", ["math"], "llm-judge"),
  ENV("proof_genselect", "Proof generation with selection", ["math"], "custom-domain"),
  ENV("proof_verification", "Formal proof verification", ["math"], "custom-domain"),

  // Knowledge / MCQA
  ENV("mcqa", "Multiple-choice QA with letter extraction", ["knowledge"], "exact-match"),
  ENV("gpqa_diamond", "Graduate-level science MCQA (GPQA Diamond)", ["knowledge"], "exact-match"),
  ENV("multichallenge", "Multi-challenge benchmark", ["knowledge"], "llm-judge"),

  // Coding
  ENV("code_gen", "LiveCodeBench-format code generation", ["coding"], "code-execution"),
  ENV("competitive_coding_challenges", "Competitive coding eval via CCC", ["coding"], "code-execution"),
  ENV("bird_sql", "BIRD SQL execution against DB fixtures", ["coding"], "code-execution"),
  ENV("text_to_sql", "Text-to-SQL with execution", ["coding"], "code-execution"),
  ENV("spider2_lite", "Spider 2.0 Lite SQL execution", ["coding"], "code-execution"),
  ENV("swerl_gen", "SWE-RL: patch application + test execution", ["coding"], "code-execution"),
  ENV("swerl_llm_judge", "SWE-RL: domain LLM judge", ["coding"], "llm-judge"),

  // Instruction following
  ENV("instruction_following", "IFEval-style constraint checks", ["instruction_following"], "structural"),
  ENV("ifbench", "AllenAI IFBench instruction-following", ["instruction_following"], "structural"),
  ENV("format_verification", "Output-format constraint checks (citations, lists)", ["instruction_following"], "structural"),

  // Agent / tool use
  ENV("xlam_fc", "Function-call argument matching", ["agent"], "structural"),
  ENV(
    "single_step_tool_use_with_argument_comparison",
    "Single-step tool-call argument comparison",
    ["agent"],
    "structural",
  ),
  ENV("structured_outputs", "JSON schema validation against expected output shape", ["agent", "other"], "structural"),
  ENV("structeval", "Argument and structure comparison", ["agent", "other"], "structural"),
  ENV("ns_tools", "ns_tools agent benchmark", ["agent"], "stateful"),
  ENV("workplace_assistant", "Multi-toolkit agent eval (email, calendar, CRM)", ["agent"], "stateful"),
  ENV("calendar", "Calendar tool dialog with stateful scheduling", ["agent"], "stateful"),
  ENV("tavily_search", "Web search with verified answer extraction", ["agent", "knowledge"], "stateful"),
  ENV("google_search", "Web search via Google", ["agent", "knowledge"], "stateful"),
  ENV("finance_sec_search", "Finance SEC document search", ["agent", "knowledge"], "stateful"),
  ENV("newton_bench", "Newton physics-reasoning with stateful tools", ["agent", "math"], "stateful"),
  ENV("aviary", "FutureHouse Aviary multi-turn tool-calling", ["agent"], "stateful"),
  ENV("openenv", "OpenEnv configs (coding, echo, maze)", ["agent"], "stateful"),
  ENV("terminus_judge", "Terminus agent rollout judge", ["agent"], "llm-judge"),
  ENV("browsecomp_advanced_harness", "BrowseComp advanced harness", ["agent", "knowledge"], "stateful"),
  ENV("indirect_prompt_injection", "Indirect prompt-injection adversarial", ["safety", "agent"], "custom-domain"),

  // Reference / examples
  ENV("example_single_tool_call", "Reference: single tool call", ["agent"], "stateful"),
  ENV("example_multi_step", "Reference: multi-step tool calling", ["agent"], "stateful"),
  ENV("example_session_state_mgmt", "Reference: session-keyed state (counter pattern)", ["agent"], "stateful"),

  // Long context
  ENV("ruler", "RULER long-context benchmark", ["long_context"], "exact-match"),
  ENV("mrcr", "MRCR long-context benchmark", ["long_context"], "exact-match"),
  ENV("omniscience", "Omniscience benchmark", ["long_context"], "llm-judge"),

  // Games / step-based
  ENV("blackjack", "Blackjack — Gymnasium-style episodic env", ["games"], "step-based"),
  ENV("gymnasium", "Gymnasium-style env base", ["games"], "step-based"),
  ENV("reasoning_gym", "Reasoning Gym wrapper", ["other"], "custom-domain"),

  // Custom domains
  ENV("rdkit_chemistry", "Molecule equivalence via RDKit", ["other"], "custom-domain"),
  ENV("ether0", "Ether0 chemistry / safety", ["safety"], "custom-domain"),
  ENV("aalcr", "Legal/contract reasoning (AALCR)", ["other"], "custom-domain"),
  ENV("cvdp", "Comprehensive Verilog Design Problems", ["coding"], "custom-domain"),
  ENV("nvarc", "NVARC benchmark", ["other"], "custom-domain"),
  ENV("arc_agi", "ARC-AGI benchmark", ["other"], "custom-domain"),
  ENV("gdpval", "GDPVal real-world professional knowledge-work eval", ["other"], "custom-domain"),

  // Safety
  ENV("abstention", "Abstention scoring (model declines unanswerable)", ["safety"], "custom-domain"),
  ENV("jailbreak_detection", "Jailbreak-detection scoring", ["safety"], "custom-domain"),
  ENV("over_refusal_detection", "Over-refusal scoring (declines benign)", ["safety"], "custom-domain"),
  ENV("xstest", "XSTest safety / over-refusal", ["safety"], "custom-domain"),

  // Multi-modal
  ENV("labbench2_vlm", "LAB-Bench v2 VLM (vision-language)", ["multi_modal"], "exact-match"),
  ENV("vlm_eval_kit", "VLM evaluation kit", ["multi_modal"], "custom-domain"),
  ENV("asr_with_pc", "ASR with punctuation and capitalization", ["multi_modal"], "exact-match"),

  // Other
  ENV("equivalence_llm_judge", "Generic configurable LLM-as-judge", ["other"], "llm-judge"),
  ENV("equivalence_rule", "Rule-based equivalence (no LLM)", ["other"], "exact-match"),
  ENV("genrm_compare", "Pairwise comparison via generative reward model", ["other"], "llm-judge"),
  ENV("circle_click", "Circle click benchmark", ["games"], "step-based"),
  ENV("circle_count", "Circle count benchmark", ["games"], "step-based"),
  ENV("terminal_multi_harness", "Terminal multi-harness", ["coding", "agent"], "stateful"),
];

export const DOMAIN_LABELS: Record<Domain, string> = {
  math: "Math",
  coding: "Coding",
  agent: "Agent",
  knowledge: "Knowledge",
  instruction_following: "Instruction following",
  long_context: "Long context",
  safety: "Safety",
  games: "Games",
  multi_modal: "Multi-modal",
  other: "Other",
};

export const PATTERN_LABELS: Record<ScorerPattern, string> = {
  "exact-match": "Exact-match",
  "llm-judge": "LLM-as-judge",
  "code-execution": "Code execution",
  structural: "Structural",
  stateful: "Stateful",
  "step-based": "Step-based",
  "custom-domain": "Custom domain",
};
