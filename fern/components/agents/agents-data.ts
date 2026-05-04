/**
 * Agent harness data — source of truth for the Agent Catalog.
 *
 * Each entry describes a shipped agent harness under `responses_api_agents/`.
 * Adding a harness here is part of shipping a new agent server; the data
 * drives the catalog page at /get-started/choose-an-agent.
 */

export type AgentShape =
  /** Linear loop: model → tool calls → repeat until final answer or max_steps. */
  | "model-tool-loop"
  /** Multi-turn correction: model emits → verifier returns → model retries. */
  | "correction-loop"
  /** Wraps an external library/runtime (Verifiers, Harbor, Aviary, LangGraph, gymnasium). */
  | "external-wrapper"
  /** Domain-specific harness (SWE-bench, CVDP, browse, etc.). */
  | "domain-specific";

export interface Agent {
  id: string;
  /** Display name (the directory name). */
  name: string;
  shape: AgentShape;
  /** Whether this is a recommended default for new users. */
  recommended_default: boolean;
  /** True when rollouts are trainable as a multi-turn trajectory (preserves tokens + context). */
  trainable_multi_turn: boolean;
  /** Whether this harness has its own resources server (false = uses external scoring). */
  uses_gym_verify: boolean;
  /** Source path. */
  source_path: string;
  /** One-line description of what this agent orchestrates. */
  description: string;
  /** When to copy or wire this agent. */
  use_when: string;
  /** Tags surfaced in the UI. */
  tags: string[];
}

const REPO_BASE = "https://github.com/NVIDIA-NeMo/Gym/tree/main";

export function agentSourceUrl(agent: Agent): string {
  return `${REPO_BASE}/${agent.source_path}`;
}

export const AGENTS: Agent[] = [
  {
    id: "simple_agent",
    name: "simple_agent",
    shape: "model-tool-loop",
    recommended_default: true,
    trainable_multi_turn: true,
    uses_gym_verify: true,
    source_path: "responses_api_agents/simple_agent",
    description: "The default. Model + tool-call loop until the model emits a final assistant message or hits max_steps.",
    use_when: "Most evals — single-turn QA, code generation, math, multi-step tool calls. Start here unless you have a specific reason.",
    tags: ["default", "single-turn", "multi-step", "tool-use"],
  },
  {
    id: "non_executing_simple_agent",
    name: "non_executing_simple_agent",
    shape: "model-tool-loop",
    recommended_default: false,
    trainable_multi_turn: false,
    uses_gym_verify: true,
    source_path: "responses_api_agents/non_executing_simple_agent",
    description: "Forwards a single request and passes the model response straight to the verifier. Does not execute tool calls.",
    use_when: "The tool call itself is the final answer — function-calling format checks, schema validation.",
    tags: ["single-turn", "no-tool-execution"],
  },
  {
    id: "proof_refinement_agent",
    name: "proof_refinement_agent",
    shape: "correction-loop",
    recommended_default: false,
    trainable_multi_turn: false,
    uses_gym_verify: true,
    source_path: "responses_api_agents/proof_refinement_agent",
    description: "Multi-turn self-correction loop: model gets verifier error feedback and retries up to max_correction_turns.",
    use_when: "Iterative refinement helps — formal proofs, code with a compiler in the loop. Eval-only: resets context per turn, not directly trainable as a multi-turn trajectory.",
    tags: ["multi-turn", "correction", "eval-only"],
  },
  {
    id: "gymnasium_agent",
    name: "gymnasium_agent",
    shape: "external-wrapper",
    recommended_default: false,
    trainable_multi_turn: true,
    uses_gym_verify: true,
    source_path: "responses_api_agents/gymnasium_agent",
    description: "Drives a Gymnasium-style reset()/step() loop against a GymnasiumServer resources server.",
    use_when: "Your environment is a Gymnasium-style state machine (Blackjack, classic control, custom episodic envs).",
    tags: ["gymnasium", "stateful", "step-based"],
  },
  {
    id: "tool_simulation_agent",
    name: "tool_simulation_agent",
    shape: "model-tool-loop",
    recommended_default: false,
    trainable_multi_turn: true,
    uses_gym_verify: true,
    source_path: "responses_api_agents/tool_simulation_agent",
    description: "Required tool execution results are produced by simulation rather than real execution.",
    use_when: "You want deterministic tool outputs for reproducible evals or for synthetic-data generation.",
    tags: ["simulated-tools", "reproducibility"],
  },
  {
    id: "verifiers_agent",
    name: "verifiers_agent",
    shape: "external-wrapper",
    recommended_default: false,
    trainable_multi_turn: true,
    uses_gym_verify: false,
    source_path: "responses_api_agents/verifiers_agent",
    description: "Bridges to Prime Intellect's Verifiers library, including environments from the Environments Hub.",
    use_when: "You want to run an existing Verifiers environment (e.g. acereason-math, ascii-tree) inside Gym.",
    tags: ["external-wrapper", "verifiers"],
  },
  {
    id: "langgraph_agent",
    name: "langgraph_agent",
    shape: "external-wrapper",
    recommended_default: false,
    trainable_multi_turn: false,
    uses_gym_verify: true,
    source_path: "responses_api_agents/langgraph_agent",
    description: "Adapter for LangGraph graphs: iterative reflection, subagent orchestration, parallel thinking, ReWOO.",
    use_when: "You want a graph-shaped harness from the LangGraph ecosystem. Note: non-monotonic trajectories are not directly trainable with NeMo RL.",
    tags: ["external-wrapper", "langgraph", "graph", "eval-only"],
  },
  {
    id: "harbor_agent",
    name: "harbor_agent",
    shape: "external-wrapper",
    recommended_default: false,
    trainable_multi_turn: true,
    uses_gym_verify: false,
    source_path: "responses_api_agents/harbor_agent",
    description: "Runs Harbor agents (e.g. terminus-2) in Harbor-managed environments and returns Gym-compatible outputs.",
    use_when: "You want to evaluate a Harbor agent inside Gym.",
    tags: ["external-wrapper", "harbor", "swe"],
  },
  {
    id: "aviary_agent",
    name: "aviary_agent",
    shape: "external-wrapper",
    recommended_default: false,
    trainable_multi_turn: true,
    uses_gym_verify: true,
    source_path: "responses_api_agents/aviary_agent",
    description: "Bridges to FutureHouse's Aviary framework. Pair with resources_servers/aviary.",
    use_when: "You want to run Aviary environments and datasets inside Gym.",
    tags: ["external-wrapper", "aviary"],
  },
  {
    id: "stirrup_agent",
    name: "stirrup_agent",
    shape: "external-wrapper",
    recommended_default: false,
    trainable_multi_turn: true,
    uses_gym_verify: true,
    source_path: "responses_api_agents/stirrup_agent",
    description: "Uses the Stirrup agent loop to evaluate models on GDPVal.",
    use_when: "Real-world professional knowledge-work tasks (finance, law, healthcare, engineering).",
    tags: ["external-wrapper", "stirrup", "gdpval"],
  },
  {
    id: "mini_swe_agent",
    name: "mini_swe_agent",
    shape: "domain-specific",
    recommended_default: false,
    trainable_multi_turn: true,
    uses_gym_verify: true,
    source_path: "responses_api_agents/mini_swe_agent",
    description: "Integrates the Mini-SWE-Agent harness for SWE-Bench.",
    use_when: "Software-engineering tasks at SWE-Bench scope.",
    tags: ["swe", "external-wrapper"],
  },
  {
    id: "swe_agents",
    name: "swe_agents",
    shape: "domain-specific",
    recommended_default: false,
    trainable_multi_turn: true,
    uses_gym_verify: true,
    source_path: "responses_api_agents/swe_agents",
    description: "Coding-agent harnesses (SWE-Bench style) with sandbox/repo interaction.",
    use_when: "Larger-footprint SWE tasks beyond mini_swe_agent.",
    tags: ["swe", "sandboxed"],
  },
  {
    id: "browsecomp_agent",
    name: "browsecomp_agent",
    shape: "domain-specific",
    recommended_default: false,
    trainable_multi_turn: true,
    uses_gym_verify: true,
    source_path: "responses_api_agents/browsecomp_agent",
    description: "Browse-and-answer harness for the BrowseComp benchmark.",
    use_when: "Web-search / browsing benchmarks.",
    tags: ["browse", "search"],
  },
  {
    id: "tau2",
    name: "tau2",
    shape: "domain-specific",
    recommended_default: false,
    trainable_multi_turn: true,
    uses_gym_verify: true,
    source_path: "responses_api_agents/tau2",
    description: "Harness for the τ²-bench customer-service evaluation.",
    use_when: "Multi-turn customer-service / tool-use evaluation.",
    tags: ["customer-service", "multi-turn"],
  },
  {
    id: "cvdp_agent",
    name: "cvdp_agent",
    shape: "domain-specific",
    recommended_default: false,
    trainable_multi_turn: true,
    uses_gym_verify: true,
    source_path: "responses_api_agents/cvdp_agent",
    description: "Hardware-design harness. Pair with resources_servers/cvdp.",
    use_when: "CVDP (Comprehensive Verilog Design Problems) benchmark.",
    tags: ["hardware", "verilog"],
  },
  {
    id: "labbench2_vlm_agent",
    name: "labbench2_vlm_agent",
    shape: "domain-specific",
    recommended_default: false,
    trainable_multi_turn: true,
    uses_gym_verify: true,
    source_path: "responses_api_agents/labbench2_vlm_agent",
    description: "Extends simple_agent, resolves verifier_metadata.media_dir and injects input_image blocks.",
    use_when: "Vision-language tasks where the model needs image/PDF inputs from disk (LAB-Bench v2 VLM).",
    tags: ["vlm", "vision", "multi-modal"],
  },
];

export const SHAPE_LABELS: Record<AgentShape, string> = {
  "model-tool-loop": "Model + tool-call loop",
  "correction-loop": "Correction loop",
  "external-wrapper": "External wrapper",
  "domain-specific": "Domain-specific",
};
