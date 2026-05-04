/**
 * Compatibility matrix data — single source of truth driving the three
 * /reference/compatibility/* pages.
 *
 * Centralizing this here means a fact written once can't drift across
 * provider, RL-framework, and sandbox tables. Adding a row is part of
 * shipping a new server / framework / sandbox; the next Content Audit
 * verifies the truth values against source.
 */

export type SupportStatus =
  /** Tested and shipped. */
  | "supported"
  /** Partial support; pinned version, known limitations, or community-tested. */
  | "partial"
  /** Documented as not supported (with rationale). */
  | "unsupported"
  /** Roadmap item; not yet shipped. */
  | "planned";

export interface MatrixCell {
  status: SupportStatus;
  /** Optional pin or version string surfaced on hover. */
  detail?: string;
  /** Optional source citation (file:line). */
  source?: string;
}

export interface MatrixRow {
  /** Row identifier. */
  id: string;
  /** Display label for the row (e.g. "openai_model", "NeMo RL"). */
  label: string;
  /** Optional link target — usually the per-row deep-dive doc. */
  href?: string;
  /** One cell per column key. */
  cells: Record<string, MatrixCell>;
}

export interface MatrixDefinition {
  /** Title of this matrix (header for the rendered table). */
  title: string;
  /** One-line description shown above the matrix. */
  caption: string;
  /** Column definitions, in display order. */
  columns: { key: string; label: string }[];
  rows: MatrixRow[];
}

// -----------------------------------------------------------------------------
// Provider compatibility — model server × feature.
// Source: responses_api_models/<name>/app.py + reference/compatibility/providers.mdx
// -----------------------------------------------------------------------------

export const PROVIDER_MATRIX: MatrixDefinition = {
  title: "Model server × feature",
  caption:
    "Each row is a shipped model server type. Each column is a feature it must implement to be useful for that workload.",
  columns: [
    { key: "responses", label: "/v1/responses" },
    { key: "chat", label: "/v1/chat/completions" },
    { key: "tools", label: "Tool calling" },
    { key: "reasoning", label: "Reasoning blocks" },
    { key: "concurrency_cap", label: "Concurrency cap" },
  ],
  rows: [
    {
      id: "openai_model",
      label: "openai_model",
      href: "/latest/reference/compatibility/providers#openai-apiopenaicom",
      cells: {
        responses: { status: "supported", detail: "Native — calls /v1/responses" },
        chat: { status: "partial", detail: "Available but not the canonical path" },
        tools: { status: "supported" },
        reasoning: { status: "supported", detail: "Reasoning items in Responses output" },
        concurrency_cap: { status: "unsupported", detail: "Unbounded by default" },
      },
    },
    {
      id: "azure_openai_model",
      label: "azure_openai_model",
      href: "/latest/reference/compatibility/providers#azure-openai",
      cells: {
        responses: {
          status: "partial",
          detail:
            "Synthesized — calls chat.completions.create() internally and reshapes via VLLMConverter",
          source: "responses_api_models/azure_openai_model/app.py:64",
        },
        chat: { status: "supported", detail: "Wire-level path" },
        tools: { status: "supported" },
        reasoning: { status: "supported" },
        concurrency_cap: {
          status: "supported",
          detail: "num_concurrent_requests semaphore (default 8)",
        },
      },
    },
    {
      id: "vllm_model",
      label: "vllm_model",
      href: "/latest/reference/compatibility/providers#self-hosted-vllm-remote",
      cells: {
        responses: {
          status: "partial",
          detail: "Only when is_responses_native: true is set on the config",
        },
        chat: { status: "supported" },
        tools: { status: "partial", detail: "Requires --enable-auto-tool-choice and tool-call-parser" },
        reasoning: { status: "partial", detail: "Requires --reasoning-parser on the vLLM CLI" },
        concurrency_cap: { status: "unsupported", detail: "Unbounded by default" },
      },
    },
    {
      id: "local_vllm_model",
      label: "local_vllm_model",
      href: "/latest/reference/compatibility/providers#local_vllm_model",
      cells: {
        responses: {
          status: "partial",
          detail: "Same as vllm_model — gated on is_responses_native",
        },
        chat: { status: "supported" },
        tools: { status: "supported", detail: "vllm_serve_kwargs.tool_call_parser" },
        reasoning: { status: "supported", detail: "vllm_serve_kwargs.reasoning_parser" },
        concurrency_cap: { status: "unsupported" },
      },
    },
    {
      id: "local_vllm_model_proxy",
      label: "local_vllm_model_proxy",
      cells: {
        responses: { status: "partial", detail: "Same as local_vllm_model" },
        chat: { status: "supported" },
        tools: { status: "supported" },
        reasoning: { status: "supported" },
        concurrency_cap: { status: "unsupported" },
      },
    },
    {
      id: "genrm_model",
      label: "genrm_model",
      cells: {
        responses: { status: "unsupported", detail: "Reward-model role API differs from Responses" },
        chat: {
          status: "supported",
          detail: "Custom roles: response_1, response_2, principle",
        },
        tools: { status: "unsupported", detail: "Reward-model use case; tools not applicable" },
        reasoning: { status: "partial", detail: "uses_reasoning_parser passes through" },
        concurrency_cap: { status: "unsupported" },
      },
    },
  ],
};

// -----------------------------------------------------------------------------
// RL framework compatibility — framework × integration shape.
// Source: reference/compatibility/rl-frameworks.mdx + responses_api_models/local_vllm*
// -----------------------------------------------------------------------------

export const RL_FRAMEWORK_MATRIX: MatrixDefinition = {
  title: "RL framework × integration",
  caption:
    "Status of NeMo Gym's environment / rollout / training-data integration with each supported RL trainer.",
  columns: [
    { key: "rollouts", label: "Rollout collection" },
    { key: "online_grpo", label: "Online GRPO" },
    { key: "offline", label: "Offline (SFT/DPO)" },
    { key: "multi_env", label: "Multi-env training" },
  ],
  rows: [
    {
      id: "nemo_rl",
      label: "NeMo RL",
      href: "/latest/train-models/nemo-rl-grpo",
      cells: {
        rollouts: { status: "supported", detail: "Native integration" },
        online_grpo: { status: "supported", detail: "Workplace Assistant recipe shipped" },
        offline: { status: "supported" },
        multi_env: { status: "supported" },
      },
    },
    {
      id: "unsloth",
      label: "Unsloth",
      href: "/latest/train-models/unsloth",
      cells: {
        rollouts: { status: "supported" },
        online_grpo: { status: "supported", detail: "Single-GPU friendly" },
        offline: { status: "partial" },
        multi_env: { status: "unsupported" },
      },
    },
    {
      id: "openrlhf",
      label: "OpenRLHF",
      href: "https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/python/agent_func_nemogym_executor.py",
      cells: {
        rollouts: { status: "supported", detail: "Agent executor integration shipped" },
        online_grpo: { status: "partial", detail: "Community-maintained" },
        offline: { status: "partial" },
        multi_env: { status: "unsupported" },
      },
    },
    {
      id: "nemo_customizer",
      label: "NeMo Customizer",
      cells: {
        rollouts: { status: "planned" },
        online_grpo: { status: "planned" },
        offline: { status: "planned" },
        multi_env: { status: "planned" },
      },
    },
    {
      id: "verl",
      label: "VeRL",
      href: "/latest/reference/rl-framework-integration",
      cells: {
        rollouts: { status: "planned", detail: "Roadmap; integration contract documented" },
        online_grpo: { status: "planned" },
        offline: { status: "planned" },
        multi_env: { status: "planned" },
      },
    },
  ],
};

// -----------------------------------------------------------------------------
// Sandbox / execution backend matrix — sandbox × runtime property.
// Source: reference/compatibility/sandbox-execution.mdx
// -----------------------------------------------------------------------------

export const SANDBOX_MATRIX: MatrixDefinition = {
  title: "Sandbox × runtime property",
  caption: "Where rollouts and tool calls actually execute, and what each backend guarantees.",
  columns: [
    { key: "isolation", label: "Process isolation" },
    { key: "network", label: "Network sandboxing" },
    { key: "deps", label: "Custom deps" },
    { key: "timeout", label: "Timeout enforcement" },
  ],
  rows: [
    {
      id: "local_subprocess",
      label: "Local subprocess",
      cells: {
        isolation: { status: "supported", detail: "asyncio.Semaphore + subprocess.run" },
        network: { status: "unsupported", detail: "Inherits host network" },
        deps: { status: "partial", detail: "User-managed venv per server" },
        timeout: { status: "supported", detail: "subprocess.run(timeout=N)" },
      },
    },
    {
      id: "ray_actor",
      label: "Ray actor",
      cells: {
        isolation: { status: "supported", detail: "@ray.remote actor" },
        network: { status: "unsupported", detail: "Ray cluster network unless restricted" },
        deps: { status: "supported", detail: "Per-actor runtime env" },
        timeout: { status: "supported", detail: "ray.wait / ray.cancel" },
      },
    },
    {
      id: "docker",
      label: "Docker / Podman",
      cells: {
        isolation: { status: "supported", detail: "Container isolation" },
        network: { status: "supported", detail: "--network=none or custom bridge" },
        deps: { status: "supported", detail: "Image owns the env" },
        timeout: { status: "partial", detail: "External process killer" },
      },
    },
    {
      id: "singularity",
      label: "Singularity",
      cells: {
        isolation: { status: "supported", detail: "HPC-friendly container isolation" },
        network: { status: "supported" },
        deps: { status: "supported", detail: "SIF image" },
        timeout: { status: "partial" },
      },
    },
    {
      id: "harbor_sandbox",
      label: "Harbor sandbox",
      href: "/latest/build-environments/integrations/harbor",
      cells: {
        isolation: { status: "supported", detail: "Harbor environment owns sandboxing" },
        network: { status: "supported" },
        deps: { status: "supported" },
        timeout: { status: "supported" },
      },
    },
    {
      id: "unified_sandbox",
      label: "Unified execution sandbox",
      cells: {
        isolation: { status: "planned", detail: "Issue #1048" },
        network: { status: "planned" },
        deps: { status: "planned" },
        timeout: { status: "planned" },
      },
    },
  ],
};
