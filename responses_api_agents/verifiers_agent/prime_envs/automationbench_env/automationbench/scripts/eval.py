# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""AutomationBench evaluation entry point."""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from automationbench.clients import (
    GeminiInteractionsClient,
    RetryingOpenAIChatCompletionsClient,
    StreamingAnthropicClient,
)
from verifiers.types import ClientConfig

from automationbench.console import (
    console,
    print_avg_reward,
    print_banner,
    print_config,
    print_results_header,
    print_task_result,
)
from automationbench.clients import OpenAIResponsesClient
from automationbench.domains import DEFAULT_DOMAINS, DOMAIN_ALIASES, get_combined_dataset
from automationbench.export import export_partial_results, export_results
from automationbench.pricing import PricingDatabase
from automationbench.rubric import create_rubric
from automationbench.runner import AutomationBenchEnv
from automationbench.usage import calculate_run_usage, print_usage_report


# Public Anthropic model ids that use adaptive thinking (`thinking: adaptive` +
# `output_config.effort`) rather than a manual `budget_tokens`. Unreleased preview models are
# NOT listed here — they come from ADAPTIVE_MODELS_EXTRA so their codenames never enter the tree.
_ADAPTIVE_MODELS_PUBLIC = (
    "opus-4-6",
    "opus-4-7",
    "opus-4-8",
    "opus-4.8",  # AI Gateway model ids use dots (anthropic/claude-opus-4.8)
    "opus-5",
    "sonnet-4-6",
    "sonnet-5",
    "fable-5",
)


def _resolve_adaptive_models() -> tuple[str, ...]:
    """Adaptive-thinking model substrings: the public list plus AUTO_BENCH_ADAPTIVE_MODELS.

    Preview/EAP codenames stay out of the source tree — they leak via the public sync and repo
    exports. Supply them at launch instead:

        env AUTO_BENCH_ADAPTIVE_MODELS=<name>[,<name>...] uv run auto-bench --model <id> ...

    Preview models are adaptive-ONLY: a manual `budget_tokens` either 400s outright or yields a
    degraded run that still scores, so a missing value must be loud rather than silent.
    """
    extra = tuple(
        part.strip().lower()
        for part in os.environ.get("AUTO_BENCH_ADAPTIVE_MODELS", "").split(",")
        if part.strip()
    )
    return _ADAPTIVE_MODELS_PUBLIC + extra


def _warn_if_unmatched_preview(model: str, adaptive: tuple[str, ...]) -> None:
    """Fail loudly when a model looks like an unreleased preview but matched no adaptive entry.

    Heuristic only: anything routed at the Anthropic API whose id is neither a known public
    family nor an AUTO_BENCH_ADAPTIVE_MODELS entry. That combination almost always means the env
    var is unset, which would 400 every rollout or silently produce budget-thinking results.
    """
    known_public = ("opus", "sonnet", "haiku", "fable", "claude-3", "claude-2")
    if any(m in model for m in adaptive):
        return
    if any(k in model.lower() for k in known_public):
        return
    raise SystemExit(
        f"Refusing to run: model {model!r} matched no adaptive-thinking entry and is not a known "
        "public family. Preview models are adaptive-only; a manual thinking budget either 400s or "
        "produces a silently degraded score. Set AUTO_BENCH_ADAPTIVE_MODELS (comma-separated id "
        "substrings) and re-run."
    )


def _task_name(obj: dict) -> str:
    """Task-name identifier. As of verifiers 0.2.0 the name lives in info['task_name']
    (the top-level `task` field is reserved for JSON task payloads)."""
    info = obj.get("info") if isinstance(obj, dict) else None
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            info = {}
    if isinstance(info, dict) and info.get("task_name"):
        return info["task_name"]
    return obj.get("task", "unknown")


def _resolve_api(model: str, base_url: str | None, api_override: str = "auto") -> str:
    """Return which API client to use: 'anthropic', 'chat_completions', 'responses',
    or 'gemini_interactions'.

    Explicit `api_override` (anything other than 'auto') wins. Otherwise auto-detects
    the first-party Anthropic/Google APIs by model prefix and base_url; everything
    else (gateway/proxy/alpha endpoints) stays on Chat Completions.
    """
    if api_override != "auto":
        return api_override
    if model.startswith("claude-") and (base_url is None or "anthropic.com" in base_url):
        return "anthropic"
    if model.startswith("gemini-") and (
        base_url is None or "generativelanguage.googleapis.com" in base_url
    ):
        return "gemini_interactions"
    return "chat_completions"


def _build_progress_callbacks():
    """Build on_start/on_progress callbacks that show running avg reward AND pass rate.

    Replaces the verifiers default progress bar so we can include both metrics
    in the tqdm postfix as tasks complete.
    """
    from tqdm import tqdm as _tqdm

    state: dict[str, Any] = {"pbar": None}

    def on_start(raw_inputs, filtered_inputs):
        total_rollouts = len(raw_inputs)
        total_groups = len({i["example_id"] for i in raw_inputs})
        rollouts_per_example = total_rollouts // total_groups if total_groups > 0 else 0

        if (
            isinstance(filtered_inputs, list)
            and filtered_inputs
            and isinstance(filtered_inputs[0], list)
        ):
            remaining = sum(len(g) for g in filtered_inputs)
        else:
            remaining = len(filtered_inputs) if filtered_inputs else 0
        saved = total_rollouts - remaining

        if not filtered_inputs:
            return

        if isinstance(filtered_inputs[0], list):
            pbar_total = total_groups
            pbar_initial = saved // rollouts_per_example if rollouts_per_example else 0
            desc = f"Processing {total_groups} groups ({total_rollouts} total rollouts)"
        else:
            pbar_total = total_rollouts
            pbar_initial = saved
            desc = f"Processing {total_rollouts} rollouts"

        state["pbar"] = _tqdm(
            total=pbar_total,
            initial=pbar_initial,
            desc=desc,
            postfix={"reward": "?", "pass_rate": "?"},
        )

    def on_progress(all_outputs, new_outputs, new_metadata):
        pbar = state["pbar"]
        if pbar is None:
            return
        pbar.update(1)
        passes = 0
        total = 0
        for o in all_outputs:
            metrics = o.get("metrics") or {}
            v = metrics.get("task_completed_correctly")
            if v is None:
                v = 1.0 if float(o.get("reward", 0.0)) == 1.0 else 0.0
            passes += 1 if float(v) == 1.0 else 0
            total += 1
        pr = passes / total if total else 0.0
        avg_r = new_metadata.get("avg_reward") or 0.0
        pbar.set_postfix({"reward": f"{avg_r:.3f}", "pass_rate": f"{pr:.3f}"})

    return on_start, on_progress


async def run_evaluation(
    model: str,
    domains: list[str],
    base_url: str | None = None,
    api_key: str | None = None,
    api_key_var: str = "OPENAI_API_KEY",
    num_examples: int = -1,
    max_turns: int = 50,
    input_cost: float | None = None,
    output_cost: float | None = None,
    export_json: str | None = None,
    save_every: int = 1,
    skip: int = 0,
    tasks: str | None = None,
    reasoning_effort: str | None = None,
    extra_headers: dict[str, str] | None = None,
    extra_body: str | None = None,
    api: str = "auto",
    toolset: str = "zapier",
    max_concurrent: int = 100,
    search_top_k: int | None = None,
    batch: bool = False,
    batch_linger_seconds: float = 20.0,
    batch_poll_seconds: float = 30.0,
) -> None:
    """Run AutomationBench evaluation."""
    # Create dataset and rubric
    dataset = get_combined_dataset(domains)

    # Skip first N tasks if requested
    if skip > 0:
        indices = list(range(skip, len(dataset)))
        dataset = dataset.select(indices)
        console.print(f"[key]Skipping first {skip} tasks, starting from task {skip + 1}[/key]")

    # Filter by specific task names if requested
    if tasks:
        task_names = [t.strip() for t in tasks.split(",")]
        indices = [i for i, row in enumerate(dataset) if _task_name(row) in task_names]
        if not indices:
            console.print(f"[bold red]No tasks found matching: {task_names}[/bold red]")
            return
        dataset = dataset.select(indices)
        console.print(f"[key]Filtering to {len(indices)} tasks: {task_names}[/key]")

    rubric = create_rubric()

    resolved_api = _resolve_api(model, base_url, api)
    use_anthropic_api = resolved_api == "anthropic"

    # Build sampling_args
    sampling_args = None
    if reasoning_effort:
        if use_anthropic_api:
            # Opus 4.6+ and Sonnet 4.6 support adaptive thinking with output_config effort.
            # Older models (Haiku 4.5, Sonnet 4.5, etc.) require manual budget_tokens.
            _adaptive_models = _resolve_adaptive_models()
            _warn_if_unmatched_preview(model, _adaptive_models)
            if any(m in model for m in _adaptive_models):
                sampling_args = {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": reasoning_effort},
                    "max_tokens": 64000,
                }
            else:
                # Map effort to thinking budget for older models
                _budget = {"low": 2000, "medium": 8000, "high": 16000, "xhigh": 24000, "max": 32000}
                budget_tokens = _budget.get(reasoning_effort, 8000)
                sampling_args = {
                    "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
                    "max_tokens": 64000,
                }
        else:
            # Pass reasoning_effort as a top-level sampling arg. On the direct
            # gemini_interactions path the client maps it to
            # generation_config.thinking_level (minimal/low/medium/high, validated
            # there). On the gateway path the gateway maps it to the
            # provider-native format (validated enum none/minimal/low/
            # medium/high/xhigh/max; for Gemini it becomes thinkingLevel — works for
            # flash variants too, unlike LiteLLM's pro-only vertex_ai allowlist).
            # Do NOT resurrect the LiteLLM-era flash workaround of sending a raw
            # thinkingConfig via extra_body: the gateway silently DROPS unknown
            # top-level fields (probed live 2026-07-14 — a bogus thinkingLevel value
            # returns 200 while a bogus reasoning_effort 400s), so that path
            # disables reasoning control entirely on flash runs.
            sampling_args = {"reasoning_effort": reasoning_effort}

    if extra_body:
        # Merge raw JSON into every request body (e.g. AI Gateway providerOptions
        # like {"providerOptions": {"gateway": {"caching": "auto"}}}).
        parsed = json.loads(extra_body)
        sampling_args = sampling_args or {}
        sampling_args["extra_body"] = {**sampling_args.get("extra_body", {}), **parsed}

    # Create environment
    env = AutomationBenchEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        toolset=toolset,
        search_top_k=search_top_k,
    )
    console.print(f"[key]Toolset:[/key] {toolset}")

    # Determine which API key var to use. The Gemini default only kicks in when
    # --api-key-var was left at its default, so an explicit var still wins.
    if use_anthropic_api:
        effective_key_var = "ANTHROPIC_API_KEY"
    elif resolved_api == "gemini_interactions" and api_key_var == "OPENAI_API_KEY":
        effective_key_var = "GEMINI_API_KEY"
    else:
        effective_key_var = api_key_var

    # If --api-key provided, set it in the env var for the client to read
    if api_key is not None:
        os.environ[effective_key_var] = api_key
    elif not os.environ.get(effective_key_var):
        raise ValueError(
            f"No API key found. Set {effective_key_var} environment variable, or pass --api-key argument."
        )

    if batch and resolved_api not in ("anthropic", "responses"):
        raise ValueError(
            "--batch requires --api anthropic or --api responses "
            "(chat-completions/proxy endpoints have no batch API)"
        )

    if resolved_api == "anthropic":
        if batch:
            from automationbench.batch import (
                AnthropicBatchBackend,
                BatchStreamingAnthropicClient,
                WaveBatcher,
            )

            anthropic_base = os.environ.get("ANTHROPIC_BASE_URL")
            if anthropic_base and "api.anthropic.com" not in anthropic_base:
                raise ValueError(
                    "--batch requires the first-party Anthropic API; unset "
                    f"ANTHROPIC_BASE_URL (currently {anthropic_base}) — gateways/proxies "
                    "do not serve /v1/messages/batches"
                )
            client = BatchStreamingAnthropicClient(AsyncAnthropic())
            backend = AnthropicBatchBackend(client.client, poll_seconds=batch_poll_seconds)
            client.attach_batcher(
                WaveBatcher(
                    backend.run_wave,
                    linger_seconds=batch_linger_seconds,
                    on_wave=lambda n, size: console.print(
                        f"[key]Batch wave {n}:[/key] {size} request(s) submitted"
                    ),
                )
            )
        else:
            client = StreamingAnthropicClient(AsyncAnthropic())
    elif resolved_api == "gemini_interactions":
        client = GeminiInteractionsClient(
            ClientConfig(
                api_key_var=effective_key_var,
                api_base_url=base_url or GeminiInteractionsClient.DEFAULT_BASE_URL,
                extra_headers=extra_headers or {},
            )
        )
    else:
        config = ClientConfig(
            api_key_var=effective_key_var,
            api_base_url=base_url or "https://api.openai.com/v1",
            extra_headers=extra_headers or {},
        )
        if resolved_api == "responses":
            if batch:
                from automationbench.batch import (
                    BatchOpenAIResponsesClient,
                    OpenAIResponsesBatchBackend,
                    WaveBatcher,
                )

                if base_url and "api.openai.com" not in base_url:
                    raise ValueError(
                        "--batch requires the first-party OpenAI API; drop --base-url "
                        f"(currently {base_url}) — gateways/proxies do not serve /v1/batches"
                    )
                client = BatchOpenAIResponsesClient(config)
                backend = OpenAIResponsesBatchBackend(
                    client.client, poll_seconds=batch_poll_seconds
                )
                client.attach_batcher(
                    WaveBatcher(
                        backend.run_wave,
                        linger_seconds=batch_linger_seconds,
                        on_wave=lambda n, size: console.print(
                            f"[key]Batch wave {n}:[/key] {size} request(s) submitted"
                        ),
                    )
                )
            else:
                client = OpenAIResponsesClient(config)
        else:
            client = RetryingOpenAIChatCompletionsClient(config)
    console.print(f"[key]API:[/key] {resolved_api}")

    if batch:
        # Every live rollout should ride the same wave — there is no client-side
        # rate pressure on the batch path, so run all tasks concurrently.
        if max_concurrent < len(dataset):
            console.print(
                f"[key]Batch mode:[/key] raising max_concurrent {max_concurrent} -> "
                f"{len(dataset)} so all tasks share each wave"
            )
            max_concurrent = len(dataset)
        console.print(
            "[key]Batch mode:[/key] 50% token pricing; reported costs use standard "
            "rates — halve for actual spend"
        )

    print_banner()
    print_config(model, domains, base_url)

    if reasoning_effort:
        console.print(f"[key]Reasoning effort:[/key] {reasoning_effort}")

    # Run evaluation
    # Note: save_results=True causes pyarrow type errors with mixed metric types
    # TODO: Fix verifiers or implement custom incremental saving
    start_time = time.time()
    on_start, on_progress = _build_progress_callbacks()
    results = await env.evaluate(
        client=client,
        model=model,
        sampling_args=sampling_args,
        num_examples=num_examples,
        rollouts_per_example=1,
        max_concurrent=max_concurrent,
        state_columns=["_usage", "_debug", "_assertion_results", "_end_state", "_perf"],
        on_start=on_start,
        on_progress=on_progress,
    )
    duration_seconds = time.time() - start_time

    # Normalize: verifiers 0.1.10 returns {"outputs": [...], "metadata": {...}}
    # Each output has: task, reward, info, prompt, completion, _usage, _debug, etc.
    raw_outputs = results["outputs"]
    metadata = results["metadata"]

    outputs: list[dict[str, Any]] = []
    for ro in raw_outputs:
        outputs.append(
            {
                "task": _task_name(ro),
                "reward": ro.get("reward", 0.0),
                "info": ro.get("info", {}),
                "prompt": ro.get("prompt"),
                "completion": ro.get("completion"),
                "_usage": ro.get("_usage"),
                "_debug": ro.get("_debug"),
                "_assertion_results": ro.get("_assertion_results"),
                "_end_state": ro.get("_end_state"),
                "_perf": ro.get("_perf"),
            }
        )

    # Calculate and print usage
    pricing_db = PricingDatabase(
        input_cost_override=input_cost,
        output_cost_override=output_cost,
    )
    usage = calculate_run_usage(results, model, pricing_db)
    print_usage_report(usage)

    # Print results
    print_results_header()
    avg_reward = (
        metadata["avg_reward"]
        if isinstance(metadata, dict)
        else getattr(metadata, "avg_reward", 0.0)
    )
    binary_scores = [
        float(ro.get("metrics", {}).get("task_completed_correctly", 0.0)) for ro in raw_outputs
    ]
    pass_rate = sum(binary_scores) / len(binary_scores) if binary_scores else None
    print_avg_reward(avg_reward, pass_rate)

    # Print detailed results per task
    for i, ro in enumerate(raw_outputs):
        print_task_result(i + 1, _task_name(ro), ro["reward"] == 1.0)

    # Export results if requested
    if export_json:
        export_results(
            outputs,
            usage,
            model,
            domains,
            export_json,
            duration_seconds=duration_seconds,
            toolset=toolset,
            reasoning_effort=reasoning_effort,
            extra_meta={"batch": True} if batch else None,
            step_cap=max_turns,
        )
        console.print(f"\n[key]Results exported to:[/key] {export_json}")


def main():
    load_dotenv()  # Load API keys from .env file

    parser = argparse.ArgumentParser(
        description="Run AutomationBench evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OpenAI
  auto-bench --model gpt-5-mini

  # Anthropic (via OpenAI-compatible endpoint)
  auto-bench --model claude-haiku-4-5-20251001 --base-url https://api.anthropic.com/v1 --api-key-var ANTHROPIC_API_KEY

  # Local model (e.g., vLLM, ollama)
  auto-bench --model llama --base-url http://localhost:8000/v1
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model name to use for evaluation (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible API endpoint",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (defaults to value from --api-key-var env var)",
    )
    parser.add_argument(
        "--api-key-var",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable to read API key from (default: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=-1,
        help="Number of examples to evaluate (-1 for all)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum number of steps (model responses) per task (default: 50)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="all",
        help="Comma-separated list of domains to run, or 'all' (default: all)",
    )
    parser.add_argument(
        "--input-cost",
        type=float,
        default=None,
        help="Per-token input cost in USD (overrides pricing lookup)",
    )
    parser.add_argument(
        "--output-cost",
        type=float,
        default=None,
        help="Per-token output cost in USD (overrides pricing lookup)",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Path to export results as JSON (default: visualizer/runs/local/{model}-{timestamp}.json)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save incremental results every N tasks (default: 1, -1 to disable)",
    )
    parser.add_argument(
        "--recover",
        type=str,
        default=None,
        help="Recover partial results from an interrupted run. Provide path to the .partial.json directory.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N tasks (0-indexed). Use --skip 70 to start from task 71.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of task names to run (e.g., 'sales.foo,sales.bar')",
    )
    parser.add_argument(
        "--extra-body",
        type=str,
        default=None,
        help='JSON object merged into every request body (e.g. \'{"providerOptions":{"gateway":{"caching":"auto"}}}\')',
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort level. OpenAI: none/minimal/low/medium/high/xhigh. Claude: low/medium/high/xhigh/max. Gemini: minimal/low/medium/high.",
    )
    parser.add_argument(
        "--headers",
        type=str,
        action="append",
        metavar="KEY=VALUE",
        help="Custom HTTP header to include in requests (can be used multiple times)",
    )
    parser.add_argument(
        "--api",
        type=str,
        default="auto",
        choices=["auto", "anthropic", "chat_completions", "responses", "gemini_interactions"],
        help=(
            "Override which API to hit. 'auto' (default) picks Anthropic Messages "
            "for claude-* models on Anthropic endpoints and Gemini Interactions "
            "for gemini-* models on Google endpoints, else Chat Completions. "
            "Use 'anthropic', 'chat_completions', 'responses', or "
            "'gemini_interactions' to force one."
        ),
    )
    parser.add_argument(
        "--toolset",
        type=str,
        default="api",
        choices=["zapier", "limited_zapier", "api"],
        help="Toolset to use: 'api' (default) for structured endpoint tools, 'zapier' for meta-tools discovery mode, 'limited_zapier' for per-task tool lists",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=100,
        help="Maximum number of concurrent tasks (default: 100)",
    )
    parser.add_argument(
        "--search-top-k",
        type=int,
        default=None,
        help="Hard cap on search_tools top_k (zapier toolset only). None = no cap (default: 20).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Transport turns through the provider Batch API (50%% token cost). "
        "Identical per-turn requests to the interactive path — conversations are "
        "split across batch requests. Requires --api anthropic or responses on the "
        "first-party endpoint; slower wall-clock (waves are sequential per task).",
    )
    parser.add_argument(
        "--batch-linger-seconds",
        type=float,
        default=20.0,
        help="Quiet period to coalesce concurrent tasks' turns into one batch wave (default: 20).",
    )
    parser.add_argument(
        "--batch-poll-seconds",
        type=float,
        default=30.0,
        help="Batch status poll interval (default: 30).",
    )
    parser.add_argument(
        "--ensure-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After the run, auto re-run any aborted tasks (transient cut-offs) and merge "
        "them in, looping until zero aborts. Guarantees the final file has no aborts. "
        "Use --no-ensure-complete to skip.",
    )
    parser.add_argument(
        "--ensure-rounds",
        type=int,
        default=3,
        help="Max re-run rounds for --ensure-complete (default: 3).",
    )

    args = parser.parse_args()

    # Handle recovery mode
    if args.recover:
        console.print(f"[key]Recovering partial results from:[/key] {args.recover}")
        output_path = args.export_json or f"recovered-{Path(args.recover).stem}.json"
        result = export_partial_results(args.recover, output_path)
        console.print(f"\n[success]✓ Recovered {result['meta']['total_tasks']} tasks[/success]")
        console.print(f"  Passed: {result['summary']['passed_count']}")
        console.print(f"  Failed: {result['summary']['failed_count']}")
        console.print(f"  Avg score: {result['summary']['avg_score']:.0%}")
        console.print(f"\n[key]Results exported to:[/key] {output_path}")
        return

    # Parse domains (expand aliases like "private" and "public")
    if args.domains == "all":
        domains = DEFAULT_DOMAINS
    else:
        domains = []
        for d in args.domains.split(","):
            d = d.strip()
            if d in DOMAIN_ALIASES:
                domains.extend(DOMAIN_ALIASES[d])
            else:
                domains.append(d)

    # Parse custom headers
    extra_headers = None
    if args.headers:
        extra_headers = {}
        for header in args.headers:
            if "=" not in header:
                parser.error(f"Invalid header format: {header}. Use KEY=VALUE format.")
            key, value = header.split("=", 1)
            extra_headers[key] = value

    # Auto-generate export path if not provided
    export_json = args.export_json
    if export_json is None:
        # Millisecond precision + toolset suffix prevents collisions between
        # concurrent runs of the same model/effort on different toolsets.
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        model_safe = args.model.replace("/", "-")
        effort_suffix = f"-{args.reasoning_effort}" if args.reasoning_effort else ""
        toolset_suffix = f"-{args.toolset}" if args.toolset and args.toolset != "api" else ""
        export_json = (
            f"visualizer/runs/local/{model_safe}{effort_suffix}{toolset_suffix}-{timestamp}.json"
        )

    asyncio.run(
        run_evaluation(
            model=args.model,
            domains=domains,
            base_url=args.base_url,
            api_key=args.api_key,
            api_key_var=args.api_key_var,
            num_examples=args.num_examples,
            max_turns=args.max_steps,
            input_cost=args.input_cost,
            output_cost=args.output_cost,
            export_json=export_json,
            save_every=args.save_every,
            skip=args.skip,
            tasks=args.tasks,
            reasoning_effort=args.reasoning_effort,
            extra_headers=extra_headers,
            extra_body=args.extra_body,
            api=args.api,
            toolset=args.toolset,
            max_concurrent=args.max_concurrent,
            search_top_k=args.search_top_k,
            batch=args.batch,
            batch_linger_seconds=args.batch_linger_seconds,
            batch_poll_seconds=args.batch_poll_seconds,
        )
    )

    # Completeness gate: re-run any tasks that were cut off mid-work (aborts) and
    # merge them in, so the final file is guaranteed abort-free. Runs after the
    # main loop (separate event loop), healing the exported file in place. A
    # full-dataset run with --skip/--tasks slicing is left alone (partial by design).
    if args.ensure_complete and export_json and not args.tasks and args.skip == 0:
        from automationbench.scripts.complete_run import heal_run_file

        heal_run_file(
            export_json,
            rename_original=False,
            base_url=args.base_url,
            api_key_var=args.api_key_var,
            api=args.api,
            max_concurrent=args.max_concurrent,
            max_rounds=args.ensure_rounds,
            step_cap=args.max_steps,
            # Healed re-runs must keep the run's transport extras — without these,
            # gateway runs lose their FinOps tags and providerTimeouts on exactly
            # the retried tasks.
            extra_headers=extra_headers,
            extra_body=args.extra_body,
            batch=args.batch,
            batch_linger_seconds=args.batch_linger_seconds,
            batch_poll_seconds=args.batch_poll_seconds,
        )


if __name__ == "__main__":
    main()
