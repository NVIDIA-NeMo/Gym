import argparse
import json
import sys

import dspy
import requests


class GymAgent(dspy.Module):
    def __init__(self, agent_url, seed, timeout):
        super().__init__()
        self.agent_url = agent_url
        self.timeout = timeout
        self.predict = dspy.Predict("task -> answer")
        self.predict.signature = self.predict.signature.with_instructions(seed)

    def forward(self, row):
        rcp = row.get("responses_create_params") or {}
        base_input = rcp.get("input") or [
            {"role": "user", "content": row.get("problem") or row.get("question") or ""}
        ]
        task = base_input[-1]["content"]
        body = {
            **row,
            "responses_create_params": {
                **rcp,
                "input": [{"role": "system", "content": self.predict.signature.instructions}] + base_input,
            },
        }
        try:
            r = requests.post(f"{self.agent_url}/run", json=body, timeout=self.timeout).json()
            reward, answer = float(r.get("reward", 0.0)), str(r.get("response", {}))[:2000]
            err = r.get("error_message")
        except requests.exceptions.ConnectionError:
            raise SystemExit(f"\nCannot reach agent at {self.agent_url}. Use the agent URL from your ng_run logs.")
        except Exception as e:
            reward, answer, err = 0.0, "", f"agent call failed: {type(e).__name__}"
        if dspy.settings.trace is not None:
            dspy.settings.trace.append((self.predict, {"task": task[:2000]}, dspy.Prediction(answer=answer)))
        return dspy.Prediction(success=reward, trajectory=answer, feedback=err or f"reward={reward}")


def metric_simple(example, pred, trace=None):
    return pred.success


def metric_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    fb = (
        "SUCCESS."
        if pred.success
        else f"FAILURE. Trajectory:\n{pred.trajectory}\nDiagnose and propose a clearer strategy."
    )
    return dspy.Prediction(score=pred.success, feedback=fb)


def load_dataset(path):
    rows = [json.loads(line) for line in open(path)]
    return [dspy.Example(row=r).with_inputs("row") for r in rows]


def run(args):
    lm = dspy.LM(
        f"openai/{args.reflection_model}",
        api_key=args.reflection_api_key,
        api_base=args.reflection_base_url,
        temperature=1.0,
        max_tokens=4000,
        num_retries=10,
    )
    dspy.configure(lm=lm)

    try:
        requests.get(f"{args.agent_url}/", timeout=10).raise_for_status()
    except Exception:
        raise SystemExit(f"Agent not reachable at {args.agent_url}. Copy the agent URL from your ng_run logs.")

    examples = load_dataset(args.data)
    train = examples[: args.train_n]
    val = examples[args.train_n : args.train_n + args.val_n]
    print(f"optimizer={args.optimizer} train={len(train)} val={len(val)}", flush=True)

    def build():
        return GymAgent(args.agent_url, args.seed, args.timeout)

    evaluate = dspy.Evaluate(devset=val, metric=metric_simple, num_threads=4, display_progress=True)
    baseline = evaluate(build()).score / 100.0

    if args.optimizer == "gepa":
        teleprompter = dspy.GEPA(
            metric=metric_feedback, max_metric_calls=args.max_calls, reflection_lm=lm, track_stats=True
        )
        optimized = teleprompter.compile(build(), trainset=train, valset=val)
    else:
        from dspy.teleprompt import MIPROv2

        teleprompter = MIPROv2(metric=metric_simple, auto="light", prompt_model=lm)
        optimized = teleprompter.compile(
            build(),
            trainset=train,
            valset=val,
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
            requires_permission_to_run=False,
        )

    after = evaluate(optimized).score / 100.0
    best_prompt = optimized.predict.signature.instructions
    print(f"\n{args.optimizer}: baseline {baseline} -> optimized {after}", flush=True)
    print("\n===== BEST PROMPT =====\n" + best_prompt)

    if args.out:
        result = {
            "optimizer": args.optimizer,
            "seed_prompt": args.seed,
            "baseline_val_acc": baseline,
            "optimized_val_acc": after,
            "best_prompt": best_prompt,
            "train_n": len(train),
            "val_n": len(val),
            "max_calls": args.max_calls,
            "curve": curve_from(optimized),  # [{candidate, eval_calls, val_acc, topline}] for plotting
        }
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote results + curve to {args.out}", flush=True)
    return optimized


def curve_from(optimized):
    """Per-candidate val-accuracy curve from DSPy GEPA's track_stats results, where `topline` is the running-max accuracy to plot against candidate index (empty for MIPRO)."""
    dr = getattr(optimized, "detailed_results", None)
    scores = list(getattr(dr, "val_aggregate_scores", []) or [])
    if not scores:
        return []
    calls = list(getattr(dr, "discovery_eval_counts", []) or [])
    curve, best = [], float("-inf")
    for i, s in enumerate(scores):
        best = max(best, s)
        curve.append({"candidate": i, "eval_calls": calls[i] if i < len(calls) else None,
                      "val_acc": s, "topline": best})
    return curve


def main(default_optimizer=None):
    if default_optimizer and not any(a.startswith("--optimizer") for a in sys.argv):
        sys.argv += ["--optimizer", default_optimizer]
    p = argparse.ArgumentParser(description="DSPy-optimize a NeMo Gym agent's system prompt.")
    p.add_argument("--optimizer", choices=["gepa", "mipro"], default="gepa")
    p.add_argument("--agent-url", required=True, help="Gym agent /run endpoint, e.g. http://127.0.0.1:18123")
    p.add_argument("--data", required=True, help="Gym-format JSONL (responses_create_params.input + verifier fields)")
    p.add_argument("--seed", default="Solve the problem.", help="Initial system prompt to optimize")
    p.add_argument("--reflection-model", required=True, help="Reflection/prompt LLM id")
    p.add_argument("--reflection-base-url", required=True, help="OpenAI-compatible base URL for the LLM")
    p.add_argument("--reflection-api-key", required=True)
    p.add_argument("--train-n", type=int, default=12)
    p.add_argument("--val-n", type=int, default=20)
    p.add_argument("--max-calls", type=int, default=120, help="GEPA rollout budget")
    p.add_argument("--timeout", type=int, default=600, help="Per-rollout timeout (s)")
    p.add_argument("--out", default="gepa_results.json", help="Write results + iteration curve here (JSON)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
