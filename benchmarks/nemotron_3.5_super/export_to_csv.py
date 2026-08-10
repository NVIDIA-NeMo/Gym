import json
from argparse import ArgumentParser
from csv import DictWriter
from datetime import datetime
from zoneinfo import ZoneInfo


parser = ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--jsonl-fpath-base", type=str, required=True)
args = parser.parse_args()

aggregate_metrics_fpath = f"{args.jsonl_fpath_base}_aggregate_metrics.json"
csv_fpath = f"{args.jsonl_fpath_base}_export.csv"

with open(aggregate_metrics_fpath) as f:
    aggregate_metrics = json.load(f)

agent_to_metrics = {d["agent_ref"]["name"]: d for d in aggregate_metrics}


def v(agent: str, value: str) -> float:
    return agent_to_metrics[agent]["agent_metrics"][value]


row = {
    "Model nickname": "",
    "Note": "",
    "Model path": args.model_path,
    "Date run": datetime.now(ZoneInfo("America/Los_Angeles")),
    "Gym commit": None,  # TODO
    "Tau3-Banking": v("tau2_banking_knowledge_bm25_grep_artificial_analysis_agent", "mean/reward"),
    "Tau3-Average": sum(
        (
            v("tau2_benchmark_agent", "airline/reward"),
            v("tau2_benchmark_agent", "retail/reward"),
            v("tau2_benchmark_agent", "telecom/reward"),
            v("tau2_banking_knowledge_bm25_grep_artificial_analysis_agent", "mean/reward"),
        )
    )
    / 4,
    "SciCode": v("scicode_benchmark_agent", "mean/reward"),
    "AA-LCR": v("aalcr_benchmark_simple_agent", "mean/reward"),
    "AA-Omniscience (OmniIndex)": v("omniscience_omniscience_simple_agent", "mean/reward"),
    "GPQA Diamond": v("gpqa_mcqa_simple_agent", "mean/reward"),
}

with open(csv_fpath, "w") as f:
    writer = DictWriter(f, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
