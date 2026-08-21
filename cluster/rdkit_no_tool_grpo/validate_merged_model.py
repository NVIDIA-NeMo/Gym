#!/usr/bin/env python3
"""Compare fixed-prompt logits from PEFT and the atomically merged HF model."""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import UTC, datetime
from pathlib import Path

PROMPTS = (
    "For the molecule CCO, what is the exact RDKit atom count?",
    "For c1ccccc1, report the RDKit ring count as an integer.",
    "Does CC(=O)O contain a carboxylic acid functional group?",
    "What is the molecular formula of CCN?",
    "For CCOC(=O)C, how many oxygen atoms are present?",
    "Is the SMILES C1CCCCC1 aromatic according to RDKit?",
    "For CC(C)C, report the number of rotatable bonds.",
    "What is the exact formal charge of C[N+](C)(C)C?",
    "For O=C=O, report the heavy atom count.",
    "Does c1ccncc1 contain a nitrogen atom?",
    "For CC#N, report the number of triple bonds.",
    "What is the RDKit hydrogen-bond donor count for NCCO?",
    "For CC(Cl)Br, how many halogen atoms are present?",
    "Does COC contain an ether linkage?",
    "For C1=CC=CN=C1, report the ring count.",
    "What is the total atom count for [Na+].[Cl-]?",
)


def load_model(model_path: str, adapter_dir: Path | None):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    return model


def fixed_logits(model, tokenizer):
    import torch

    outputs = []
    for prompt in PROMPTS:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=True,
        ).to("cuda:0")
        with torch.inference_mode():
            outputs.append(model(input_ids=rendered).logits[0, -1].float().cpu())
    return torch.stack(outputs)


def release_cuda() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--merged-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    args = parser.parse_args()

    from torch.nn import functional as F
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    print("Loading base + ES adapter for fixed-prompt logits", flush=True)
    peft_model = load_model(args.base_model, args.adapter_dir.resolve())
    peft_logits = fixed_logits(peft_model, tokenizer)
    del peft_model
    release_cuda()

    print("Loading merged HF model for fixed-prompt logits", flush=True)
    merged_model = load_model(str(args.merged_model_dir.resolve()), None)
    merged_logits = fixed_logits(merged_model, tokenizer)
    del merged_model
    release_cuda()

    cosines = F.cosine_similarity(peft_logits, merged_logits, dim=1)
    top1_equal = peft_logits.argmax(dim=1).eq(merged_logits.argmax(dim=1))
    passed = bool(top1_equal.all() and cosines.min().item() >= args.min_cosine)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "prompt_count": len(PROMPTS),
        "minimum_cosine_similarity": cosines.min().item(),
        "mean_cosine_similarity": cosines.mean().item(),
        "top1_match_count": int(top1_equal.sum().item()),
        "minimum_required_cosine_similarity": args.min_cosine,
        "passed": passed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f".{os.environ.get('SLURM_JOB_ID', 'local')}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise RuntimeError("Merged HF model failed fixed-prompt parity")


if __name__ == "__main__":
    main()
