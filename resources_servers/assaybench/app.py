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
"""AssayBench resources server: phenotypic CRISPR screen prediction as gene ranking.

AssayBench (arXiv:2605.10876) describes a CRISPR screen in plain text and asks for the 100 genes
most likely to be hits, ranked. The verifier scores that list against the screen's relevance
vector with the paper's metrics -- adjusted nDCG@k (AnDCG, the headline number), Precision@k and
directional FDR@k -- as implemented by the ``assaybench`` PyPI package the authors released. The
metric code and the HGNC gene mapper are imported from that package, not reimplemented, so a
score here is the same computation as a score in the paper.

The reference harness ran every model through DSPy ``ChainOfThought`` and read the gene list out of
the ``[[ ## answer ## ]]`` section of the reply; ``verify()`` reads the reply the same way (see
``gene_parsing.py``). ``reward`` is AnDCG@100 in [0, 1].
"""

import asyncio
import math
import re
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)
from resources_servers.assaybench.gene_parsing import (
    extract_genes_from_raw_response,
    parse_dspy_completion,
    parse_genes_from_output,
)


# How the gene list was recovered from the reply (`extraction_mode` on the response).
EXTRACTION_DSPY_ANSWER = "dspy_answer"  # the `[[ ## answer ## ]]` field, as the reference harness read it
EXTRACTION_QUALITY_GATE = "quality_gate_fallback"  # answer field too short; raw-reply scan won
EXTRACTION_RAW_FALLBACK = "raw_fallback"  # DSPy could not parse the reply; raw-reply scan
EXTRACTION_NONE = "none"  # nothing that looks like a gene list

# The metrics the paper reports (figures/journal_figures_common.py::METRICS in the upstream repo),
# at the cutoff it reports them. `RankingMetrics` only emits precision/fdr@k when the model
# produced at least k distinct genes; a rollout without the key drops out of that metric's mean,
# exactly as in the reference aggregation.
PAPER_METRICS = (
    "adjusted_ndcg@100",
    "precision@100",
    "fdr@100",
    "normalized_precision@100",
    "normalized_fdr@100",
)
DIAGNOSTIC_METRICS = ("hallucination_rate",)

THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def slugify_phenotype(value: Optional[str]) -> Optional[str]:
    """`"Fitness / Proliferation / Viability"` -> `"fitness_proliferation_viability"` for metric keys."""
    if not value:
        return None
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or None


def is_truncated(response: Any) -> bool:
    """True when the model hit its output budget before finishing (Responses API `incomplete`)."""
    if getattr(response, "status", None) == "incomplete":
        return True
    details = getattr(response, "incomplete_details", None)
    reason = details.get("reason") if isinstance(details, dict) else getattr(details, "reason", None)
    return reason == "max_output_tokens"


def extract_predicted_genes(text: str, min_expected_genes: int = 0) -> tuple[List[str], str, bool]:
    """Recover the ranked gene list from a reply. Returns (genes, extraction_mode, dspy_parse_failed).

    The reference harness's happy path: DSPy parses the ``[[ ## answer ## ]]`` field, then
    ``parse_genes_from_output`` splits it on commas. When DSPy could not parse the reply it retried
    the request with its JSON adapter -- a second model call this single-shot server cannot make --
    and only if that also failed did it scan the raw reply with ``extract_genes_from_raw_response``.
    Here a DSPy parse failure goes straight to that scan; ``dspy_parse_failed`` is reported so the
    rate of this divergence is visible in the aggregate metrics.

    ``min_expected_genes`` is the harness's "quality gate": if the answer field yielded fewer genes
    than this, the raw-reply scan is tried and kept when it finds more. The harness set it to 20 but
    only reached it for its Biomni client, so it defaults to 0 (off) here.
    """
    fields = parse_dspy_completion(text)
    if fields is None:
        genes = extract_genes_from_raw_response(text)
        return genes, (EXTRACTION_RAW_FALLBACK if genes else EXTRACTION_NONE), True

    genes = parse_genes_from_output(fields["answer"])
    mode = EXTRACTION_DSPY_ANSWER
    if len(genes) < min_expected_genes:
        fallback = extract_genes_from_raw_response(text)
        if len(fallback) > len(genes):
            genes, mode = fallback, EXTRACTION_QUALITY_GATE
    if not genes:
        mode = EXTRACTION_NONE
    return genes, mode, False


def scalar_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Keep the numeric entries of a ``RankingMetrics.evaluate`` result, dropping lists and NaNs.

    NaN is what upstream emits for a normalized metric whose normalizer is zero; the reference
    aggregation skipped those values, and omitting the key makes Gym's aggregation skip them too.
    """
    out: Dict[str, float] = {}
    for key, value in results.items():
        if isinstance(value, bool) or isinstance(value, (list, tuple, dict, str)) or value is None:
            continue
        number = float(value)
        if math.isnan(number):
            continue
        out[key] = number
    return out


def score_assaybench_rollout(rollout: Dict[str, Any]) -> Dict[str, float]:
    """Named scores for ``compute_pass_majority_metrics``; keys absent from a rollout are skipped."""
    metrics = rollout.get("metrics") or {}
    scores = {name: metrics[name] for name in (*PAPER_METRICS, *DIAGNOSTIC_METRICS) if name in metrics}
    scores["dspy_parse_failed"] = float(bool(rollout.get("dspy_parse_failed", False)))
    scores["truncated"] = float(bool(rollout.get("truncated", False)))
    scores["empty_prediction"] = float(rollout.get("num_predicted", 0) == 0)
    return scores


class AssayBenchResourcesServerConfig(BaseResourcesServerConfig):
    # verify() is a pure function of the request body and this config.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    # `RankingMetrics(k_values=...)`; the reference results cache used exactly these.
    k_values: List[int] = [5, 10, 20, 50, 100]
    # Which entry of the metrics dict becomes `reward`. AnDCG@100 is the paper's headline metric.
    reward_metric: str = "adjusted_ndcg@100"
    # Canonicalize predicted symbols to HGNC with the package's GeneMapper (aliases, previous
    # symbols, "BBC3 (PUMA)"-style annotations). On in the reference evaluation.
    use_gene_mapper: bool = True
    # The reference harness's quality gate (see `extract_predicted_genes`). 0 disables it.
    min_expected_genes: int = 0
    # Drop a closed `<think>...</think>` block before parsing, for models whose reasoning arrives
    # inline rather than as a separate reasoning item. The reference harness served its open
    # models behind a vLLM reasoning parser, so it never saw the trace in `content` either.
    strip_think_blocks: bool = True


class AssayBenchRunRequest(BaseRunRequest):
    # Fields arrive as flat row columns (see benchmarks/assaybench/prepare.py); the validator below also accepts them
    # nested under `verifier_metadata`.
    verifier_metadata: Optional[Dict[str, Any]] = None

    relevance_genes: List[str]
    relevance_scores: List[float]
    dataset_name: Optional[str] = None
    split: Optional[str] = None
    question: Optional[str] = None
    cleaned_phenotype: Optional[str] = None
    screen_category: Optional[str] = None
    author: Optional[str] = None
    source_id: Optional[str] = None
    num_genes: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _lift_verifier_metadata(cls, data: Any) -> Any:
        """Accept the row's fields nested under `verifier_metadata` or at the top level. Top-level wins."""
        if isinstance(data, dict) and isinstance(data.get("verifier_metadata"), dict):
            return {**data["verifier_metadata"], **data}
        return data


class AssayBenchVerifyRequest(AssayBenchRunRequest, BaseVerifyRequest):
    pass


class AssayBenchVerifyResponse(AssayBenchVerifyRequest, BaseVerifyResponse):
    # Inherits the request fields so `cleaned_phenotype` reaches the rollout dict that
    # compute_subset_metrics groups by.
    predicted_genes: List[str]
    num_predicted: int
    extraction_mode: str
    dspy_parse_failed: bool
    truncated: bool
    # Slug of `cleaned_phenotype`, the grouping key for the per-phenotype breakdown (paper Fig. 3).
    phenotype_group: Optional[str] = None
    # Every scalar `RankingMetrics.evaluate` returned: adjusted_ndcg@k, precision@k, fdr@k, ...
    metrics: Dict[str, float]


class AssayBenchResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: AssayBenchResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        # Deferred import: the package loads its HGNC tables at import time (a few seconds).
        from assaybench.benchmark.metrics import RankingMetrics

        self._ranking_metrics = RankingMetrics(
            k_values=list(self.config.k_values),
            use_gene_mapper=self.config.use_gene_mapper,
        )
        if self._ranking_metrics.gene_mapper is not None:
            # GeneMapper builds its lookup tables lazily on first use. Force that here, on one
            # thread, so the concurrent verify() calls below only ever read them.
            self._ranking_metrics.gene_mapper.map_gene("TP53")

    def _evaluate(self, predicted_genes: List[str], genes: List[str], scores: List[float]) -> Dict[str, float]:
        results = self._ranking_metrics.evaluate(
            predicted_genes=predicted_genes,
            ground_truth_genes=genes,
            relevance_scores=scores,
        )
        return scalar_metrics(results)

    async def verify(self, body: AssayBenchVerifyRequest) -> AssayBenchVerifyResponse:
        """Score one reply: parse the ranked gene list, then run the paper's metrics on it."""
        text = body.response.output_text or ""
        if self.config.strip_think_blocks:
            text = THINK_BLOCK_PATTERN.sub("", text)

        genes, extraction_mode, dspy_parse_failed = extract_predicted_genes(text, self.config.min_expected_genes)

        # Pure CPU work over a ~14k-gene table; keep it off the event loop.
        metrics = await asyncio.to_thread(self._evaluate, genes, body.relevance_genes, body.relevance_scores)

        reward = metrics.get(self.config.reward_metric)
        failure_reason = None
        if reward is None:
            reward = 0.0
            failure_reason = f"Metric {self.config.reward_metric!r} was not computed for this rollout."
        if not genes:
            failure_reason = "No gene list could be recovered from the response."

        return AssayBenchVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            failure_reason=failure_reason,
            predicted_genes=genes,
            num_predicted=len(genes),
            extraction_mode=extraction_mode,
            dspy_parse_failed=dspy_parse_failed,
            truncated=is_truncated(body.response),
            phenotype_group=slugify_phenotype(body.cleaned_phenotype),
            metrics=metrics,
        )

    # ──────────────────────────────────────────────────────────
    # Aggregate metrics
    # ──────────────────────────────────────────────────────────

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """The paper's metrics pooled over screens, plus the per-phenotype breakdown of its Figure 3.

        ``pass@1[avg-of-k]/adjusted_ndcg@100`` is the paper's AnDCG@100 protocol: per-screen mean
        over the k runs, then mean over screens -- times 100, as every Gym aggregate is.
        """
        if not tasks:
            return {}

        metrics = compute_pass_majority_metrics(tasks, score_fn=score_assaybench_rollout)[0]
        metrics.update(compute_subset_metrics(tasks, subset_key="phenotype_group", score_fn=score_assaybench_rollout))
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline: token counts and the paper's three Table 2 columns at the highest k.

        The paper's "Precision@k" is normalized by ``min(k, G+)`` (its §3.2), which is the package's
        ``normalized_precision@k``; its "dFDR@k" is the plain ``fdr@k``.
        """
        key: Dict[str, Any] = {}

        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(
            highest_k_metrics(
                agent_metrics,
                "pass@1[avg-of-{k}]",
                score_names=[
                    "adjusted_ndcg@100",
                    "normalized_precision@100",  # the paper's "Precision@100" (divides by min(k, #hits))
                    "fdr@100",  # the paper's "dFDR@100"
                    "hallucination_rate",
                ],
            )
        )
        return key


if __name__ == "__main__":
    AssayBenchResourcesServer.run_webserver()
