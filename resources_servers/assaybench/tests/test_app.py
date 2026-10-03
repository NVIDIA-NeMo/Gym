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

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from benchmarks.assaybench.prepare import PROMPT_CONFIG_PATH, REPO_ROOT
from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config
from nemo_gym.server_utils import ServerClient
from resources_servers.assaybench.app import (
    EXTRACTION_DSPY_ANSWER,
    EXTRACTION_NONE,
    EXTRACTION_QUALITY_GATE,
    EXTRACTION_RAW_FALLBACK,
    AssayBenchResourcesServer,
    AssayBenchResourcesServerConfig,
    AssayBenchVerifyRequest,
    extract_predicted_genes,
    scalar_metrics,
)
from resources_servers.assaybench.gene_parsing import (
    extract_genes_from_raw_response,
    parse_dspy_completion,
    parse_genes_from_output,
)


DATA_DIR = Path(__file__).absolute().parent.parent / "data"

# Every data-backed test reads the committed 5-row synthetic `data/example.jsonl`. The real
# benchmark rows are gitignored and fetched from Hugging Face, so a test that read them would
# skip on every CI checkout.
EXAMPLE_ROWS = [json.loads(line) for line in (DATA_DIR / "example.jsonl").read_text(encoding="utf-8").splitlines()]
ROWS_BY_NAME = {row["dataset_name"]: row for row in EXAMPLE_ROWS}


def _dspy_reply(genes: List[str], reasoning: str = "Thinking about the biology.", completed: bool = True) -> str:
    text = f"[[ ## reasoning ## ]]\n{reasoning}\n\n[[ ## answer ## ]]\n{', '.join(genes)}\n"
    if completed:
        text += "\n[[ ## completed ## ]]\n"
    return text


def _ranked_hits(row: Dict[str, Any]) -> List[str]:
    """The screen's positive-relevance genes, strongest first: the ideal ranking."""
    pairs = [(gene, score) for gene, score in zip(row["relevance_genes"], row["relevance_scores"]) if score > 0]
    return [gene for gene, _ in sorted(pairs, key=lambda pair: -pair[1])]


def _paper_andcg(predicted: List[str], genes: List[str], scores: List[float], k: int) -> float:
    """AnDCG@k written out from Appendix B.2 of the paper, independently of the package."""
    relevance = dict(zip(genes, scores))
    values: List[Optional[float]] = [relevance.get(gene) for gene in predicted]
    values += [0.0] * (k - len(values))
    condensed = [value for value in values[:k] if value is not None]

    def dcg(sequence: List[float]) -> float:
        return sum(value / math.log2(index + 2) for index, value in enumerate(sequence))

    ideal = sorted((max(score, 0.0) for score in scores), reverse=True)[:k]
    idcg = dcg(ideal)
    ndcg = dcg(condensed) / idcg if idcg else 0.0
    mean_relevance = sum(scores) / len(scores)
    ndcg_random = dcg([mean_relevance] * min(k, len(scores))) / idcg if idcg else 0.0
    return max((ndcg - ndcg_random) / (1 - ndcg_random), 0.0)


class TestGeneParsing:
    def test_dspy_completion_splits_into_fields(self) -> None:
        fields = parse_dspy_completion(_dspy_reply(["TP53", "MYC"], reasoning="because"))
        assert fields == {"reasoning": "because", "answer": "TP53, MYC"}
        # The `[[ ## completed ## ]]` marker is optional, and a header may carry content on its own line.
        assert parse_dspy_completion(_dspy_reply(["TP53"], completed=False))["answer"] == "TP53"
        assert parse_dspy_completion("[[ ## reasoning ## ]] r\n[[ ## answer ## ]] TP53, MYC") == {
            "reasoning": "r",
            "answer": "TP53, MYC",
        }

    def test_dspy_first_section_wins_and_preamble_is_ignored(self) -> None:
        # ChatAdapter.parse keeps the first occurrence of each field and drops text before any header.
        text = "Sure! Here goes.\n[[ ## reasoning ## ]]\nr1\n[[ ## answer ## ]]\nTP53\n[[ ## answer ## ]]\nMYC\n"
        assert parse_dspy_completion(text) == {"reasoning": "r1", "answer": "TP53"}

    def test_dspy_requires_every_output_field(self) -> None:
        # An answer with no reasoning section is an AdapterParseError upstream.
        assert parse_dspy_completion("[[ ## answer ## ]]\nTP53, MYC\n") is None
        assert parse_dspy_completion("TP53, MYC, KRAS") is None

    def test_parse_genes_from_output_filters_non_symbols(self) -> None:
        # Upstream behaviour, quirks included: the symbol regex is anchored, so the "extract a
        # symbol from inside a noisy token" branch never fires -- "1. MYC" and "KRAS\\end{...}" are
        # dropped, and so is C1orf43 (a real HGNC symbol with lowercase letters).
        text = "TP53, brca1, 1. MYC, HLA-A, C1orf43, DATA, KRAS\\end{solution}, , this is not a gene"
        assert parse_genes_from_output(text) == ["TP53", "HLA-A"]
        assert parse_genes_from_output(None) == parse_genes_from_output("") == []

    def test_raw_fallback_prefers_longest_comma_line_then_numbered_list(self) -> None:
        text = "Top hits:\nTP53, MYC, KRAS, EGFR, BRCA1, PTEN\nAlso: RB1, CDK1, PLK1, AURKB, RAN\n"
        assert extract_genes_from_raw_response(text) == ["TP53", "MYC", "KRAS", "EGFR", "BRCA1", "PTEN"]
        numbered = "\n".join(f"{i + 1}. {gene}" for i, gene in enumerate(["TP53", "MYC", "KRAS", "EGFR", "BRCA1"]))
        assert extract_genes_from_raw_response(numbered) == ["TP53", "MYC", "KRAS", "EGFR", "BRCA1"]
        # Fewer than five gene-like tokens anywhere is "no list".
        assert extract_genes_from_raw_response("Maybe TP53 or MYC, hard to say.") == []
        assert extract_genes_from_raw_response("") == []


class TestExtractPredictedGenes:
    def test_happy_path_reads_only_the_answer_field(self) -> None:
        # The reasoning mentions genes too; only the answer field is read on the happy path.
        text = _dspy_reply(["TP53", "MYC", "KRAS"], reasoning="EGFR and KRAS matter")
        assert extract_predicted_genes(text) == (["TP53", "MYC", "KRAS"], EXTRACTION_DSPY_ANSWER, False)

    def test_unparseable_reply_falls_back_to_the_raw_scan(self) -> None:
        genes, mode, failed = extract_predicted_genes("Here you go: TP53, MYC, KRAS, EGFR, BRCA1, PTEN")
        assert genes == ["TP53", "MYC", "KRAS", "EGFR", "BRCA1", "PTEN"]
        assert (mode, failed) == (EXTRACTION_RAW_FALLBACK, True)

    def test_nothing_recoverable(self) -> None:
        assert extract_predicted_genes("I cannot help with that.") == ([], EXTRACTION_NONE, True)
        assert extract_predicted_genes(_dspy_reply([], reasoning="no idea")) == ([], EXTRACTION_NONE, False)

    def test_quality_gate_is_off_by_default_and_rescans_when_enabled(self) -> None:
        text = "[[ ## reasoning ## ]]\nTP53, MYC, KRAS, EGFR, BRCA1, PTEN, RB1\n\n[[ ## answer ## ]]\nTP53, MYC\n"
        assert extract_predicted_genes(text)[0] == ["TP53", "MYC"]
        genes, mode, _ = extract_predicted_genes(text, min_expected_genes=20)
        assert genes == ["TP53", "MYC", "KRAS", "EGFR", "BRCA1", "PTEN", "RB1"]
        assert mode == EXTRACTION_QUALITY_GATE


class TestHelpers:
    def test_scalar_metrics_drops_lists_and_nans(self) -> None:
        cleaned = scalar_metrics(
            {"adjusted_ndcg@100": 0.5, "predicted_genes": ["TP53"], "normalized_fdr@100": float("nan"), "n": 3}
        )
        assert cleaned == {"adjusted_ndcg@100": 0.5, "n": 3.0}


class TestAssayBenchApp:
    @pytest.fixture(scope="class")
    def server(self) -> AssayBenchResourcesServer:
        config = AssayBenchResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="assaybench")
        return AssayBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_response(self, text: str, status: Optional[str] = None, incomplete_reason: Optional[str] = None):
        return NeMoGymResponse(
            id="test_response_id",
            created_at=1234567890.0,
            model="test_model",
            object="response",
            status=status,
            incomplete_details={"reason": incomplete_reason} if incomplete_reason else None,
            output=[
                NeMoGymResponseOutputMessage(
                    id="test_msg",
                    role="assistant",
                    type="message",
                    content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    def _create_request(self, text: str, row: Dict[str, Any], **response_kwargs) -> AssayBenchVerifyRequest:
        return AssayBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_response(text, **response_kwargs),
            **{key: value for key, value in row.items() if key not in ("question", "responses_create_params")},
        )

    @pytest.mark.asyncio
    async def test_ideal_ranking_scores_one(self, server) -> None:
        row = ROWS_BY_NAME["example_5"]
        result = await server.verify(self._create_request(_dspy_reply(_ranked_hits(row)), row))
        assert result.reward == pytest.approx(1.0)
        assert result.extraction_mode == EXTRACTION_DSPY_ANSWER
        assert result.num_predicted == 5
        assert result.metrics["adjusted_ndcg@100"] == pytest.approx(1.0)
        assert result.metrics["hallucination_rate"] == 0.0
        assert result.failure_reason is None

    @pytest.mark.asyncio
    async def test_reward_is_the_paper_formula(self, server) -> None:
        # A partial, reordered list with one out-of-screen gene, checked against Appendix B.2
        # written out by hand -- so this pins the package to the paper, not the paper to the package.
        # (With a 40-gene library the random baseline is high -- a random list covers the whole
        # screen -- so the list has to be mostly right to land strictly between 0 and 1.)
        row = ROWS_BY_NAME["example_2"]
        predicted = ["PLK1", "RAN", "CDK1", "AAAS", "RPL15", "RPS28", "EGFR", "SF3B5", "AURKB", "UBL5", "MYC"]
        result = await server.verify(self._create_request(_dspy_reply(predicted), row))
        expected = _paper_andcg(predicted, row["relevance_genes"], row["relevance_scores"], k=100)
        assert 0.0 < expected < 1.0
        assert result.reward == pytest.approx(expected)
        assert result.metrics["adjusted_ndcg@100"] == pytest.approx(expected)
        # AAAS is a real HGNC symbol not in this 40-gene library: unscored, not a hallucination.
        assert result.metrics["hallucination_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_gene_mapper_canonicalizes_aliases(self, server) -> None:
        # Aliases and previous symbols resolve to the approved symbol before scoring. (Lowercase
        # "p53" never reaches the mapper: the upstream answer parser only keeps uppercase tokens.)
        row = ROWS_BY_NAME["example_1"]
        hits = _ranked_hits(row)
        aliased = ["P53" if gene == "TP53" else "WAF1" if gene == "CDKN1A" else gene for gene in hits]
        assert aliased != hits
        canonical = await server.verify(self._create_request(_dspy_reply(hits), row))
        result = await server.verify(self._create_request(_dspy_reply(aliased), row))
        assert canonical.reward == pytest.approx(1.0)
        assert result.reward == pytest.approx(canonical.reward)
        assert result.metrics["num_predictions_normalized"] == 2
        assert result.metrics["hallucination_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_opposite_direction_hits_are_penalized(self, server) -> None:
        row = ROWS_BY_NAME["example_3_inc"]
        good = await server.verify(self._create_request(_dspy_reply(["GPX4", "SLC7A11", "NF2"]), row))
        bad = await server.verify(self._create_request(_dspy_reply(["ACSL4", "TP53", "GPX4", "SLC7A11", "NF2"]), row))
        assert good.reward > bad.reward
        assert bad.metrics["fdr@5"] == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_empty_or_refused_response_scores_zero(self, server) -> None:
        row = ROWS_BY_NAME["example_4"]
        for text in ("", "I'm sorry, I can't predict experimental outcomes."):
            result = await server.verify(self._create_request(text, row))
            assert result.reward == 0.0
            assert result.extraction_mode == EXTRACTION_NONE
            assert result.dspy_parse_failed is True
            assert result.num_predicted == 0
            assert result.failure_reason is not None

    @pytest.mark.asyncio
    async def test_unparseable_reply_is_scored_from_the_raw_scan(self, server) -> None:
        row = ROWS_BY_NAME["example_4"]
        text = "Ranked candidates:\nACE2, TMPRSS2, CTSL, VPS35, SNX27, EGFR\n"
        result = await server.verify(self._create_request(text, row))
        assert result.reward == pytest.approx(1.0)
        assert result.extraction_mode == EXTRACTION_RAW_FALLBACK
        assert result.dspy_parse_failed is True

    @pytest.mark.asyncio
    async def test_think_block_is_stripped_before_parsing(self, server) -> None:
        row = ROWS_BY_NAME["example_5"]
        text = "<think>\n[[ ## answer ## ]]\nEGFR, KRAS\n</think>\n" + _dspy_reply(_ranked_hits(row))
        result = await server.verify(self._create_request(text, row))
        assert result.reward == pytest.approx(1.0)
        assert result.predicted_genes[0] == "IFNAR1"

    @pytest.mark.asyncio
    async def test_precision_metrics_only_exist_at_reachable_cutoffs(self, server) -> None:
        # RankingMetrics emits precision@k / fdr@k only when the model produced >= k genes. A
        # rollout without the key drops out of that metric's mean, as in the reference aggregation.
        row = ROWS_BY_NAME["example_2"]
        short = await server.verify(self._create_request(_dspy_reply(_ranked_hits(row)), row))
        assert "precision@10" in short.metrics and "precision@100" not in short.metrics
        hundred = _ranked_hits(row) + [f"ZZ{i}" for i in range(100)]
        long = await server.verify(self._create_request(_dspy_reply(hundred), row))
        assert long.metrics["precision@100"] == pytest.approx(1.0)
        assert long.metrics["normalized_precision@100"] == pytest.approx(1.0)
        assert long.metrics["fdr@100"] == 0.0
        assert long.metrics["hallucination_rate"] == pytest.approx(100 / 111)

    @pytest.mark.asyncio
    async def test_row_metadata_survives_onto_the_response(self, server) -> None:
        row = ROWS_BY_NAME["example_1"]
        result = await server.verify(self._create_request(_dspy_reply(["TOP2A"]), row))
        assert result.dataset_name == "example_1"
        assert result.cleaned_phenotype == "Drug / Chemical / Environmental Response"
        assert result.phenotype_group == "drug_chemical_environmental_response"
        assert result.split == "example"
        assert result.num_genes == 40

    @pytest.mark.asyncio
    async def test_truncation_is_reported_but_the_partial_list_is_still_scored(self, server) -> None:
        row = ROWS_BY_NAME["example_5"]
        text = "[[ ## reasoning ## ]]\nlong\n\n[[ ## answer ## ]]\nIFNAR1, JAK1, TY"
        result = await server.verify(
            self._create_request(text, row, status="incomplete", incomplete_reason="max_output_tokens")
        )
        assert result.truncated is True
        assert result.predicted_genes == ["IFNAR1", "JAK1", "TY"]
        assert result.reward > 0.0

    @pytest.mark.asyncio
    async def test_accepts_fields_nested_under_verifier_metadata(self, server) -> None:
        row = ROWS_BY_NAME["example_5"]
        request = AssayBenchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_response(_dspy_reply(_ranked_hits(row))),
            verifier_metadata={
                key: value for key, value in row.items() if key not in ("question", "responses_create_params")
            },
        )
        result = await server.verify(request)
        assert result.reward == pytest.approx(1.0)
        assert result.cleaned_phenotype == row["cleaned_phenotype"]

    @pytest.mark.asyncio
    async def test_unknown_reward_metric_is_reported_not_raised(self) -> None:
        config = AssayBenchResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name="assaybench", reward_metric="precision@100"
        )
        server = AssayBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        row = ROWS_BY_NAME["example_5"]
        result = await server.verify(self._create_request(_dspy_reply(_ranked_hits(row)), row))
        assert result.reward == 0.0
        assert "precision@100" in result.failure_reason

    @pytest.mark.asyncio
    async def test_aggregate_metrics_follow_the_paper_protocol(self, server) -> None:
        def rollout(task: int, index: int, andcg: float, phenotype: str, **extra) -> Dict[str, Any]:
            metrics = {"adjusted_ndcg@100": andcg, "hallucination_rate": 0.1, **extra.pop("metrics", {})}
            return {
                TASK_INDEX_KEY_NAME: task,
                ROLLOUT_INDEX_KEY_NAME: index,
                "reward": andcg,
                "metrics": metrics,
                "phenotype_group": phenotype,
                "num_predicted": 100,
                "dspy_parse_failed": False,
                "truncated": False,
                **extra,
            }

        responses = [
            rollout(0, 0, 0.2, "fitness_proliferation_viability", metrics={"normalized_precision@100": 0.5}),
            rollout(0, 1, 0.4, "fitness_proliferation_viability"),
            rollout(1, 0, 0.1, "host_pathogen_infection_response", dspy_parse_failed=True),
            rollout(1, 1, 0.3, "host_pathogen_infection_response", num_predicted=0, truncated=True),
        ]
        result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=responses))
        metrics = result.agent_metrics

        # Per-screen mean over runs, then mean over screens, times 100: (0.3 + 0.2) / 2.
        assert metrics["pass@1[avg-of-2]/adjusted_ndcg@100"] == pytest.approx(25.0)
        assert metrics["pass@2/adjusted_ndcg@100"] == pytest.approx(35.0)  # max over runs
        # normalized_precision@100 is only present on one rollout; the others do not drag it down.
        assert metrics["pass@1[avg-of-2]/normalized_precision@100"] == pytest.approx(50.0)
        assert metrics["pass@1[avg-of-2]/dspy_parse_failed"] == pytest.approx(25.0)
        assert metrics["pass@1[avg-of-2]/truncated"] == pytest.approx(25.0)
        assert metrics["pass@1[avg-of-2]/empty_prediction"] == pytest.approx(25.0)
        assert metrics["fitness_proliferation_viability/pass@1[avg-of-2]/adjusted_ndcg@100"] == pytest.approx(30.0)
        assert metrics["host_pathogen_infection_response/pass@1[avg-of-2]/adjusted_ndcg@100"] == pytest.approx(20.0)

        key = result.key_metrics
        assert key["pass@1[avg-of-2]/adjusted_ndcg@100"] == pytest.approx(25.0)
        assert key["pass@1[avg-of-2]/normalized_precision@100"] == pytest.approx(50.0)
        assert key["pass@1[avg-of-2]/hallucination_rate"] == pytest.approx(10.0)
        assert "pass@1[avg-of-2]/truncated" not in key


class TestPrompt:
    """The rendered messages must be byte-identical to what the reference harness sent.

    The expected strings were captured from dspy 3.3.1 with
    ``ChatAdapter().format(ChainOfThought(RankingSignature).predict.signature, demos=[], inputs=...)``
    for the signature defined in the harness's ``create_ranking_signature``.
    """

    DSPY_SYSTEM = (
        "Your input fields are:\n"
        "1. `question` (str): The gene ranking task description\n"
        "Your output fields are:\n"
        "1. `reasoning` (str): \n"
        "2. `answer` (str): Comma-separated list of genes in ranked order\n"
        "All interactions will be structured in the following way, with the appropriate values filled in.\n"
        "\n"
        "[[ ## question ## ]]\n"
        "{question}\n"
        "\n"
        "[[ ## reasoning ## ]]\n"
        "{reasoning}\n"
        "\n"
        "[[ ## answer ## ]]\n"
        "{answer}\n"
        "\n"
        "[[ ## completed ## ]]\n"
        "In adhering to this structure, your objective is: \n"
        "        Signature for gene ranking task."
    )
    COLLECTION_SUFFIX = (
        "\n\nYour goal is to provide a list of genes that meet the screen criteria, even if you do not have "
        "access to the actual experimental data. The genes must use HGNC symbols. Use your knowledge of "
        "biology, gene function, and relevant pathways to predict which genes are most likely to be hits. Do "
        "not refuse to answer or say you need more data—make your best predictions based on your "
        "understanding of the biological context."
    )
    DSPY_USER_TAIL = (
        "\n\nRespond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, "
        "then `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."
    )

    def test_rendered_messages_match_the_dspy_capture(self) -> None:
        prompt_config = load_prompt_config(str(REPO_ROOT / PROMPT_CONFIG_PATH))
        row = ROWS_BY_NAME["example_1"]
        messages = apply_prompt_to_row(row, prompt_config)["responses_create_params"]["input"]
        # The committed fixture was rendered with the same template.
        assert messages == row["responses_create_params"]["input"]
        assert [message["role"] for message in messages] == ["system", "user"]
        assert messages[0]["content"] == self.DSPY_SYSTEM
        assert messages[1]["content"] == (
            "[[ ## question ## ]]\n" + row["question"] + self.COLLECTION_SUFFIX + self.DSPY_USER_TAIL
        )


class TestRawFallbackStrategies:
    """The vendored whole-reply scan, strategy by strategy (upstream's Biomni path included)."""

    GENES = ["TP53", "MYC", "KRAS", "EGFR", "BRCA1", "PTEN"]

    def test_solution_block_answer_field_wins_when_it_holds_a_list(self) -> None:
        genes = ", ".join(self.GENES)
        block = "\\begin{solution}\n" + json.dumps({"answer": genes}) + "\n\\end{solution}"
        assert extract_genes_from_raw_response("tool output...\n" + block) == self.GENES
        # Truncated JSON: the answer field is recovered by regex instead.
        truncated = '\\begin{solution}\n{"answer": "' + genes + "\n\\end{solution}"
        assert extract_genes_from_raw_response(truncated) == self.GENES
        # A short answer field is ignored in favour of a proper gene line elsewhere.
        short = '\\begin{solution}\n{"answer": "TP53, MYC"}\n\\end{solution}\n' + genes
        assert extract_genes_from_raw_response(short) == self.GENES

    def test_prose_without_separators_uses_the_token_sweep(self) -> None:
        text = "I would look at TP53 then MYC then KRAS then EGFR then BRCA1 for this screen."
        assert extract_genes_from_raw_response(text) == ["TP53", "MYC", "KRAS", "EGFR", "BRCA1"]
