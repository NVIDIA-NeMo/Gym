# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from app import MolDeToxResourcesServer, MolDeToxStatus, MolDeToxVerifyRequest, _extract_json_answer
from metrics import canonical_smiles, decode_safe, exact_match, is_valid, levenshtein, score_smiles, tanimoto

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


# Task 3 single-step row id=2, endpoint cyp2c19_veith. The edit is a benzene ring
# to a pyrimidine, so the toxic input is a close but distinct structure.
TOXIC = "Cc1cccc(CNc2cc(-c3c(C)noc3C)ncn2)c1"
GOLD = "Cc1cnc(CNc2cc(-c3c(C)noc3C)ncn2)cn1"

# The same molecule as GOLD written with a different atom ordering. GOLD is
# already RDKit's canonical output, so canonicalising it returns it unchanged and
# cannot stand in for a model's own spelling.
GOLD_NONCANONICAL = "Cc1cnc(CNc2cc(ncn2)-c2c(onc2C)C)cn1"


def _make_server() -> MolDeToxResourcesServer:
    config = BaseResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return MolDeToxResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_request(text: str, gold: str = GOLD, **meta_overrides) -> MolDeToxVerifyRequest:
    meta = {
        "answer": gold,
        "scoring_mode": "smiles",
        "endpoint": "cyp2c19_veith",
        "dataset_name": "metabolism",
        "toxic_smiles": TOXIC,
    }
    meta.update(meta_overrides)
    return MolDeToxVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Q"}]},
        response=_make_response(text),
        verifier_metadata=meta,
    )


class TestVerify:
    async def test_gold_answer_scores_one(self) -> None:
        result = await _make_server().verify(_make_request(f'{{"answer": "{GOLD}"}}'))
        assert result.reward == 1.0
        assert result.status == MolDeToxStatus.SCORED.value
        assert result.validity == 1.0
        assert result.tanimoto_morgan == 1.0
        assert result.levenshtein == 0.0

    async def test_noncanonical_form_of_gold_still_matches(self) -> None:
        """Structural equality, not string equality: a model may emit any valid writing.

        The two strings must differ as text and agree as molecules, or the test
        passes just as well against a verifier that compares raw strings.
        """
        assert GOLD_NONCANONICAL != GOLD
        assert canonical_smiles(GOLD_NONCANONICAL) == canonical_smiles(GOLD)
        result = await _make_server().verify(_make_request(f'{{"answer": "{GOLD_NONCANONICAL}"}}'))
        assert result.reward == 1.0

    async def test_echoing_the_toxic_input_scores_zero(self) -> None:
        """The degenerate no-edit baseline must fail, while similarity stays high."""
        result = await _make_server().verify(_make_request(f'{{"answer": "{TOXIC}"}}'))
        assert result.reward == 0.0
        assert result.status == MolDeToxStatus.SCORED.value
        assert result.validity == 1.0
        assert result.tanimoto_morgan > 0.5

    async def test_wrong_but_valid_molecule_scores_zero(self) -> None:
        result = await _make_server().verify(_make_request('{"answer": "CCO"}'))
        assert result.reward == 0.0
        assert result.validity == 1.0
        assert result.tanimoto_morgan < 0.2

    async def test_invalid_smiles_is_scored_but_invalid(self) -> None:
        result = await _make_server().verify(_make_request('{"answer": "C1CC"}'))
        assert result.reward == 0.0
        assert result.status == MolDeToxStatus.SCORED.value
        assert result.validity == 0.0
        assert result.levenshtein is None

    async def test_answer_in_fenced_block_with_prose(self) -> None:
        text = f'Replacing the benzene with a pyrimidine:\n\n```json\n{{"answer": "{GOLD}"}}\n```\n'
        result = await _make_server().verify(_make_request(text))
        assert result.reward == 1.0
        assert result.extracted_answer == GOLD

    async def test_last_answer_wins_when_model_revises(self) -> None:
        text = f'First thought: {{"answer": "CCO"}}\nOn reflection: {{"answer": "{GOLD}"}}'
        result = await _make_server().verify(_make_request(text))
        assert result.reward == 1.0

    async def test_a_later_bare_answer_beats_an_earlier_fenced_draft(self) -> None:
        """Candidates are ordered by position, not by kind.

        Ordering fenced blocks ahead of bare objects scored this reply on the
        discarded draft, and the bare-vs-bare case above stayed green throughout.
        """
        text = f'Let me think.\n```json\n{{"answer": "CCO"}}\n```\nOn reflection: {{"answer": "{GOLD}"}}'
        result = await _make_server().verify(_make_request(text))
        assert result.reward == 1.0
        assert result.extracted_answer == GOLD

    async def test_prose_without_json_is_no_answer(self) -> None:
        result = await _make_server().verify(_make_request("I would replace the benzene ring."))
        assert result.status == MolDeToxStatus.NO_ANSWER.value
        assert result.reward == 0.0

    async def test_empty_output(self) -> None:
        result = await _make_server().verify(_make_request(""))
        assert result.status == MolDeToxStatus.EMPTY_OUTPUT.value
        assert result.reward == 0.0

    async def test_oversized_answer_is_rejected_before_rdkit_sees_it(self) -> None:
        """A 15,000-character chain segfaults `Chem.MolToSmiles`, killing the process.

        Reaching the assertions at all is the point of this test: before the cap,
        this call terminated the interpreter rather than returning.
        """
        result = await _make_server().verify(_make_request('{"answer": "%s"}' % ("C" * 15000)))
        assert result.status == MolDeToxStatus.ANSWER_TOO_LONG.value
        assert result.reward == 0.0
        assert result.validity is None

    async def test_unknown_scoring_mode_is_flagged_not_failed(self) -> None:
        """An unscorable row must not masquerade as a model miss."""
        result = await _make_server().verify(_make_request(f'{{"answer": "{GOLD}"}}', scoring_mode="nonexistent"))
        assert result.status == MolDeToxStatus.UNSUPPORTED_MODE.value
        assert result.reward == 0.0

    async def test_unparseable_gold_is_flagged(self) -> None:
        result = await _make_server().verify(_make_request('{"answer": "CCO"}', gold="not-a-molecule"))
        assert result.status == MolDeToxStatus.BAD_GOLD.value

    async def test_metadata_is_echoed_for_stratification(self) -> None:
        result = await _make_server().verify(_make_request(f'{{"answer": "{GOLD}"}}'))
        assert result.endpoint == "cyp2c19_veith"
        assert result.dataset_name == "metabolism"


# Same row id=2, expressed as SAFE. GOLD_SAFE decodes to GOLD.
GOLD_SAFE = "Cc1cnc6cn1.c15cc7ncn1.c17c(C)noc1C.C46.N45"
TOXIC_SAFE = "Cc1cccc6c1.c15cc7ncn1.c17c(C)noc1C.C46.N45"


class TestSafeMode:
    async def test_gold_safe_string_scores_one(self) -> None:
        result = await _make_server().verify(
            _make_request(f'{{"answer": "{GOLD_SAFE}"}}', gold=GOLD_SAFE, scoring_mode="safe")
        )
        assert result.reward == 1.0
        assert result.validity == 1.0

    async def test_safe_and_smiles_agree_on_the_same_molecule(self) -> None:
        """The two Task 3 configs encode one molecule two ways; both must score it identically."""
        assert decode_safe(GOLD_SAFE) is not None
        assert canonical_smiles(decode_safe(GOLD_SAFE)) == canonical_smiles(GOLD)

    async def test_undecodable_safe_prediction_is_invalid(self) -> None:
        """Upstream counts a failed SAFE reconstruction as invalid rather than masking it."""
        result = await _make_server().verify(
            _make_request('{"answer": "not.a.safe.string"}', gold=GOLD_SAFE, scoring_mode="safe")
        )
        assert result.reward == 0.0
        assert result.validity == 0.0

    async def test_echoing_toxic_safe_scores_zero(self) -> None:
        result = await _make_server().verify(
            _make_request(f'{{"answer": "{TOXIC_SAFE}"}}', gold=GOLD_SAFE, scoring_mode="safe")
        )
        assert result.reward == 0.0
        assert result.validity == 1.0


class TestFragmentMode:
    GOLD_FRAGS = "C14CSC3O1.Nc1ccn4c(=O)n1"

    async def _score(self, answer: str, gold: str = GOLD_FRAGS, task: str = "task2"):
        return await _make_server().verify(
            _make_request(f'{{"answer": "{answer}"}}', gold=gold, scoring_mode=f"fragments_{task}")
        )

    async def test_exact_fragment_set_scores_one(self) -> None:
        result = await self._score(self.GOLD_FRAGS)
        assert result.reward == 1.0
        assert result.fragment_f1 == 1.0

    async def test_fragment_order_does_not_matter(self) -> None:
        """SAFE is order-invariant, so the multiset is what counts."""
        result = await self._score("Nc1ccn4c(=O)n1.C14CSC3O1")
        assert result.reward == 1.0

    async def test_partial_overlap_scores_zero_with_partial_f1(self) -> None:
        result = await self._score("C14CSC3O1.CCO")
        assert result.reward == 0.0
        assert 0.0 < result.fragment_f1 < 1.0

    async def test_missing_a_fragment_is_not_exact(self) -> None:
        result = await self._score("C14CSC3O1")
        assert result.reward == 0.0
        assert result.fragment_f1 == pytest.approx(2 / 3)

    async def test_molecule_metrics_absent_in_fragment_mode(self) -> None:
        """Fragments are unparseable as molecules; absent metrics must be None, not 0.0."""
        result = await self._score(self.GOLD_FRAGS)
        assert result.validity is None
        assert result.tanimoto_morgan is None

    async def test_repeated_fragment_does_not_dilute_f1(self) -> None:
        """F1 is defined over sets (Appendix D.1), so a repeat neither helps nor hurts.

        Scoring the multiset instead returns 0.5 here. Reward is unaffected either
        way, because exact match is defined over the multiset and this is a miss.
        """
        first, second = self.GOLD_FRAGS.split(".")
        result = await self._score(f"{first}.{first}")
        assert result.fragment_f1 == pytest.approx(2 / 3)
        assert result.reward == 0.0
        assert second not in (result.extracted_answer or "")

    async def test_task2_distance_is_per_fragment_not_whole_string(self) -> None:
        """Appendix D.2 takes each predicted fragment's nearest gold fragment.

        Both predicted fragments here are gold fragments, so every minimum is 0
        and the mean is 0, even though the strings differ by a whole fragment's
        worth of characters. A whole-string edit distance returns 16.
        """
        first, _ = self.GOLD_FRAGS.split(".")
        result = await self._score(f"{first}.{first}")
        assert result.levenshtein == pytest.approx(0.0)

    async def test_task1_reports_no_edit_distance(self) -> None:
        """Appendix D.1 defines no distance for Task 1, so none may be emitted.

        Reporting one would publish a number with no upstream counterpart, and a
        default of 0.0 would read as a perfect score.
        """
        result = await self._score("C14CSC3O1", task="task1")
        assert result.levenshtein is None
        assert result.fragment_f1 == pytest.approx(2 / 3)
        assert result.reward == 0.0


class TestPreparerScope:
    """The README states nothing here reads the ~202,000-row train split.

    A free-form `--split` string made that claim false: it was interpolated
    straight into both the QA and toxicity-cliff download paths, so `--split
    train` downloaded and prepared train data despite the declared scope.
    """

    @staticmethod
    def _parser():
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import prepare_moldetox

        return prepare_moldetox.build_parser()

    def test_train_split_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            self._parser().parse_args(["--output", "/dev/null", "--split", "train"])

    def test_test_split_is_accepted_and_is_the_default(self) -> None:
        parser = self._parser()
        assert parser.parse_args(["--output", "/dev/null"]).split == "test"
        assert parser.parse_args(["--output", "/dev/null", "--split", "test"]).split == "test"


class TestSystemPrompt:
    """Upstream publishes a system prompt per task family (Tables H, I, J, K).

    An earlier version sent none, on the reasoning that the shipped `question`
    already names the JSON contract. It does, but the system prompts add a HARD
    CONSTRAINTS block it does not, and omitting them was a measurable divergence.
    """

    @staticmethod
    def _prepare():
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import prepare_moldetox

        return prepare_moldetox

    def test_every_config_maps_to_a_system_prompt(self) -> None:
        from prompts import SYSTEM_PROMPTS

        prepare = self._prepare()
        assert set(SYSTEM_PROMPTS) == set(prepare.SUPPORTED_CONFIGS)

    def test_system_message_precedes_the_question(self) -> None:
        prepare = self._prepare()
        row = {
            "question": "Q",
            "answer": {"answer": "CCO"},
            "endpoint": "e",
            "dataset_name": "d",
            "source_index": 0,
            "id": 1,
        }
        out = prepare.format_row(row, "task3_smiles_gen_single", {})
        messages = out["responses_create_params"]["input"]
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[1]["content"] == "Q"

    def test_the_hard_constraints_block_is_present(self) -> None:
        """The block is the part the `question` does not carry, so it is the part worth asserting."""
        from prompts import SYSTEM_PROMPTS

        for config, prompt in SYSTEM_PROMPTS.items():
            assert "HARD CONSTRAINTS:" in prompt, config
            assert "Output ONLY the JSON object." in prompt, config

    def test_task_families_get_different_prompts(self) -> None:
        """SMILES and SAFE generation differ in what they ask for; one prompt for both would be wrong."""
        from prompts import SYSTEM_PROMPTS

        smiles = SYSTEM_PROMPTS["task3_smiles_gen_single"]
        safe = SYSTEM_PROMPTS["task3_safe_gen_single"]
        assert smiles != safe
        assert "Do not return SAFE fragments." in smiles
        assert "full non-toxic SAFE string" in safe

    def test_opt_out_reproduces_the_no_system_prompt_shape(self) -> None:
        """The flag exists so the divergence stays measurable, so it must actually drop the turn."""
        prepare = self._prepare()
        row = {
            "question": "Q",
            "answer": {"answer": "CCO"},
            "endpoint": "e",
            "dataset_name": "d",
            "source_index": 0,
            "id": 1,
        }
        out = prepare.format_row(row, "task3_smiles_gen_single", {}, system_prompt=False)
        assert [m["role"] for m in out["responses_create_params"]["input"]] == ["user"]


class TestMalformedMetadata:
    """`verifier_metadata` is `dict[str, Any]`, so the types are not guaranteed.

    Each of these raised before the guards were added, and an exception here is
    an HTTP 500 that aborts the eval rather than scoring one row zero.
    """

    async def _verify(self, **meta_overrides):
        return await _make_server().verify(_make_request(f'{{"answer": "{GOLD}"}}', **meta_overrides))

    async def test_non_string_scoring_mode_does_not_raise(self) -> None:
        """A list is unhashable, so the `SCORERS` lookup itself raised TypeError."""
        result = await self._verify(scoring_mode=[])
        assert result.status == MolDeToxStatus.BAD_METADATA.value
        assert result.reward == 0.0
        assert result.failure_reason is not None

    async def test_non_string_gold_does_not_raise(self) -> None:
        """`_gold_is_usable` called `.strip()`, raising AttributeError on an int."""
        result = await self._verify(answer=7, scoring_mode="fragments_task2")
        assert result.status == MolDeToxStatus.BAD_METADATA.value
        assert result.reward == 0.0
        assert result.failure_reason is not None

    async def test_non_string_provenance_is_dropped_not_fatal(self) -> None:
        """Echoing a list raised a pydantic ValidationError building the response.

        The label is provenance only, so losing it must not cost the row: the
        prediction is still scored, and only the unusable label becomes None.
        """
        result = await self._verify(endpoint=[])
        assert result.status == MolDeToxStatus.SCORED.value
        assert result.reward == 1.0
        assert result.endpoint is None

    async def test_failure_reason_marks_harness_faults_only(self) -> None:
        """A zero must be attributable: data fault, or the model actually missing.

        Model-behaviour zeros carry no reason, so a consumer filtering on
        `failure_reason` does not silently discard real misses.
        """
        unscorable = await self._verify(answer=None)
        assert unscorable.status == MolDeToxStatus.BAD_GOLD.value
        assert unscorable.failure_reason is not None

        real_miss = await _make_server().verify(_make_request('{"answer": "CCO"}'))
        assert real_miss.status == MolDeToxStatus.SCORED.value
        assert real_miss.reward == 0.0
        assert real_miss.failure_reason is None


class TestKeyMetrics:
    """The headline line must not carry similarity metrics.

    Echoing the toxic input scores 0.711 mean Morgan Tanimoto and zero accuracy,
    so promoting similarity would advertise the degenerate non-edit as partial
    success. The default `get_key_metrics` promotes every `mean/*` entry.
    """

    AGENT_METRICS = {
        "mean/reward": 0.12,
        "mean/validity": 0.95,
        "mean/tanimoto_morgan": 0.711,
        "mean/tanimoto_rdk": 0.64,
        "mean/tanimoto_maccs": 0.80,
        "mean/levenshtein": 6.7,
        "mean/fragment_f1": 0.4,
        "mean/input_tokens": 575.8,
        "mean/output_tokens": 33.4,
        "mean/total_tokens": 609.2,
        "max/reward": 1.0,
    }

    def test_similarity_metrics_are_not_promoted(self) -> None:
        key = _make_server().get_key_metrics(self.AGENT_METRICS)
        assert "mean/reward" in key
        for name in (
            "mean/tanimoto_morgan",
            "mean/tanimoto_rdk",
            "mean/tanimoto_maccs",
            "mean/validity",
            "mean/levenshtein",
            "mean/fragment_f1",
        ):
            assert name not in key, f"{name} must not be a headline metric"

    def test_token_counts_are_promoted_and_nothing_else(self) -> None:
        key = _make_server().get_key_metrics(self.AGENT_METRICS)
        assert set(key) == {"mean/reward", "mean/input_tokens", "mean/output_tokens", "mean/total_tokens"}

    def test_absent_metrics_are_omitted_not_zero_filled(self) -> None:
        """Fragment configs produce no token-free run; a missing key must not appear as 0.0."""
        key = _make_server().get_key_metrics({"mean/reward": 0.5})
        assert key == {"mean/reward": 0.5}


class TestExtraction:
    @pytest.mark.parametrize(
        "text",
        [
            '{"answer": "CCO"}',
            '  {"answer": "CCO"}  ',
            '```json\n{"answer": "CCO"}\n```',
            '```\n{"answer": "CCO"}\n```',
            'Here is my answer: {"answer": "CCO"}',
            '{"answer": "CCO", "reasoning": "..."}',
        ],
    )
    def test_accepted_forms(self, text: str) -> None:
        assert _extract_json_answer(text) == "CCO"

    @pytest.mark.parametrize("text", ["", "no json here", "{}", '{"other": "CCO"}', '{"answer": ""}', '{"answer": 5}'])
    def test_rejected_forms(self, text: str) -> None:
        assert _extract_json_answer(text) is None


class TestMetrics:
    def test_multi_component_salt_canonicalises(self) -> None:
        """15 golds in task3_smiles_gen_single are dot-separated species, not SAFE fragments."""
        assert canonical_smiles("CCO.Cl") is not None

    def test_stereochemistry_can_be_ignored(self) -> None:
        strict = canonical_smiles("C[C@H](N)O")
        lenient = canonical_smiles("C[C@H](N)O", keep_stereo=False)
        assert strict != lenient
        assert exact_match("C[C@@H](N)O", "C[C@H](N)O") is False
        assert exact_match("C[C@@H](N)O", "C[C@H](N)O", keep_stereo=False) is True

    def test_unparseable_prediction_never_raises(self) -> None:
        assert is_valid("C1CC") is False
        assert exact_match("C1CC", GOLD) is False
        assert tanimoto("C1CC", GOLD, "morgan") == 0.0

    def test_levenshtein_counts_missing_prediction_as_full_distance(self) -> None:
        assert levenshtein(None, GOLD) == len(GOLD)
        assert levenshtein(GOLD, GOLD) == 0

    def test_unusable_prediction_reports_no_distance(self) -> None:
        """Charging full gold length inflated the SAFE mean to 10.07 vs upstream 6.70."""
        assert score_smiles("C1CC", GOLD)["levenshtein"] is None
        assert score_smiles(GOLD, GOLD)["levenshtein"] == 0.0

    def test_unknown_fingerprint_raises(self) -> None:
        with pytest.raises(ValueError):
            tanimoto(GOLD, GOLD, "nonexistent")
