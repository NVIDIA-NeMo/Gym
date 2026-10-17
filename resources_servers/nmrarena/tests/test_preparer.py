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

"""The preparer with the download stubbed: digest gate, corpus checks, row shape, argument boundaries."""

import hashlib
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_nmrarena as prep  # noqa: E402


def _record(i: int, cls: str, smiles: str = "CCO") -> dict:
    return {
        "compound_id": f"id-{cls}-{i}",
        "publication_id": f"pub-{i}",
        "smiles": smiles,
        "n_complex": 0.1 * i,
        "h_nmr": "H_NMR (400 MHz, CDCl_3) δ 3.69 (q, J = 7.0 Hz, 2H), 1.22 (t, J = 7.0 Hz, 3H)",
        "c_nmr": "C_NMR  (100 MHz, DMSO_d6) δ 58.3, 18.4",
        "doi": "10.0000/synthetic",
    }


def _corpus(n_classes=21, per_class=5) -> dict:
    return {f"cls_{c}": {f"{i}.0": _record(i, f"cls_{c}") for i in range(1, per_class + 1)} for c in range(n_classes)}


@pytest.fixture
def pinned(monkeypatch):
    """Serve ``corpus`` as the pinned file, with the digest gate pointed at it."""

    def install(corpus: dict, digest_matches: bool = True):
        raw = json.dumps(corpus, ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest() if digest_matches else "0" * 64
        monkeypatch.setattr(prep, "DATASET_SHA256", digest)
        fetched = []

        def http_get(url: str) -> bytes:
            fetched.append(url)
            assert url == prep.DATASET_URL
            return raw

        return http_get, fetched

    return install


def _run(http_get, tmp_path, *args) -> list[dict]:
    out = tmp_path / "out.jsonl"
    prep.main(["--output", str(out), *args], http_get=http_get)
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def test_rows_carry_upstream_request_decoding_and_truth(pinned, tmp_path) -> None:
    http_get, fetched = pinned(_corpus())
    rows = _run(http_get, tmp_path)
    assert len(rows) == 105 and fetched == [prep.DATASET_URL]
    first = rows[0]
    params = first["responses_create_params"]
    assert (params["temperature"], params["max_output_tokens"]) == (1.0, 24576)
    assert "top_p" not in params and "seed" not in params
    system, user = params["input"]
    assert system["role"] == "system" and system["content"].startswith("You are a senior organic chemist")
    assert "Provide up to 10 structures and fill at least 10 slots." in system["content"]
    assert user["content"] == (
        "Determine the structure of a single organic molecule from its NMR data.\n"
        "1H NMR: (400 MHz, CDCl3) δ 3.69 (q, J = 7.0 Hz, 2H), 1.22 (t, J = 7.0 Hz, 3H)\n"
        "13C NMR: (100 MHz, DMSO-d6) δ 58.3, 18.4\n"
        "\nPropose up to 10 candidate structures, ranked best-first, in the required JSON format."
    )
    meta = first["verifier_metadata"]
    assert meta["smiles"] == "CCO" and meta["primary_class"] == "cls_0" and meta["compound_id"] == "id-cls_0-1"
    assert meta["dataset_commit"] == prep.GITHUB_COMMIT and meta["dataset_sha256"] == prep.DATASET_SHA256
    assert first["agent_ref"] == {"type": "responses_api_agents", "name": "nmrarena_simple_agent"}


def test_num_candidates_reaches_both_turns(pinned, tmp_path) -> None:
    http_get, _ = pinned(_corpus())
    rows = _run(http_get, tmp_path, "--num-candidates", "3")
    system, user = rows[0]["responses_create_params"]["input"]
    assert "Provide up to 3 structures and fill at least 3 slots." in system["content"]
    assert "Propose up to 3 candidate structures" in user["content"]


def test_digest_mismatch_fails_closed_before_parsing(pinned, tmp_path) -> None:
    http_get, _ = pinned(_corpus(), digest_matches=False)
    with pytest.raises(prep.CorpusError, match="sha256"):
        prep.main(["--output", str(tmp_path / "o.jsonl")], http_get=http_get)
    assert not (tmp_path / "o.jsonl").exists()


@pytest.mark.parametrize(
    ("corpus", "message"),
    [
        (_corpus(per_class=4), "loaded 84"),
        (_corpus(n_classes=15, per_class=7), "found 15 classes"),
    ],
)
def test_wrong_shape_fails_closed(pinned, tmp_path, corpus, message) -> None:
    http_get, _ = pinned(corpus)
    with pytest.raises(prep.CorpusError, match=message):
        prep.main(["--output", str(tmp_path / "o.jsonl")], http_get=http_get)
    assert not (tmp_path / "o.jsonl").exists()


def test_uneven_classes_with_the_right_total_fail_closed(pinned, tmp_path) -> None:
    corpus = _corpus()
    moved = corpus["cls_0"].pop("5.0")
    corpus["cls_1"]["6.0"] = dict(moved, compound_id="moved")
    http_get, _ = pinned(corpus)
    with pytest.raises(prep.CorpusError, match="cls_0 has 4"):
        prep.main(["--output", str(tmp_path / "o.jsonl")], http_get=http_get)


def test_duplicate_id_or_unparseable_gold_fail_closed(pinned, tmp_path) -> None:
    corpus = _corpus()
    corpus["cls_0"]["2.0"]["compound_id"] = corpus["cls_0"]["1.0"]["compound_id"]
    http_get, _ = pinned(corpus)
    with pytest.raises(prep.CorpusError, match="unique"):
        prep.main(["--output", str(tmp_path / "o.jsonl")], http_get=http_get)

    corpus = _corpus()
    corpus["cls_3"]["3.0"]["smiles"] = "C(C)(C)(C)(C)C"
    http_get, _ = pinned(corpus)
    with pytest.raises(prep.CorpusError, match="gold smiles does not parse"):
        prep.main(["--output", str(tmp_path / "o.jsonl")], http_get=http_get)


def test_missing_spectrum_fails_closed(pinned, tmp_path) -> None:
    corpus = _corpus()
    corpus["cls_2"]["1.0"]["c_nmr"] = "  "
    http_get, _ = pinned(corpus)
    with pytest.raises(prep.CorpusError, match="no c_nmr"):
        prep.main(["--output", str(tmp_path / "o.jsonl")], http_get=http_get)


def test_limit_truncates_after_validation(pinned, tmp_path) -> None:
    http_get, _ = pinned(_corpus())
    assert len(_run(http_get, tmp_path, "--limit", "7")) == 7
    http_get, _ = pinned(_corpus(per_class=4))
    with pytest.raises(prep.CorpusError):
        prep.main(["--output", str(tmp_path / "o.jsonl"), "--limit", "1"], http_get=http_get)


@pytest.mark.parametrize("flag", ["--limit", "--num-candidates"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_counts_are_rejected_before_any_fetch(tmp_path, flag, value) -> None:
    def never(url: str) -> bytes:
        raise AssertionError("no fetch should happen when arguments are invalid")

    with pytest.raises(SystemExit):
        prep.main(["--output", str(tmp_path / "o.jsonl"), flag, value], http_get=never)
