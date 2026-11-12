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
"""OCSR verifier: score a SMILES string read from a molecule image against the gold SMILES.

The policy is asked for ``{"smiles": "<SMILES>", "confidence": <0-1>}``. The answer is read
from the last such JSON object; if there is none, from the last RDKit-valid SMILES in a
backtick span or a loose ``"smiles"`` field (``salvaged``). Reasoning blocks are removed first.

Reward is ``exact``: RDKit canonical isomeric SMILES of prediction and gold are equal.
Diagnostics returned with every row: ``graph`` (equal without stereo), ``inchikey``
(standard InChIKey equal), ``tanimoto`` (Morgan r=2, 2048 bits), ``valid``, ``format_ok``.
"""

from pathlib import Path
from typing import Any, ClassVar, Optional

from pydantic import ConfigDict

from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
    observe_programmatic_grading,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture


try:  # imported as a package (tests) or as a script (server)
    from .chem import extract_answer, gold_is_usable, score
except ImportError:
    from chem import extract_answer, gold_is_usable, score


class OcsrResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class OcsrVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any]


class OcsrVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    extracted_smiles: Optional[str] = None
    format_ok: float = 0.0
    valid: float = 0.0
    exact: float = 0.0
    graph: float = 0.0
    inchikey: float = 0.0
    tanimoto: float = 0.0


def _final_text(body: BaseVerifyRequest) -> str:
    """Text of the assistant message(s); reasoning items are not part of the answer."""
    texts: list[str] = []
    for item in body.response.output:
        if getattr(item, "type", None) != "message" or getattr(item, "role", None) != "assistant":
            continue
        content = getattr(item, "content", None)
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(c.text for c in content if isinstance(getattr(c, "text", None), str))
    return "\n".join(texts)


class OcsrVerifier:
    async def verify(self, body: OcsrVerifyRequest) -> OcsrVerifyResponse:
        gold = body.verifier_metadata.get("gold_smiles")
        if not isinstance(gold, str) or not gold_is_usable(gold):
            # A row whose gold label RDKit cannot read measures nothing about the policy: masked, and no grade recorded.
            return OcsrVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                mask_sample=True,
                failure_kind="ocsr:bad_gold",
                failure_reason=f"gold SMILES is missing or not parseable by RDKit: {gold!r}",
            )
        # Grading is a pure RDKit calculation (no model call), recorded as programmatic grading.
        return observe_programmatic_grading(lambda: self._programmatic_score(body, gold))

    def _programmatic_score(self, body: OcsrVerifyRequest, gold: str) -> OcsrVerifyResponse:
        smiles, how = extract_answer(_final_text(body))
        s = score(smiles, gold)
        return OcsrVerifyResponse(
            **body.model_dump(),
            reward=s["exact"],
            extracted_smiles=smiles,
            format_ok=float(how == "json"),
            **{k: s[k] for k in ("valid", "exact", "graph", "inchikey", "tanimoto")},
        )


class OcsrResourcesServer(OcsrVerifier, SimpleResourcesServer):
    config: OcsrResourcesServerConfig

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        keep = (
            "mean/reward",
            "mean/exact",
            "mean/graph",
            "mean/inchikey",
            "mean/tanimoto",
            "mean/valid",
            "mean/format_ok",
            "mean/output_tokens",
        )
        return {k: agent_metrics[k] for k in keep if k in agent_metrics}


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=OcsrVerifier,
    request_model=OcsrVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    OcsrResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = OcsrResourcesServer.run_webserver()  # noqa: F401
