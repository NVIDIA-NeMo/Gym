# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade ether0's three oracle-backed rewards in this process instead of over HTTP.

``sol_eval`` and ``rxn_forward`` call ``fetch_solubility``, ``fetch_purchasable`` and
``fetch_forward_rxn``, which upstream sends to a separate ``ether0-serve`` service. That service
only wraps three models: ``molsol.KDESol`` (solubility), a ZINC20 ``molbloom`` filter
(purchasability) and the MolTrans forward-reaction model. ``install`` loads the same three models
here, through upstream's own ``ether0.server`` classes, and rebinds the three functions on
``ether0.rewards``. They return what ``ether0.clients`` returns for the same server reply.

Two files are needed:

- the MolTrans checkpoint ``USPTO480k_model_step_400000.pt`` (linked from upstream's
  ``packages/remotes/README.md``), from ``moltrans_model_path`` or upstream's
  ``ETHER0_REMOTES_MOLTRANS_MODEL_PATH``, defaulting to the nemo-gym/ether0 cache.
  Missing checkpoints are downloaded and checksum-verified before loading;
- the ZINC20 filter ``zinc20.bloom`` (the ``zinc20`` catalog of ``molbloom`` 2.3.4), from
  ``zinc20_bloom_path``. Without a path ``molbloom`` uses ``~/.cache/molbloom``, downloading
  the 2 GB file there on first use.

``install`` fails when a download fails, an explicit bloom file is missing, the checkpoint is not pinned, or a
model does not answer a control query, so a run cannot start with an oracle that would score
every answer zero.
"""

import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

from resources_servers.ether0.setup_ether0 import ETHER0_REMOTES_SRC_PATH
from resources_servers.ether0.setup_moltrans import MOLTRANS_PATH_ENV_VAR, ensure_moltrans


logger = logging.getLogger(__name__)

ORACLE_FUNCTIONS = ("fetch_solubility", "fetch_purchasable", "fetch_forward_rxn")
# The reward functions that call an oracle, by their name in a row's solution string.
ORACLE_EVAL_FUNCTIONS = frozenset({"sol_eval", "rxn_forward"})
REMOTE_ORACLES_ENV_VAR = "ETHER0_REMOTES_API_BASE_URL"

_MOLTRANS_CONTROL = ("CC(=O)Cl.CCO>>", "CCOC(C)=O")  # acetyl chloride + ethanol -> ethyl acetate

_solubility: Any = None
_moltrans: Any = None


class OracleError(RuntimeError):
    """An oracle model failed, so the row was not measured."""


def fetch_solubility(query_smiles: str) -> dict[str, Any]:
    if "." in query_smiles:
        return {"smiles": query_smiles, "error": "Only single molecules are supported"}
    try:
        prediction = _solubility.run(smiles=query_smiles)
    except Exception as error:
        raise OracleError(f"Solubility model failed on {query_smiles!r}: {error}") from error
    if prediction is False:
        return {"smiles": query_smiles, "error": "Solubility prediction failed."}
    return {"smiles": query_smiles, "solubility": prediction.tolist()[0]}


def fetch_purchasable(query_smiles_list: list[str] | str) -> dict[str, bool]:
    import molbloom

    smiles = [query_smiles_list] if isinstance(query_smiles_list, str) else query_smiles_list
    return {s: molbloom.buy(s, canonicalize=True, catalog="zinc20") for s in smiles}


def fetch_forward_rxn(query_rxn_smiles: str) -> dict[str, str]:
    from fastapi import HTTPException

    reaction = query_rxn_smiles.replace(" ", "")
    if reaction.count(">") != 2:
        return {"smiles": query_rxn_smiles, "error": f"Syntax error in the reaction SMILES: {reaction}"}
    try:
        # Upstream reports SMILES that RDKit cannot parse as an HTTP 400, which its client turns
        # into an error reply: the answer is wrong, the oracle did not fail.
        query = _moltrans.canonicalize_smiles(".".join(part for part in reaction.split(">")[:-1] if part))
    except HTTPException as error:
        return {"smiles": query_rxn_smiles, "error": str(error.detail)}
    try:
        product, _ = _moltrans.run(query)
    except Exception as error:
        raise OracleError(f"MolTrans failed on {query_rxn_smiles!r}: {error}") from error
    return {"smiles": query_rxn_smiles, "product": product}


def install(moltrans_model_path: str | None, zinc20_bloom_path: str | None) -> None:
    """Load the three oracle models, check each with a control query, and rebind them on ``ether0.rewards``."""
    global _solubility, _moltrans

    bloom = Path(zinc20_bloom_path) if zinc20_bloom_path else None
    if bloom is not None and not bloom.is_file():
        raise FileNotFoundError(f"ether0 oracle file not found: {bloom}")
    checkpoint = ensure_moltrans(moltrans_model_path)

    # TensorFlow (solubility) and torch (MolTrans) fail over CUDA libraries when they share a
    # process, and the node's GPUs belong to the model servers: keep this process on CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # OpenNMT-py 2.3.0 loads the checkpoint with a bare torch.load, which torch 2.6 restricts.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    # ether0.server reads the checkpoint location from this variable.
    os.environ[MOLTRANS_PATH_ENV_VAR] = str(checkpoint)
    if str(ETHER0_REMOTES_SRC_PATH) not in sys.path:
        sys.path.append(str(ETHER0_REMOTES_SRC_PATH))

    import molbloom
    from ether0 import rewards
    from ether0.data import mol_from_smiles
    from ether0.server import MolecularTransformer, Solubility

    # molbloom only reads zinc20 from ~/.cache/molbloom, downloading it when absent. Handing it
    # a staged filter keeps buy() unchanged and skips that.
    if bloom is not None:
        molbloom._filters["zinc20"] = molbloom.BloomFilter(str(bloom))
    else:
        molbloom.buy("C1=CC=CC=C1", catalog="zinc20")
    _solubility = Solubility()
    _moltrans = MolecularTransformer()

    solubility = fetch_solubility("CCO").get("solubility")
    if solubility is None or not math.isfinite(solubility):
        raise OracleError(f"Solubility control failed: CCO -> {solubility!r}")
    if molbloom.canon("c1ccccc1") not in molbloom._filters["zinc20"]:
        raise OracleError(f"The zinc20 filter ({bloom or 'molbloom cache'}) does not contain benzene.")
    reaction, expected = _MOLTRANS_CONTROL
    product = fetch_forward_rxn(reaction).get("product", "")
    fragments = [mol_from_smiles(fragment) for fragment in product.split(".") if fragment]
    target = mol_from_smiles(expected)
    if not any(fragment is not None and rewards.exact_mol_match(fragment, target) == 1.0 for fragment in fragments):
        raise OracleError(f"MolTrans control failed: {reaction} -> {product!r}, expected {expected}")

    for name in ORACLE_FUNCTIONS:
        setattr(rewards, name, globals()[name])
    logger.info("ether0 oracle rewards are graded in-process (%s, %s).", checkpoint, bloom or "molbloom cache")
