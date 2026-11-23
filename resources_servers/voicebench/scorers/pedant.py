# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PEDANT answer-correctness scorer for SD-QA, loaded from the inference-only export in ``vendor/pedant``.

VoiceBench's SD-QA evaluator reports a ``panda`` metric from ``qa_metrics.pedant.PEDANT`` next to the judge
accuracy. The pickles that ``qa-metrics`` downloads need scikit-learn 1.3 and do not load on current releases, so
the TF-IDF vocabulary/IDF vector and the linear classifier are read from ``vendor/pedant`` instead (see its
README.md for the source revision and hashes). ``qa-metrics`` still provides the pinned PEDANT feature code.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np


PEDANT_DIR = Path(__file__).resolve().parents[1] / "vendor" / "pedant"


class PortableLinearClassifier:
    """Inference-only port of PEDANT's scikit-learn ``SGDClassifier`` (one coefficient row, two classes)."""

    def __init__(self, coef: np.ndarray, intercept: np.ndarray, classes: np.ndarray) -> None:
        self.coef = coef
        self.intercept = intercept
        self.classes = classes

    def decision_function(self, features: Any) -> np.ndarray:
        scores = np.asarray(features @ self.coef.T) + self.intercept
        return scores[:, 0]

    def predict(self, features: Any) -> np.ndarray:
        return self.classes[(self.decision_function(features) > 0.0).astype(int)]


def load_pedant(pedant_module: Any = None, pedant_dir: Path = PEDANT_DIR) -> Any:
    """Build a ``PEDANT`` instance backed by the vendored classifier, verified against ``parity.json``."""
    if pedant_module is None:
        import qa_metrics.pedant as pedant_module

    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer

    meta = json.loads((pedant_dir / "model.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((pedant_dir / "vocabulary.json").read_text(encoding="utf-8"))
    config = json.loads((pedant_dir / "vectorizer.json").read_text(encoding="utf-8"))
    config["ngram_range"] = tuple(config["ngram_range"])

    vectorizer = TfidfVectorizer(**config, vocabulary=vocabulary)
    vectorizer.fit(["portable pedant initialization"])
    vectorizer.idf_ = np.load(pedant_dir / "idf.npy", allow_pickle=False)
    model = PortableLinearClassifier(
        coef=np.load(pedant_dir / "coef.npy", allow_pickle=False),
        intercept=np.load(pedant_dir / "intercept.npy", allow_pickle=False),
        classes=np.load(pedant_dir / "classes.npy", allow_pickle=False),
    )
    if list(model.coef.shape) != meta["coef_shape"] or model.classes.tolist() != meta["classes"]:
        raise RuntimeError("Portable PEDANT weights do not match model.json")

    parity = json.loads((pedant_dir / "parity.json").read_text(encoding="utf-8"))
    features = hstack([vectorizer.transform(parity["texts"]), np.asarray(parity["extras"], dtype=np.float64)])
    if not np.allclose(model.decision_function(features), parity["decision"], rtol=0.0, atol=1e-12):
        raise RuntimeError("Portable PEDANT decision parity check failed")
    if model.predict(features).tolist() != parity["prediction"]:
        raise RuntimeError("Portable PEDANT prediction parity check failed")

    pedant = pedant_module.PEDANT.__new__(pedant_module.PEDANT)
    pedant.model = model
    pedant.tokenizer = vectorizer
    return pedant


def pedant_correct(pedant: Any, *, reference: str, answer: str, question: str) -> bool:
    """VoiceBench's call: lower-cased reference, answer and question."""
    return bool(pedant.evaluate([reference.lower()], answer.lower(), question.lower()))
