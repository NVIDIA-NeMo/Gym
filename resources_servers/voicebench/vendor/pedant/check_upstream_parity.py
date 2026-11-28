# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check ``parity.json`` against the upstream qa-metrics pickles (not run by the test suite).

The upstream classifier and vectorizer are scikit-learn 1.3 joblib pickles that need an old environment, for example
Python 3.11 with ``scikit-learn==1.3.2 numpy<2 scipy<1.12``. Download ``qa_metrics/classifier/lr_classifier`` and
``qa_metrics/classifier/tf-idf_vectorizer`` at the revision in README.md into one directory, then run::

    python check_upstream_parity.py <directory with the two pickles> [parity.json]

The files are third-party data, so they are checked against the SHA-256 values below and read with an unpickler that
only reconstructs the numpy, scipy and scikit-learn classes these two files contain. The script prints the largest
difference between the decision values it computes and those in ``parity.json`` and exits non-zero if the predictions
differ or the difference exceeds 1e-12.
"""

import hashlib
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from joblib.numpy_pickle import NumpyUnpickler
from scipy.sparse import hstack


SHA256 = {
    "lr_classifier": "9c3dd5998745ece1cfb6cb16b953358f794e8716d63a08f3b212d8bc32a59b8b",  # pragma: allowlist secret
    "tf-idf_vectorizer": "e886c9ba3ddb15a4b693b075dfe19cfad36dc74e7f8f195b8d9e98e989f2ba4f",  # pragma: allowlist secret
}
ALLOWED_GLOBALS = {
    ("joblib.numpy_pickle", "NumpyArrayWrapper"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy", "float64"),
    ("numpy", "int64"),
    ("numpy", "int32"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("collections", "OrderedDict"),
    ("scipy.sparse._csr", "csr_matrix"),
    ("scipy.sparse._csc", "csc_matrix"),
    ("sklearn.linear_model._stochastic_gradient", "SGDClassifier"),
    ("sklearn.linear_model._sgd_fast", "Log"),
    ("sklearn.linear_model._sgd_fast", "Hinge"),
    ("sklearn.linear_model._sgd_fast", "ModifiedHuber"),
    ("sklearn.linear_model._sgd_fast", "SquaredHinge"),
    ("sklearn.feature_extraction.text", "TfidfVectorizer"),
    ("sklearn.feature_extraction.text", "TfidfTransformer"),
    ("sklearn.feature_extraction.text", "CountVectorizer"),
}


class _RestrictedUnpickler(NumpyUnpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


def _load(directory: Path, name: str):
    data = (directory / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256[name]:
        raise SystemExit(f"{name} does not match the SHA-256 recorded in README.md")
    return _RestrictedUnpickler(name, io.BytesIO(data), True).load()


def main() -> None:
    directory = Path(sys.argv[1])
    fixture = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("parity.json")
    model, vectorizer = _load(directory, "lr_classifier"), _load(directory, "tf-idf_vectorizer")
    parity = json.loads(fixture.read_text(encoding="utf-8"))
    features = hstack([vectorizer.transform(parity["texts"]), np.asarray(parity["extras"], dtype=np.float64)])
    difference = float(np.max(np.abs(model.decision_function(features) - np.asarray(parity["decision"]))))
    same_predictions = model.predict(features).tolist() == parity["prediction"]
    print(f"max |decision difference| = {difference:.3e}; predictions equal: {same_predictions}")
    if not same_predictions or difference > 1e-12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
