# PEDANT classifier export

SD-QA's `panda` metric in [VoiceBench](https://github.com/MatthewCYM/VoiceBench) is `qa_metrics.pedant.PEDANT`
from [`qa-metrics`](https://github.com/zli12321/qa_metrics) (MIT, see [LICENSE](LICENSE), which is byte-identical to the upstream file at that commit, including its unfilled `[2024] [Zongxia Li]` brackets). Its classifier and
TF-IDF vectorizer are scikit-learn 1.3 pickles that do not load on current scikit-learn and NumPy, so this
directory holds an inference-only export of them: the classifier's coefficients, intercept and classes, and the
vectorizer's vocabulary and IDF vector. It was exported from these files at qa-metrics commit
`26623c6ca3313e7a58c48fd6ce2b8579eb0c742e` (2025-07-18):

| Source file | Git blob | SHA-256 |
|---|---|---|
| `qa_metrics/classifier/lr_classifier` | `ed2403c19941e42c1d6f335c4ba7d1ab3dc3971b` | `9c3dd5998745ece1cfb6cb16b953358f794e8716d63a08f3b212d8bc32a59b8b` |
| `qa_metrics/classifier/tf-idf_vectorizer` | `fd1bf7d90eb5ad6155d43c72bde9cab5b35e2bd2` | `e886c9ba3ddb15a4b693b075dfe19cfad36dc74e7f8f195b8d9e98e989f2ba4f` |

The model has one `(1, 94056)` coefficient row, classes `correct` and `incorrect`, and a 94,053-word unigram
vocabulary; the last three features are PEDANT's F1, precision and recall, and a decision above 0 predicts
`incorrect`. `scorers/pedant.py` rebuilds the vectorizer, checks its decision values and predictions against the
fixed fixture `parity.json` on every load, and attaches both to a `qa_metrics` `PEDANT` object, so the feature code
stays the pinned `qa-metrics==0.2.17`.

## Which classifier this is

`qa-metrics` 0.2.17 downloads the classifier from the upstream `master` branch when `PEDANT` is constructed, and
upstream has replaced `qa_metrics/classifier/lr_classifier` (and the vectorizer) many times since 2024-02. The three
most recent models:

| Date | Commit | `lr_classifier` blob | Size |
|---|---|---|---|
| 2024-04-23 | `86c01fc` | `8e831bccb9a75ab43652b224cdfa4e2a35dccee3` | 406,208 B |
| 2024-10-28 | `930750d` | `d3b962ffee7067394f89730be2ba9a107ec80e51` | 557,808 B |
| **2024-10-31** | `1c19479` | **`ed2403c19941e42c1d6f335c4ba7d1ab3dc3971b`** (this export) | 753,600 B |

This export is the newest model, in place since 2024-10-31 and still current at the revision above. A `panda`
value is comparable to scores computed with `qa-metrics` after that date. `panda` values from before 2024-10-28
(which may include numbers published with the VoiceBench paper) can come from the earlier models, so do not compare
against them without re-scoring.

## How `parity.json` was checked

`parity.json` holds five (answer, reference, question) texts with PEDANT's F1/precision/recall features, and the
decision values and predictions of the classifier. [check_upstream_parity.py](check_upstream_parity.py) recomputes
them from the upstream pickles above (SHA-256-checked, read with an unpickler that only reconstructs numpy, scipy
and scikit-learn classes). It was run under Python 3.11.15, `scikit-learn==1.3.2`, `numpy==1.26.4`, `scipy==1.11.4`
and `joblib==1.6.0`: the largest difference from the committed decision values is 0.0, and the five predictions are
equal. In the same environment the exported coefficients, intercept, classes, vocabulary, IDF vector and vectorizer
parameters compare equal to the upstream objects. The test suite cannot run the script, because the pickles do not
load on a current scikit-learn, so this check is manual.
