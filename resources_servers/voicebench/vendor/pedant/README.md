# PEDANT classifier export

SD-QA's `panda` metric in [VoiceBench](https://github.com/MatthewCYM/VoiceBench) is `qa_metrics.pedant.PEDANT`
from [`qa-metrics`](https://github.com/zli12321/qa_metrics) (MIT, see [LICENSE](LICENSE), which is byte-identical to the upstream file at that commit, including its unfilled `[2024] [Zongxia Li]` brackets). Its classifier and
TF-IDF vectorizer are scikit-learn 1.3 pickles that do not load on current scikit-learn and NumPy, so this
directory holds an inference-only export of them: the classifier's coefficients, intercept and classes, and the
vectorizer's vocabulary and IDF vector. It was exported from these files at qa-metrics commit
`26623c6ca3313e7a58c48fd6ce2b8579eb0c742e`:

| Source file | SHA-256 |
|---|---|
| `qa_metrics/classifier/lr_classifier` | `9c3dd5998745ece1cfb6cb16b953358f794e8716d63a08f3b212d8bc32a59b8b` |
| `qa_metrics/classifier/tf-idf_vectorizer` | `e886c9ba3ddb15a4b693b075dfe19cfad36dc74e7f8f195b8d9e98e989f2ba4f` |

The model has one `(1, 94056)` coefficient row, classes `correct` and `incorrect`, and a 94,053-word unigram
vocabulary; the last three features are PEDANT's F1, precision and recall, and a decision above 0 predicts
`incorrect`. `scorers/pedant.py` rebuilds the vectorizer, checks its decision values and predictions against the
fixed fixture `parity.json` on every load, and attaches both to a `qa_metrics` `PEDANT` object, so the feature code
stays the pinned `qa-metrics==0.2.17`.
