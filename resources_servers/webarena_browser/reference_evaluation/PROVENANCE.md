# WebArena evaluator provenance

The evaluator modules in this package are derived from NVIDIA's
`osworld_internal` `nemotron-v3` branch at commit
`3b775dc538931ead0cb6b4922349da9c6d493dab`.

Pinned source identities:

| Source module | SHA-256 | Top-level definitions |
|---|---|---:|
| `webarena/common/classic_evaluation.py` | `123d11370563cedea1ccf01a1838540be7ed5ea1aef26ff7179a8e913311e976` | 50 |
| `webarena/common/visualwebarena_evaluation.py` | `dbe24e4cb4e13ef1966f3ec377bfba74b0926b77d7ccecf7f2d5f01747369d15` | 76 |
| `webarena/common/eval_snapshots.py` | `36b3a6ffdf82aabf9e22d102df33ef8368fc8466d1789a3105710dbcb3326bb4` | 28 |
| `webarena/common/eval_collision.py` | `b6d2834c7d196f4eed7263b400598136511008ff55ececa726bd9fbc0bc37b3f` | 14 |

The scoring algorithms remain benchmark-specific. Gym supplies only the
surrounding lifecycle and replaces two infrastructure imports:

- collision planning lives in `nemo_gym.web.evaluation_collision`, where
  dataset preparation and the runtime can share it;
- local WebArena-family evaluation navigates directly with Playwright instead
  of importing WebVoyager's Cloudflare/CAPTCHA handler.

The Gym lifecycle preserves both snapshot sources used by the pinned
runner: synchronous site/API snapshots and live-browser snapshots are merged
at task start and task end before deltas are built. The source modules were
compared structurally at the top-level function/class-body boundary; changes
are confined to imports, local infrastructure boundaries, formatting, and the
documented wrappers above. All 168 pinned definitions were present with zero
missing, extra, or changed AST bodies in the final audit.

Public benchmark credentials are resolved through the WebArena site's
environment-aware login boundary. No model checkpoint, deployment secret, or
site state is vendored here.

## Source URL, license and original notices

Source repository and original URL: see the [immutable public import-provenance record](https://github.com/NVIDIA-NeMo/Gym/blob/d46f2f60d2f6eda11e94deb48e6a16b21b5389fc/resources_servers/webarena_browser/reference_evaluation/PROVENANCE.md#source-url-license-and-original-notices). Original revision: `3b775dc538931ead0cb6b4922349da9c6d493dab` (`nemotron-v3`). This permalink identifies the published provenance record, not a new upstream source revision.

The root LICENSE at that exact original revision is Apache-2.0 and contains `Copyright 2024 XLANG NLP Lab`; its original locator and verification are retained in the [same immutable public record](https://github.com/NVIDIA-NeMo/Gym/blob/d46f2f60d2f6eda11e94deb48e6a16b21b5389fc/resources_servers/webarena_browser/reference_evaluation/PROVENANCE.md#source-url-license-and-original-notices). The complete original text is retained in `SOURCE_LICENSE`. The pinned tree has no additional LICENSE, COPYING or NOTICE files, and the three original module headers contain no separate copyright notices.

| Original file | Gym file |
| --- | --- |
| `webarena/common/classic_evaluation.py` | `resources_servers/webarena_browser/reference_evaluation/classic_evaluation.py` |
| `webarena/common/eval_snapshots.py` | `resources_servers/webarena_browser/reference_evaluation/eval_snapshots.py` |
| `webarena/common/eval_collision.py` | `nemo_gym/web/evaluation_collision.py` |

The 92-definition AST comparison above describes the initial port only. Subsequent Gym changes include infrastructure error classification, snapshot validation, per-tab failure propagation and deployment-normalized keys; current code is not claimed to have unchanged AST bodies.
