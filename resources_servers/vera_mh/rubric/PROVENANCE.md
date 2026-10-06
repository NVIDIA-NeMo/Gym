# VERA-MH rubric and prompt assets (vendored, pinned)

These files are copied unchanged from the official repository https://github.com/SpringCare/VERA-MH at commit
`2c9d1fcbb68e1a2df64171c18b3e4d4c18b2f89e` (2026-08-25; the v1.2 rubric first released in tag `v1.2.0`, commit
`55ab0e3d33c9f14acd1b2b0a42327818993e8975`, 2026-07-16). They are redistributed under the repository's license
(`LICENSE` in this directory, Spring Care, Inc.), which permits copying and distribution with the copyright notice,
permission notice, and disclaimer intact. `benchmarks/vera_mh/prepare.py` re-downloads each file from the pinned
commit and refuses to run if any SHA-256 below has drifted.

| File | Upstream path | SHA-256 |
|---|---|---|
| `rubric.tsv` | `data/rubric.tsv` | `5bc029ea47395c35d34dc28dc0f2eef907796e1bf0f9b78b5cf68d444809cdee` |
| `rubric_prompt_beginning.txt` | `data/rubric_prompt_beginning.txt` | `9c5d1d6099a89b4d66cabef8320bb455732e80940a2e40b4baebb1f37f1e31f0` |
| `question_prompt.txt` | `data/question_prompt.txt` | `6c7487face8e9e2e2348cce03afb713f8586b06f62627f1bab01a146a5b8e584` |
| `persona_prompt_template.txt` | `data/persona_prompt_template.txt` | `14ddd27dde3b15bd62db54a0b2e129bf040423b60c7fe3aff0fa420b71d3ca1f` |
| `persona_prompt_reminder.txt` | `data/persona_prompt_reminder.txt` | `9efd2a2bbfa1ede2bfb487d2f9a44b8900eabbd86a118a28d43d9e6ab54e194a` |
| `LICENSE` | `LICENSE` | `9c24119c9d3ce9c37395719b80051bfdc35e138c51f0cefceaeb8aca7f32dd51` |

The 100-row persona sheet (`data/personas.tsv`, SHA-256
`07f0aa92cde50469d18aff640ed03e0df93e8863aed75ea102e4ba97a124330b`) is not vendored; `prepare.py` downloads it from
the same commit into the gitignored benchmark data directory and materializes one task per persona and user simulator.
