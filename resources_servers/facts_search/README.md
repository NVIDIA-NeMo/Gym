# FACTS Search resources server

The server exposes exactly one model-visible tool, `brave_search`, backed by Brave Web Search. It requests five results and returns each result’s title, URL, description, and complete `extra_snippets`, matching Google’s public implementation. It also owns the current Gemini 3.5 Flash A/B/C verifier and public aggregate metrics.

No alternate search provider is configured. Missing Brave or judge credentials fail startup/evaluation instead of changing the benchmark.

The verifier extracts the final answer from the last assistant message and inserts the question, public gold answer, and prediction into the byte-pinned public grader prompt. The public parser uses the first uppercase A/B/C and falls back to C; an independent strict parser records malformed judge replies without changing the leaderboard score path. Judge transport failures go to Gym's failure sidecar.

Key outputs are F1, overall accuracy, attempted accuracy, not-attempted rate, judge-valid rate, search hops, query count, forced-final rate, and empty/truncated answer rates. `task_data.py` documents the four public row fields consumed by the verifier.
