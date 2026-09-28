# FACTS Search metrics

The primary metric is the public FACTS F1: the harmonic mean of overall accuracy and attempted accuracy. A grader label of A is correct, B is incorrect, and C is not attempted. Hedging rate is the C fraction; attempted accuracy is A divided by A plus B. Average searches is the mean number of search-enabled hops, matching the linked public implementation’s `n_hops` counter (parallel queries inside one turn are one hop). Query count is also reported separately.

The adapter reports a 95% bootstrap interval using 1,000 full-size resamples and seed 0. It also reports judge-valid rate, empty/truncated response rates, forced-final rate, and the exact counts behind every rate. The official notebook parser uses the first uppercase A/B/C and falls back to C; this remains the leaderboard score path, while a strict parser independently flags malformed judge replies. Judge transport failures surface to Gym's failure sidecar instead of being mislabeled as policy hedging.
