# Review validation evidence

`review-smoke.json` records the source revisions, exact bounded sample scope, native check outcomes and live verification wall times for both harnesses. It omits credentials, deployment paths, private endpoints, raw browser profiles and screenshots. Native task/check labels derive from UniPat-AI/SaaS-Bench at commit48c22deb18b98c0ed78f81e7f3be82bc162de2c8 under CC-BY-4.0. Counts confirm retained screenshot evidence; images and full actor logs are deployment artifacts, not included here. The native checks show what was measured without exposing those artifacts.

The run validates the public pinned data manifest, preparation, app startup, isolated network/exec, model interaction, native live-state grading, and cleanup. The five-step cap intentionally does not establish task completion or a reference benchmark score. The official gym eval profile ran on one collected sample per harness; its mean/min/max/median describe only that five-step smoke subset. Verification timing is a single observation per harness, not a latency distribution or throughput result. Full reference-score parity and upstream all-server CI remain unqualified.

Local regressions cover confirmed model endings retaining partial grades, judge/provider faults, missing SCORE, stopped apps, bounded verification retries, no-generation recovery, stale bundle rejection, detached app cleanup, explicit run roots, concurrency defaults, manifest schema and container environment isolation.


## Session lifecycle smoke

[The session smoke receipt](session-smoke.json) records a hosted Kimi K3/high run through Factory, the SaaS Environment Server and each native harness. Job 2636414 completed on one CPU node. Each case used media_027 with a five-step budget; both returned a valid, unmasked native partial score of 0.143. OpenCode retained three screenshots with five tool calls and zero tool errors; browser-use retained two screenshots with six tool calls and zero tool errors. This validates execution and scoring, not full task completion or reference benchmark performance.

The serving/worker sources and default benchmark configs match the receipt's functional revision. The subsequent static session-type registry registration is covered by 60 configuration-composition tests. The official compliance inspector reports zero findings. Static admission checks cover all 106 benchmark tasks under both harnesses: 212 context packets, maximum 12003 bytes, within Gym's 64 KiB context limit; this is not 212 model rollouts.
