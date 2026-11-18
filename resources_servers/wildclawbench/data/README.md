# WildClawBench example provenance

Five prepared inputs and corresponding real session rollouts: four healthy Kimi K3 tasks
(productivity, multimodal coding, social and safety) and one deliberately missing-judge-key
case. The masked case made no policy calls and must not be counted as a model zero. These
examples are not a full-suite score or leaderboard reproduction.

Prompts come from the MIT-licensed public source at the exact revision in
[`source-pin.json`](../../../benchmarks/wildclawbench/source-pin.json). See the
[license notice](../../../benchmarks/wildclawbench/third-party-notices.md). External
workspace assets, ground truth, images, captured model requests and credentials are not
redistributed here. Private artifact/capture paths and collector bookkeeping are omitted
from the exported rollout rows. Textual responses and rewards are preserved; inline binary
image attachments use explicit `example.invalid` placeholders. Original unmodified
responses and grading receipts are retained in the validation campaign.

The policy was `nvidia/moonshotai/kimi-k3` through NVIDIA's API, with high reasoning,
262144 context and 8192 output tokens; the native judge route was `openai/openai/gpt-5.4`.
Each healthy reward matched the original native score file. The lifecycle was
Resources seed, agent seed, OpenClaw activation, confirmed agent close, native verification,
and Resources cleanup. The example inputs can be rerun using the benchmark configuration
and provisioned assets; a healthy rerun of the fifth input needs the judge credential.
