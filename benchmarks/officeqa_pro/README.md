# OfficeQA Pro

[OfficeQA Pro](https://github.com/databricks/officeqa) evaluates how well agents
find evidence and reason over documents. Its 133 questions require searching
697 U.S. Treasury Bulletins, interpreting tables and charts, and computing
answers from the retrieved evidence. Use it to compare models and agent
harnesses on document retrieval, parsing and numerical reasoning.

This integration gives an agent harness such as OpenCode or Claude Code access
to the PDF corpus and scores final answers with the OfficeQA reference scorer.

## Prepare

Accept the [dataset's access conditions](https://huggingface.co/datasets/databricks/officeqa)
and authenticate with Hugging Face (`HF_TOKEN` or `hf auth login`), then run:

```bash
gym eval prepare --benchmark officeqa_pro
gym env validate officeqa_pro
```

Preparation downloads the questions and PDFs. For offline use, add
`+officeqa_pro_source_csv=/path/to/officeqa_pro.csv` and
`+officeqa_pro_corpus_root=/path/to/treasury_bulletin_pdfs` to the prepare command.
Local copies must match the pinned release; preparation checks the full CSV and
corpus hashes and records the corpus digest in `data/runtime/corpus.sha256`.
Sandbox startup checks read-only access and a pinned filename-and-size manifest
without reading PDF contents. Deploy the hash-verified `document_store/` unchanged.

## Run

Build the document tools and OpenCode images from the repository root:

```bash
docker build -f resources_servers/officeqa/Containerfile -t officeqa-tools .
docker build -f resources_servers/officeqa/Containerfile.opencode \
  --build-arg DOCUMENT_TOOLS_IMAGE=officeqa-tools -t officeqa-opencode .
```

The build prints the SHA-256 of `/opt/cli/opencode`. Supply that
architecture-specific hash when running; the stock agent verifies and stages
this preinstalled executable without downloading it.
The x64 image uses OpenCode's baseline binary, which does not require AVX2.

Make the CLI image available to your OpenSandbox cluster. Copy
`data/runtime/document_store/` to a directory on its hosts and mount only that
PDF directory; the rest of the preparation bundle contains reference answers
and must remain on the driver.

With OpenSandbox credentials (`OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY`) set:

```bash
gym eval run --benchmark officeqa_pro \
  --model-url http://MODEL_HOST:8000/v1 --model MODEL --model-api-key KEY \
  +officeqa_document_image=REGISTRY/IMAGE@sha256:DIGEST \
  +officeqa_opencode_binary_sha256=BINARY_SHA256 \
  +officeqa_documents_host_root=/cluster/document-releases \
  +officeqa_document_prefix=RELEASE \
  +use_absolute_ip=true \
  '++policy_model.responses_api_models.vllm_model.sampling_overrides={temperature: 1.0, top_p: 0.95}'
```

The sampling values above are examples. Use the selected model's recommended
settings and report the effective values with scores. Both harnesses reject
request-level `--temperature` and `--top-p`; set sampling through the model
server's `sampling_overrides` instead.

The model server must be reachable from the sandbox and support image input
for rendered document pages. Use `--num-repeats 1` for a single-attempt smoke run.

For Claude Code, build `Containerfile.claude` with the same document base,
select `--config benchmarks/officeqa_pro/configs/claude.yaml` instead of
`--benchmark officeqa_pro`, and supply
`+officeqa_claude_binary_sha256=HASH` using the hash printed by the build.
The model endpoint must support native Responses requests with image-bearing
tool results.

The [paper](https://arxiv.org/abs/2603.08655) describes the benchmark and agent
prompt. This integration adapts that prompt to the local PDF directory. Its
[document tools](../../resources_servers/officeqa/toolset/README.md) use a smaller
package set than the paper's baseline environment, so results can differ.

The paper's agents have web access; by default this configuration permits only
model-API egress and disables web search and fetch tools. To enable these tools
and use the provider's network policy, add `+officeqa_network_access=inherit`
to either harness's evaluation command. The provider must permit outbound access.
The default, `model_only`, enforces the model-API allowlist.

Inspection of the pinned
question texts found 13/133 (9.8%) explicitly requesting supplementary data from
named sources, including BLS CPI (UID0005), Macrotrends exchange rates (UID0010),
and World Bank/IMF GDP (UID0223). Another 11 request auxiliary CPI, exchange-rate
or silver-price series without naming a provider. These are potential network
dependencies, not a measured accuracy loss or proof that the PDFs lack the
inputs. All 133 questions remain included; report the network mode with scores.

## Measurement protocol

The default headline metric is `pass@1[avg-of-5]/accuracy_no_net`. With
`officeqa_network_access=inherit`, it is `pass@1[avg-of-5]/accuracy`.
Both report percentage accuracy averaged across five attempts per question,
then across questions, using the same scorer and aggregation. Masked results with
unavailable references are excluded. The table describes the defaults; record
any overrides alongside reported results.

| Setting | Default |
| --- | --- |
| Dataset | [OfficeQA Pro revision 763a8366](https://huggingface.co/datasets/databricks/officeqa/commit/763a8366abf2a3605c381d53586d844dc60fa756) |
| Scorer | [OfficeQA reward.py revision 7b9a3c15](https://github.com/databricks/officeqa/blob/7b9a3c154ef9fb40215bb67934afc43e6799de16/reward.py), numeric tolerance 0 |
| Prompt | Paper's agent prompt, adapted to the PDF directory with web search omitted; content hash stored in each task |
| Harness | OpenCode 1.19.4; optional Claude Code 2.1.285 |
| Generation limits | OpenCode: 262,144-token context, 16,384-token output; Claude Code: 256 turns |
| Time budgets | 7,200 seconds per agent session; 9,900 seconds per episode |
| Sandbox | 2 CPUs, 8 GiB memory, 10 GiB disk, no GPU |
| Network | Model API allowlist (`model_only`); web search and fetch tools disabled |

The prompt requests `<FINAL_ANSWER>` tags and says an absent tag fails. The
reference scorer also accepts a bare correct answer; this integration preserves
that behavior. Empty answers and incomplete responses score zero.

Verification also scores each answer at relative numeric tolerances of 0%,
0.1%, 1% and 5%, and records the results in `correct_tol_0`, `correct_tol_0p1`,
`correct_tol_1` and `correct_tol_5`. These scores use the same reference answers
and completion checks as the reward. They do not change the reward or the
reported accuracy. Their `pass@1[avg-of-N]` values show how many answers are
close to the reference but not exact.

`pass@1[avg-of-N]/no_answer` is the percentage of attempts that did not end in
a complete response with a nonempty `<FINAL_ANSWER>` tag. The verifier removes
reasoning before this check. An answer without the tag counts toward
`no_answer`, although the reward can still give it credit.

The verifier masks attempts whose reference answer is missing and sets their
tolerance fields to null. Rollouts saved before these fields existed do not
have them. Aggregation skips the missing fields, so reverify those rollouts to
get the tolerance and `no_answer` metrics.

## Example questions

Five example questions cover the June 2001 bulletin. They test retrieval and
arithmetic without the gated Pro question CSV. They use the same pinned PDF
corpus, prompt, document tools and OpenCode harness as the benchmark. The
examples form a separate taskset, `officeqa_pro_examples`, with one attempt per
question. Do not report their scores as OfficeQA Pro results.

```bash
gym eval prepare --config benchmarks/officeqa_pro/configs/examples.yaml
gym dataset collate --config benchmarks/officeqa_pro/configs/examples.yaml \
  --mode example_validation --output-dir benchmarks/officeqa_pro/data/example-collated
gym env start --config benchmarks/officeqa_pro/configs/examples.yaml \
  --model-url http://MODEL_HOST:8000/v1 --model MODEL --model-api-key KEY \
  +officeqa_document_image=REGISTRY/IMAGE@sha256:DIGEST \
  +officeqa_opencode_binary_sha256=BINARY_SHA256 \
  +officeqa_documents_host_root=/cluster/document-releases \
  +officeqa_document_prefix=RELEASE \
  +use_absolute_ip=true
```

Then run the collated examples on these servers:

```bash
gym eval run --no-serve \
  --input benchmarks/officeqa_pro/data/example-collated/example.jsonl \
  --output results/officeqa-pro-examples.jsonl --num-repeats 1
```

To prepare offline, set `+officeqa_pro_corpus_root=/path/to/treasury_bulletin_pdfs`
and leave `officeqa_pro_source_csv` unset. Preparation writes the reference
answers for the examples to `data/example-runtime/tasks/`, outside
`document_store/`. As with the full benchmark, deploy only the verified PDFs.

## Tests

```bash
pytest resources_servers/officeqa/tests benchmarks/officeqa_pro/tests
```

### Network isolation

Check the deployed image on real OpenSandbox before reporting `accuracy_no_net`.
Create separate `inherit` and `model_only` sandboxes with the same deployment
settings. Inside each sandbox, run:

```bash
curl --noproxy '*' --connect-timeout 5 --max-time 15 -sS \
  -o /dev/null -w '%{http_code}\n' https://example.com/
curl --noproxy '*' --connect-timeout 5 --max-time 15 -sS \
  -o /dev/null -w '%{http_code}\n' http://MODEL_HOST:PORT/v1/models
```

Use the allowlisted model endpoint's advertised address for the second command.
External HTTPS must succeed in `inherit` and fail in `model_only`; the model server must
remain reachable in both. Failure in both modes is inconclusive. Save the image
digest, network policies, commands, exit codes and stdout/stderr with the run
artifacts, then stop both sandboxes.

Also repeat the external request with `--resolve example.com:443:IP_FROM_INHERIT`,
using the address observed in the successful control. It must still fail in
`model_only` when DNS resolution is supplied explicitly.
