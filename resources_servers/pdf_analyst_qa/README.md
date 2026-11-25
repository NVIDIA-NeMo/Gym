# PDF Analyst QA

A training environment for answering questions about PDFs with OpenCode. The resources server provides the selected documents in a sandbox and grades the agent's final answer. The agent can read those PDFs and write scratch files; reference answers stay on the server.

## Dataset integration

The data pipeline supplies Gym JSONL rows and PDFs. Requests must be ready for the OpenCode harness; conversion and generation metadata belong in the data pipeline.

Each row identifies its task and contains the agent's input in `responses_create_params.input`, the reference record used for grading, and the PDFs the agent may read with their checksums. [task_data.py](task_data.py) defines the row format. The [example dataset](data/example/) shows the file formats.

Supply a dataset YAML that includes the shared [environment config](../../environments/pdf_analyst_qa/config.yaml):

```yaml
config_paths:
- environments/pdf_analyst_qa/config.yaml

pdf_analyst_qa_document_prefix: my-dataset

pdf_analyst_qa_agent:
  responses_api_agents:
    opencode_agent:
      datasets:
      - name: my_dataset
        type: train
        jsonl_fpath: /path/to/dataset/gym_input.jsonl
        license: <dataset-license>
```

Keep the dataset outside the source tree. Copy its PDFs to the configured bucket prefix, so that the document paths in the rows resolve under it.

Start Gym with `gym env start --config /path/to/dataset.yaml`. Configure the policy model through Gym's model server and keep training recipes in the training framework.

## Included example

[example.yaml](../../environments/pdf_analyst_qa/example.yaml) uses the manually authored example bundled with this repository. Publish its [document_store/](data/example/document_store/) under the example's bucket prefix, then run `gym env start --config environments/pdf_analyst_qa/example.yaml`.

## Sandbox setup

Build the sandbox image from the repository root with the OfficeQA [Containerfile](../officeqa/Containerfile), [Containerfile.opencode](../officeqa/Containerfile.opencode), and the PDF Analyst QA [Containerfile](Containerfile). This environment needs OpenCode 1.17.11 and a root image user for document staging:

```bash
docker build -f resources_servers/officeqa/Containerfile -t officeqa-tools .
docker build -f resources_servers/officeqa/Containerfile.opencode \
  --build-arg DOCUMENT_TOOLS_IMAGE=officeqa-tools \
  --build-arg OPENCODE_VERSION=1.17.11 \
  --build-arg SANDBOX_USER=root \
  -t pdf-analyst-qa-opencode-base .
docker build -f resources_servers/pdf_analyst_qa/Containerfile \
  --build-arg OPENCODE_IMAGE=pdf-analyst-qa-opencode-base \
  -t pdf-analyst-qa-opencode .
```

The final build prints the SHA-256 of `/opt/cli/opencode` and `/opt/cli/opencode-1.17.11-plugin-dependencies.tar.gz`. Set `PDF_ANALYST_QA_DOCUMENT_IMAGE` to the final image, `PDF_ANALYST_QA_OPENCODE_SHA256` to the binary checksum, and `PDF_ANALYST_QA_OPENCODE_PLUGIN_SHA256` to the archive checksum, along with the OpenSandbox provider credentials. Build for the deployment architecture and use the checksums from that image; the archive includes platform-specific optional dependencies. Rebuild and update these values when changing the runtime.

The archive uses the pinned dependency set in Gym's visual-coding image. The harness verifies it and unpacks it into each session's private OpenCode configuration directory before launch, avoiding an npm install in the network-restricted sandbox.

OpenCode 1.17.11 supports the request `temperature` and `top_p` fields supplied by NeMo RL. Set NeMo RL's `max_new_tokens` to at least 8192, the environment's configured per-call `max_output_tokens`; smaller request limits are rejected by the harness.

Each sandbox holds a copy of its task's selected PDFs. `sandbox_disk_gib` sets the sandbox's ephemeral storage limit, so it must cover the dataset's largest selection plus the agent's working files. The default of 24 GiB fits selections up to about 17.5 GiB; raise it for datasets with larger selections.

Before running on a deployment, check sandbox access and model connectivity:

```bash
python -m environments.pdf_analyst_qa.check_sandbox \
  --image "$PDF_ANALYST_QA_DOCUMENT_IMAGE" \
  --model-url <model-server-url>
```

## Verification

Answers receive 1 when the answer value is correct and the requested format is satisfied, including units and citations. Other measured answers receive 0. The response reports value correctness as `semantic_correct` and format compliance as `format_compliant`. Failures that cannot be scored have `evaluation_completed=false` and `mask_sample=true`.

Classified agent failures that Gym's default outcome policy sends to verification receive an unmasked reward of 0. These include invalid agent output, agent limits, provider out-of-memory termination, and model request rejection with HTTP 400, 413, or 422. Other failed responses remain masked, including failures with missing or malformed outcome metadata. Scored failures count in accuracy and retain their reported failure kind.

Decimal-place instructions specify numerical rounding, rather than a fixed number of printed digits. Numerically equivalent answers may add or omit trailing fractional zeros: `1.6` and `1.60` are equivalent, as are `23` and `23.0`. This does not introduce a numerical tolerance: `1.604` does not match a reference value of `1.60`.

Dataset authors should state the required rounding and make trailing fractional zeros optional in task questions. Fixed-width decimal formatting is not enforced. Keep each row's question consistent with its trusted task record, and update `prompt_version` when revising published questions; existing external datasets are not rewritten by the environment.

[answer_formats.py](verifiers/answer_formats.py) parses answers; [verifier.py](verifiers/verifier.py) grades them. Offline reverification is unsupported.

## Tests

```bash
pytest environments/pdf_analyst_qa/tests resources_servers/pdf_analyst_qa/tests
```

Live model rollouts and training runs still need validation before using this environment for training.
