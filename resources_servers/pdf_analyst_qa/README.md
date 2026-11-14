# PDF Analyst QA

A training environment for answering questions about PDFs with OpenCode. The resources server provides the selected documents in a sandbox and grades the agent's final answer. The agent can read those PDFs and write scratch files; reference answers stay on the server.

## Dataset integration

The data pipeline supplies Gym JSONL rows, task records, document selections, and read-only PDFs. Requests must be ready for the OpenCode harness; conversion and generation metadata belong in the data pipeline.

```text
<dataset_root>/gym_input.jsonl
<dataset_root>/tasks/<task_id>.json
<dataset_root>/workspaces/<workspace_id>/documents.json
<dataset_root>/document_store/<document_id>.pdf
```

Each row identifies its task and workspace and contains the agent's input in `responses_create_params.input`. [task_data.py](task_data.py) defines request identity and reference records; [documents.py](documents.py) defines document selections. The [example dataset](data/example_bundle/) shows the file formats.

Supply a dataset YAML that includes the shared [environment config](../../environments/pdf_analyst_qa/config.yaml):

```yaml
config_paths:
- environments/pdf_analyst_qa/config.yaml

pdf_analyst_qa_dataset_root: /path/to/dataset
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

Keep the dataset outside the source tree. Copy its `document_store/` to the configured bucket prefix. The dataset root points to the corresponding local directory on the resources server.

The optional document cache speeds up sandbox startup when tasks select different document sets from a large shared corpus. It caches PDFs on each worker and uses hard links to assemble each task's read-only document directory, so later sandboxes on that worker can reuse documents without downloading or copying them again. Configure it through `pdf_analyst_qa_node_cache_host_path`.

Start Gym with `gym env start --config /path/to/dataset.yaml`. Configure the policy model through Gym's model server and keep training recipes in the training framework.

## Included example

[example.yaml](../../environments/pdf_analyst_qa/example.yaml) uses the manually authored example bundled with this repository. Copy it to the runtime location specified by that YAML:

```bash
python -m environments.pdf_analyst_qa.prepare_example --output-root /tmp/pdf-analyst-qa-example
```

The helper copies the files and makes the PDFs read-only. Publish the copied `document_store/` under the example's bucket prefix, then run `gym env start --config environments/pdf_analyst_qa/example.yaml`.

## Sandbox setup

Build the sandbox image using the OfficeQA [Containerfile](../officeqa/Containerfile) followed by [Containerfile.opencode](../officeqa/Containerfile.opencode). Set the image and OpenCode checksum variables referenced by the environment config, along with the OpenSandbox provider credentials.

Before running on a deployment, check sandbox access and model connectivity:

```bash
python -m environments.pdf_analyst_qa.check_sandbox \
  --image "$PDF_ANALYST_QA_DOCUMENT_IMAGE" \
  --model-url <model-server-url>
```

Add `--node-cache-host-path` if using a worker cache.

## Verification

Answers receive 1 when the answer value is correct and the requested format is satisfied, including units and citations. Other measured answers receive 0. The response reports value correctness as `semantic_correct` and format compliance as `format_compliant`. Infrastructure and verifier failures have `evaluation_completed=false` and `mask_sample=true`.

[answer_formats.py](verifiers/answer_formats.py) parses answers; [verifier.py](verifiers/verifier.py) grades them. Offline reverification is unsupported.

## Tests

```bash
pytest environments/pdf_analyst_qa/tests resources_servers/pdf_analyst_qa/tests
```

Live model rollouts and training runs still need validation before using this environment for training.
