# XlamFc Resources Server

Function calling using the [Salesforce xlam-function-calling-60k dataset](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k).


```bash
huggingface-cli login
python resources_servers/xlam_fc/generate_dataset.py
```

```bash
gym env start \
    --model-type vllm_model \
    --resources-server xlam_fc
```

```bash
gym eval run --no-serve \
    --agent xlam_fc_simple_agent \
    --input resources_servers/xlam_fc/data/train.jsonl \
    --output results/xlam_fc_trajectory_collection.jsonl \
    --limit 10
```

## Scoring

Each row's `expected_answers` lists the function calls the model should make in reply to the user. `/verify` collects
every `function_call` item in the response and gives reward 1.0 only when each expected call is matched by name and
by the expected arguments and no other call was made; the response reports `num_expected`, `num_predicted` and
`num_correct`. Calls made after a tool output count like any other call.

The dataset's tools are not executed. Any `POST /<tool_name>` on the resources server returns a short JSON
acknowledgement (`{"status": "ok", ...}`), so the model sees no error for a tool it was told to use.

The tool schemas that `generate_dataset.py` writes follow the xlam type annotations (`int, optional` → `integer`,
`List[int]` → `array` of `integer`, `Tuple[float, float]` → `array` of `number`, `Dict[...]` → `object`), so the
declared type agrees with the native JSON values in `expected_answers`. An annotation that cannot be typed
(`Callable[...]`, `Union[int, str]`) declares no type. The registry artifact the config pins
(`xlam_function_calling_60k` version `0.0.1`, fetched by `gym dataset collate --download`) was built by the previous
generator and still declares `string` for every compound annotation; its republish is pending. Until then, re-run
the generator to get `train.jsonl` and `valid.jsonl` with the typed schemas.

## Licensing
Code: Apache 2.0
Dataset: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
