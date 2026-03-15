# chemistry_direct

Nemo-Gym RL environment for verifiable chemistry question answering
(direct generation variant).

## Task Description

The agent receives a natural-language chemistry question about a molecule,
paired with a SMILES string, and must respond with a **single number** (integer
or floating-point) or a **binary 0/1 flag**.  No tool calls; the format
instruction embedded in the prompt specifies the exact output format.

Questions are drawn from a stratified sample of the [ChEMBL database][chembl]
and cover RDKit-computable molecular properties:

| Property type | Examples | Format instruction |
|---|---|---|
| `float` | MolLogP, TPSA, MolWt | Single floating-point number |
| `count` | HeavyAtomCount, RingCount | Single integer |
| `bool` | PassLipinski | 0 or 1 |
| `presence` | HasAmide, HasAromatic | 0 or 1 |
| `fragment` | NumAmide, NumAromatic | Single integer |

## Reward Signal

| Property type | Reward |
|---|---|
| `float` | `–\|predicted – actual\|`  (negative absolute error; 0.0 = perfect) |
| `count` / `bool` / `presence` / `fragment` | 1.0 if exact match, else 0.0 |

When the model produces no parseable number, `reward = 0.0`.

For a mixed batch the `mean/reward` metric equals −MAE averaged over float
questions and accuracy averaged over discrete questions.

## Data Generation

Data is generated from the offline benchmark pipeline inside
`workdir/chemistry-benchmarking-fork/`:

```bash
# Generate stratified samples
make generate-experiment EXPERIMENT_ID=my_exp

# Export to Nemo-Gym JSONL format
make export-nemo-gym-data EXPERIMENT_ID=my_exp
# → nemo_gym_data/{train,validation,example}.jsonl

# Copy into this environment
cp nemo_gym_data/train.jsonl       path/to/nemo-gym/resources_servers/chemistry_direct/data/
cp nemo_gym_data/validation.jsonl  path/to/nemo-gym/resources_servers/chemistry_direct/data/
cp nemo_gym_data/example.jsonl     path/to/nemo-gym/resources_servers/chemistry_direct/data/
```

## Running

```bash
# Start all servers (resources + model)
config_paths="resources_servers/chemistry_direct/configs/chemistry_direct.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"

# Collect verified rollouts
ng_collect_rollouts \
    +agent_name=chemistry_direct_simple_agent \
    +input_jsonl_fpath=resources_servers/chemistry_direct/data/train.jsonl \
    +output_jsonl_fpath=rollouts_out.jsonl
```

## JSONL Format

Each row:

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "<natural-language question>\n\n<format instruction>"}]
  },
  "expected_answer": 2.3,
  "property_type": "float",
  "property": "MolLogP",
  "chembl_id": "CHEMBL25",
  "smiles": "CC(=O)Oc1ccccc1C(=O)O"
}
```

## Data License

Questions are derived from ChEMBL ([CC-BY-SA 3.0][cc]).

[chembl]: https://www.ebi.ac.uk/chembl/
[cc]: https://creativecommons.org/licenses/by-sa/3.0/
