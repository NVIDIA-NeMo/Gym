# Description

## Relevant links
1. Publicly reported scores source: https://rank.opencompass.org.cn/leaderboard-multimodal
   1. Benchmark mapping https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb
2. Model config: https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/config.py#L210
3. Dataset build call: https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/run.py#L294
4. Run inference call: https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/inference.py#L133
5. Judge model configs: https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/run.py#L363
6. Actual requests.post call to OpenAI endpoint: https://github.com/open-compass/VLMEvalKit/blob/00804217f868058f871f5ff252a7b9623c3475d9/vlmeval/api/gpt.py#L234

## Accuracy reproduction using gpt-4o-mini-2024-07-18
|Benchmark|Name for run.py|Judge|Num samples|Reported score|Original repo repro|Gym repro|
|---|---|---|---|---|---|---|
|MMBench V1.1|MMBench_DEV_EN_V11|N/A?|4876|76 (test)|75.8 (dev)|62.6 (dev)|
|MMStar|MMStar|N/A|TODO|54.8|TODO|TODO|
|MMMU|MMMU_DEV_VAL|N/A|TODO|60|TODO|TODO|
|MathVista|MathVista_MINI|gpt-4o-mini|TODO|52.5|TODO|TODO|
|HallusionBench Avg.|HallusionBench|N/A|TODO|46.1|TODO|TODO|
|AI2D|AI2D_TEST,AI2D_TEST_NO_MASK|N/A|TODO|77.8|TODO|TODO|
|OCRBench|OCRBench|N/A|1000|785|776|772|
|MMVet|MMVet|gpt-4-turbo|TODO|66.9|TODO|TODO|


## Installation details
Rather than using decord, we use decord2 which is compatible with MacOS.

## Original repo repro
```bash
git clone https://github.com/open-compass/VLMEvalKit
cd VLMEvalKit

uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install '-e .' rouge

# Modify requirements.txt to use decord2 rather than decord
sed -i '' 's/decord>=0.6.0/decord2>=3.0.0/' requirements.txt

# For some reason, clip cannot be properly imported (import error on from pkg_resources import packaging)
sed -i '' 's/import clip/# import clip/' vlmeval/dataset/utils/SArena/FID.py

# Set your OpenAI API key
echo "OPENAI_API_KEY=..." > .env

python run.py --verbose \
    --data OCRBench \
    --model GPT4o_MINI
```

## Gym repro

### Prepare data
First run the VLMEvalKit server to install dependencies.
```bash
config_paths="resources_servers/vlm_eval_kit/configs/vlm_eval_kit.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

Then cd into this directory and activate the Python environment
```bash
cd resources_servers/vlm_eval_kit
source .venv/bin/activate
```

Uncomment the dataset you are preparing at the bottom of the script. Then run
```bash
python prepare_data.py
```


### Run with Gym
```bash
WANDB_PROJECT=
EXPERIMENT_NAME=vlmevalkit/gpt-4o-mini-20240718
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/vlm_eval_kit/configs/vlm_eval_kit.yaml"
ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    +wandb_project=$WANDB_PROJECT \
    +wandb_name=$EXPERIMENT_NAME \
    ++output_jsonl_fpath=results/$EXPERIMENT_NAME.jsonl \
    ++split=validation \
    ++policy_model_name=gpt-4o-mini-2024-07-18
```

# Licensing information
Code: ?
Data: ?

Dependencies
- nemo_gym: Apache 2.0
?
