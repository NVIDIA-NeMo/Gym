#!/bin/bash
#SBATCH --job-name=mini_swe_traj_collection
#SBATCH --account=llmservice_modelalignment_sft
#SBATCH --partition=interactive  
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=1
#SBATCH --output=mini_swe_traj_collection-%j.out
#SBATCH --error=mini_swe_traj_collection-%j.err

CONFIG_PATHS="resources_servers/mini_swe_resource/configs/mini_swe_resource.yaml,responses_api_models/openai_model/configs/openai_model.yaml"

srun --container-image=/lustre/fsw/portfolios/llmservice/users/sdevare/repos/containers/singularity.sqsh \
        --container-mounts="/lustre:/lustre,/home:/home,/lustre/fsw/portfolios/llmservice/users/sdevare/repos/nemo-gym/:/nemo-gym" \
        --container-workdir=/nemo-gym \
        /bin/bash -c "
        cd /lustre/fsw/portfolios/llmservice/users/sdevare/repos/nemo-gym/
        apt-get update && apt-get install -y git
        export SINGULARITY_CACHEDIR=/tmp/singularity_cache_${SLURM_JOB_ID}_$$
        export APPTAINER_CACHEDIR=/tmp/apptainer_cache_${SLURM_JOB_ID}_$$
        export SINGULARITY_TMPDIR=/tmp/singularity_tmp_${SLURM_JOB_ID}_$$
        export APPTAINER_TMPDIR=/tmp/apptainer_tmp_${SLURM_JOB_ID}_$$
        export TMPDIR=/tmp/

        mkdir -p $SINGULARITY_CACHEDIR $APPTAINER_CACHEDIR $SINGULARITY_TMPDIR $APPTAINER_TMPDIR $TMPDIR

        uv venv --allow-existing
        source .venv/bin/activate
        uv sync

        ng_download_dataset_from_gitlab \
            +dataset_name=mini_swe_agent \
            +version=0.0.1 \
            +artifact_fpath=train.jsonl \
            +output_fpath=data/train.jsonl

        ng_run --help

        ng_run +config_paths=[$CONFIG_PATHS] \
        '+mini_swe_main_agent.responses_api_agents.mini_swe_agent.cache_dir_template=/lustre/fsw/portfolios/llmservice/users/igitman/images/swe-bench/xingyaoww_sweb.eval.x86_64.\{instance_id\}.sif' \
        +mini_swe_main_agent.responses_api_agents.mini_swe_agent.skip_if_exists=True \
        +mini_swe_main_agent.responses_api_agents.mini_swe_agent.concurrency=16 \
        +mini_swe_main_agent.responses_api_agents.mini_swe_agent.step_timeout=300 \
        +mini_swe_main_agent.responses_api_agents.mini_swe_agent.eval_timeout=1800 &

        # Wait for a few seconds to ensure the service is up before starting trajectory collection
        sleep 30

        # Run trajectory collection
        ng_collect_rollouts +agent_name=mini_swe_main_agent \
            +input_jsonl_fpath=data/train.jsonl \
            +output_jsonl_fpath=results/mini_swe_agent.jsonl
        wait
        "

wait