# export $SCRATCH_PATH=/lustre/fsw/portfolios/llmservice/users/rgala
# export DISTCP_FILE_PATH=/lustre/fsw/portfolios/llmservice/users/rgala/repos/RL/results/v3_base_qwen_8b_airline_verifier_qwen235b_mixed_data_base_model_qwen3-8b_sft_lr5e-6/step_45
# export MODEL_NAME=v3_base_qwen_8b_airline_chkpt_45
# export HF_MODEL_PATH=$DISTCP_FILE_PATH/hf
# bash /lustre/fsw/portfolios/llmservice/users/rgala/repos/RL/ritu_temp/convert_megatron_to_hf.sh $DISTCP_FILE_PATH

# export DISTCP_FILE_PATH=/lustre/fsw/portfolios/llmservice/users/rgala/repos/RL/results/qwen_8b_thinking_airline_verifier_qwen235b_mixed_data_sft_lr5e-6/step_45
# export MODEL_NAME=qwen_8b_thinking_airline_verifier_baseline
# export HF_MODEL_PATH=$DISTCP_FILE_PATH/hf

# export DISTCP_FILE_PATH=Qwen/Qwen3-8B
# export MODEL_NAME="Qwen3-4B-Instruct-2507"
# export HF_MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

export MODEL_NAME=qwen3-4b-Instruct
export HF_MODEL_PATH=/lustre/fsw/portfolios/llmservice/users/abhibhag/nemo-rl/results/20250924/workbench/qwen3_4binstruct/workbench-test/step_57-hf


export VLLM_PORT=10241

export TOOL_PARSER="hermes"
cd  /lustre/fsw/portfolios/llmservice/users/abhibhag/nemo-rl/3rdparty/Penguin-workspace/Penguin
cmd_start_server="bash resources_servers/workbench/scripts/tau2bench/trained_model.sh $MODEL_NAME $HF_MODEL_PATH $TOOL_PARSER $VLLM_PORT"

tmux new-session -d -s model "$cmd_start_server; sleep 2; bash"
echo "Started vLLM server session in tmux for $MODEL_NAME"
i=0
while ! curl -s "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; do
    sleep 10
    echo "Waiting for vLLM server... ($i/15)"
    i=$((i+1))
    if [ $i -eq 15 ]; then
        echo "ERROR: vLLM server failed to start within 15 minutes"
        exit 1
    fi
done
echo "vLLM server is ready"
DOMAINS=("telecom" "retail" "airline")
SAVE_FILE_NAME="/lustre/fsw/portfolios/llmservice/users/abhibhag/nemo-rl/results/20250924/workbench/qwen3_4binstruct/workbench-test/tau2bench"
for DOMAIN in "${DOMAINS[@]}"; do
    cmd_tb_eval="bash /lustre/fsw/portfolios/llmservice/users/rgala/repos/tau2-bench/ritu_run_dummy_baseline.sh $MODEL_NAME $DOMAIN $SAVE_FILE_NAME $VLLM_PORT"
    tmux new-session -d -s tb_${DOMAIN} "$cmd_tb_eval; sleep 2; bash"
    echo "Started tb eval session in tmux for $MODEL_NAME"
done
echo "Started tb eval session in tmux for $MODEL_NAME"