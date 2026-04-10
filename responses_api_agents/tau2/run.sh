SERVED_MODEL_NAME="Qwen3.5-397B-A17B-FP8"

USER_LLM="gpt-5.2"
AGENT_TEMPERATURE=0.6
AGENT_TOP_P=0.95
AGENT_TOP_K=20
MAX_CONCURRENCY=4
SEED=42

AGENT_LLM_ARGS="{\"api_base\": \"${VLLM_URL}/v1\", \"api_key\": \"EMPTY\", \"temperature\": ${AGENT_TEMPERATURE}, \"top_p\": ${AGENT_TOP_P}, \"top_k\": ${AGENT_TOP_K}}"

cd "${SCRIPT_DIR}"

tau2 run \
    --domain "${DOMAIN}" \
    --agent-llm "openai/${SERVED_MODEL_NAME}" \
    --agent-llm-args "${AGENT_LLM_ARGS}" \
    --user-llm "${USER_LLM}" \
    --num-trials ${NUM_TRIALS} \
    --max-concurrency ${MAX_CONCURRENCY} \
    --seed ${SEED} \
    --save-to "${SAVE_TO}" \
    --auto-resume

## check results
# tau2 evaluate-trajs data/simulations/airline_agent-Qwen3.5-397B-A17B-FP8_user-gpt5.2_nt16_seed42/results/
# tau2 evaluate-trajs data/simulations/retail_agent-Qwen3.5-397B-A17B-FP8_user-gpt5.2_nt8_seed42/results/
# tau2 evaluate-trajs data/simulations/telecom_agent-Qwen3.5-397B-A17B-FP8_user-gpt5.2_nt8_seed42/results/
