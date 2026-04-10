tau2 run \
    --domain airline \
    --agent-llm openai/Qwen3.5-397B-A17B-FP8 \
    --agent-llm-args "{\"api_base\": \"${VLLM_URL}/v1\", \"api_key\": \"EMPTY\", \"temperature\": 0.6, \"top_p\": 0.95, \"top_k\": 20}" \
    --user-llm gpt-5.2 \
    --num-trials 8 \
    --max-concurrency 4 \
    --seed 42 \
    --save-to TODO \
    --auto-resume

tau2 evaluate-trajs data/simulations/airline_agent-Qwen3.5-397B-A17B-FP8_user-gpt5.2_nt16_seed42/results/
