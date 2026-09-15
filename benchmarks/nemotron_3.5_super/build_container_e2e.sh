CONTAINER_IMAGE_PATH=vllm/vllm-openai:nightly-2a02f6efe319c885e3ccbcecde402e0028f9ec1e
if [ ! -e "results/$CONTAINER_IMAGE_PATH" ]; then
    mkdir -p "$(dirname "results/$CONTAINER_IMAGE_PATH")"
    enroot import -o "results/$CONTAINER_IMAGE_PATH" "docker://${CONTAINER_IMAGE_PATH}"
fi

CONTAINER=$(pwd)/results/$CONTAINER_IMAGE_PATH \
OUTPUT_DIR=$(pwd)/results/vllm_router/$CONTAINER_IMAGE_PATH \
sbatch benchmarks/nemotron_3.5_super/build_vllm_router_wheel.sh

INPUT_CONTAINER=$(pwd)/results/$CONTAINER_IMAGE_PATH \
OUTPUT_CONTAINER=$(pwd)/results/$CONTAINER_IMAGE_PATH___with_gym.sqsh \
VLLM_ROUTER_WHEEL=OUTPUT_DIR=$(pwd)/results/vllm_router/$CONTAINER_IMAGE_PATH/wheels/vllm_router-0.1.15-cp38-abi3-linux_aarch64.whl \
MOUNTS=$(pwd)/env.yaml:/opt/Gym/env.yaml:x-create=file \
GYM_CONFIG=benchmarks/nemotron_3.5_super/eval_container_config_with_staged.yaml \
TAU_2_MOUNT_BASE_GYM_DIR=$(pwd) \
sbatch benchmarks/nemotron_3.5_super/build_eval_container.sh
