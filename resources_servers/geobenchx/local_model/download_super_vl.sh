#!/usr/bin/env bash
# Download the Nemotron 3.5 Super VL GA candidates (private nvidia repos; the
# token needs read access to the nvidia org). About 575 GB in total.
# Rerunning resumes and skips files that are already complete.
# Requires huggingface_hub with hf_xet: pip install "huggingface_hub[hf_xet]"
#
#   download_super_vl.sh [bf16] [nvfp4] [lilicorr] [base]   (default: all four)
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/datasets/pwilson/models}"
HF_BIN="${HF_BIN:-hf}"
declare -A REPOS=(
  [bf16]=nvidia/Nemotron-3.5-Super-VL-120B-A12B-GA-candidate-MTP-boosted-20261001
  [nvfp4]=nvidia/Nemotron-Super-3.5-GA-FINAL-row105-QAD-PreStitched-BoostedMTP
  [lilicorr]=nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr
  [base]=nvidia/NVIDIA-Nemotron-3.5-Super-midtrain-67B-vision-pretrained
)

variants=("$@")
[[ ${#variants[@]} -eq 0 ]] && variants=(lilicorr nvfp4 bf16 base)
for variant in "${variants[@]}"; do
  repo="${REPOS[${variant}]:?unknown variant ${variant}}"
  HF_XET_HIGH_PERFORMANCE=1 "${HF_BIN}" download "${repo}" \
    --local-dir "${MODELS_DIR}/${repo#nvidia/}" \
    --max-workers 16
done
