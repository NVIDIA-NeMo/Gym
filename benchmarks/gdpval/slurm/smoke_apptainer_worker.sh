#!/usr/bin/env bash
set -euo pipefail

readonly apptainer_bin="${GDPVAL_APPTAINER_BIN:-apptainer}"
readonly container_path="${GDPVAL_CONTAINER_PATH:-}"
readonly smoke_dir="${GDPVAL_APPTAINER_SMOKE_DIR:-/tmp/gdpval-apptainer-smoke-${SLURM_JOB_ID:-manual}}"

if [[ -z "${container_path}" ]]; then
    echo "ERROR: GDPVAL_CONTAINER_PATH is required (path to the Apptainer image)." >&2
    exit 2
fi

test -x "${apptainer_bin}"
test -r "${container_path}"
mkdir -p "${smoke_dir}"
chmod 700 "${smoke_dir}"

echo "worker=$(hostname) arch=$(uname -m) smoke_dir=${smoke_dir}"
"${apptainer_bin}" version
"${apptainer_bin}" inspect "${container_path}" >/dev/null
echo "sif_inspect=ok"

"${apptainer_bin}" exec \
  --writable-tmpfs \
  --cleanenv \
  --pid \
  --no-mount home,tmp,bind-paths \
  --home /root \
  --mount "type=bind,src=${smoke_dir},dst=/workspace_io" \
  "${container_path}" \
  /bin/bash -lc '
    set -euo pipefail
    python -c "import importlib; modules=(\"numpy\",\"pandas\",\"polars\",\"scipy\",\"matplotlib\",\"plotly\",\"sklearn\",\"docx\",\"pptx\",\"openpyxl\",\"fitz\",\"pdfplumber\",\"reportlab\",\"weasyprint\",\"PIL\",\"cv2\",\"playwright\"); [importlib.import_module(module) for module in modules]; print(\"python_imports=ok count=17\")"
    for tool in libreoffice tesseract pandoc pdftotext gs ffmpeg dot convert pdflatex xelatex lualatex latexmk biber java gdalinfo jq unzip chromium timeout git; do
      command -v "${tool}" >/dev/null
    done
    printf "%s\n" "apptainer_workspace_roundtrip=ok" > /workspace_io/roundtrip.txt
    test "$(cat /workspace_io/roundtrip.txt)" = "apptainer_workspace_roundtrip=ok"
    echo "system_tools=ok count=20"
    echo "provider_style_exec=ok"
  '

test "$(cat "${smoke_dir}/roundtrip.txt")" = "apptainer_workspace_roundtrip=ok"
echo "host_bind_roundtrip=ok"
