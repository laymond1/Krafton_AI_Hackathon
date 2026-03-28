#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: ./scripts/with_wslim.sh <command> [args...]
EOF
  exit 1
}

if [[ $# -eq 0 ]]; then
  usage
fi

CONDA_SH="/mnt/sdab1/userHome/wslim/miniconda/etc/profile.d/conda.sh"
ENV_NAME="wslim"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Missing Conda activation script: $CONDA_SH" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$ENV_NAME"
exec "$@"
