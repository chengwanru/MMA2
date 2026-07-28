#!/usr/bin/env bash
# Clone Meta open-eqa and install the LLM-Match scorer (evaluate-predictions.py).
#
# Usage (once per machine / venv):
#   cd MMA/public_evaluations/open_eqa
#   bash setup_openeqa_official_scorer.sh
#
# Requires: git, pip, OPENAI_API_KEY when running scores (not for setup).

set -euo pipefail

OEQA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_ROOT="${OPENEQA_OFFICIAL_ROOT:-${OEQA_DIR}/third_party/open-eqa}"
REPO_URL="${OPENEQA_OFFICIAL_REPO:-https://github.com/facebookresearch/open-eqa.git}"
REPO_REF="${OPENEQA_OFFICIAL_REF:-main}"

if [[ ! -d "${OFFICIAL_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${OFFICIAL_ROOT}")"
  echo "Cloning ${REPO_URL} -> ${OFFICIAL_ROOT}"
  git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" "${OFFICIAL_ROOT}"
else
  echo "Found existing scorer repo at ${OFFICIAL_ROOT}"
fi

echo "Installing scorer Python deps (numpy, openai, tenacity, tqdm)..."
python3 -m pip install -q --upgrade "numpy>=1.26" "openai>=1.3" "tenacity>=8.2" "tqdm>=4.66"

echo "Installing openeqa package (editable)..."
python3 -m pip install -q -e "${OFFICIAL_ROOT}"

cat <<EOF

OpenEQA official scorer ready.

  export OPENAI_API_KEY=sk-...
  python run_openeqa_official_score.py \\
      --input_file results/direct_episodic_bias_tuned_10_<jobid>.json \\
      --variant ours \\
      --force \\
      --dry-run

Full benchmark (1636 questions, needs complete predictions + API credits):
  python run_openeqa_official_score.py \\
      --input_file results/openeqa_full.json \\
      --variant ours

Repo: ${OFFICIAL_ROOT}
EOF
