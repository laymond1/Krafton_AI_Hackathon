#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: ./scripts/init_run.sh <day1|day2>
EOF
  exit 1
}

DAY="${1:-}"

case "$DAY" in
  day1|day2)
    ;;
  *)
    usage
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/runs/$DAY"

copy_if_missing() {
  local src="$1"
  local dest="$2"

  if [[ ! -f "$src" ]]; then
    echo "Missing template: $src" >&2
    exit 1
  fi

  if [[ ! -e "$dest" ]]; then
    cp "$src" "$dest"
  fi
}

mkdir -p \
  "$RUN_DIR/raw_problem" \
  "$RUN_DIR/claude" \
  "$RUN_DIR/codex" \
  "$RUN_DIR/merged" \
  "$RUN_DIR/final" \
  "$ROOT/scratch/quick_tests" \
  "$ROOT/scratch/temp_outputs" \
  "$ROOT/submissions/$DAY"

copy_if_missing "$ROOT/templates/problem_brief.md" "$RUN_DIR/merged/problem_brief.md"
copy_if_missing "$ROOT/templates/solution_brief.md" "$RUN_DIR/merged/solution_brief.md"
copy_if_missing "$ROOT/templates/risk_checklist.md" "$RUN_DIR/merged/risk_checklist.md"
copy_if_missing "$ROOT/templates/experiment_log.md" "$RUN_DIR/merged/experiment_log.md"
copy_if_missing "$ROOT/templates/comparison_table.md" "$RUN_DIR/merged/comparison_table.md"
copy_if_missing "$ROOT/templates/final_submission.md" "$RUN_DIR/final/final_submission.md"

if [[ ! -e "$RUN_DIR/raw_problem/problem.md" ]]; then
  cat > "$RUN_DIR/raw_problem/problem.md" <<'EOF'
# Raw Problem

Paste the original problem statement here without rewriting it.
EOF
fi

touch "$RUN_DIR/claude/.keep" "$RUN_DIR/codex/.keep"

cat <<EOF
Initialized run directory: $RUN_DIR

Created:
- raw problem intake area
- Claude output area
- Codex output area
- merged working area
- final submission area

Suggested next steps:
0. Use the default runtime wrapper for Python commands: ./scripts/with_wslim.sh python ...
1. Paste the released problem into runs/$DAY/raw_problem/problem.md
2. Run the Claude and Codex parser prompts against the same source text
3. Save outputs into runs/$DAY/claude and runs/$DAY/codex
EOF
