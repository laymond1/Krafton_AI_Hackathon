#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: ./scripts/package_submission.sh <day1|day2> [label]

Packages runs/<day>/final plus selected supporting files into
submissions/<day>/<package>.tar.gz
EOF
  exit 1
}

DAY="${1:-}"
LABEL="${2:-submission}"

case "$DAY" in
  day1|day2)
    ;;
  *)
    usage
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/runs/$DAY"
FINAL_DIR="$RUN_DIR/final"
MERGED_DIR="$RUN_DIR/merged"
SUBMISSIONS_DIR="$ROOT/submissions/$DAY"

if [[ ! -d "$FINAL_DIR" ]]; then
  echo "Missing final directory: $FINAL_DIR" >&2
  exit 1
fi

mkdir -p "$SUBMISSIONS_DIR"

SAFE_LABEL="$(printf '%s' "$LABEL" | tr ' /' '__')"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
PACKAGE_NAME="${DAY}-${SAFE_LABEL}-${TIMESTAMP}"
STAGING_DIR="$SUBMISSIONS_DIR/$PACKAGE_NAME"
ARCHIVE_PATH="$SUBMISSIONS_DIR/$PACKAGE_NAME.tar.gz"

mkdir -p "$STAGING_DIR/final" "$STAGING_DIR/supporting"

if find "$FINAL_DIR" -mindepth 1 -print -quit | grep -q .; then
  cp -R "$FINAL_DIR/." "$STAGING_DIR/final/"
fi

copy_supporting() {
  local path="$1"

  if [[ -f "$path" ]]; then
    cp "$path" "$STAGING_DIR/supporting/"
  fi
}

copy_supporting "$MERGED_DIR/problem_brief.md"
copy_supporting "$MERGED_DIR/solution_brief.md"
copy_supporting "$MERGED_DIR/comparison_table.md"
copy_supporting "$MERGED_DIR/risk_checklist.md"
copy_supporting "$MERGED_DIR/experiment_log.md"

{
  echo "package_name: $PACKAGE_NAME"
  echo "created_at: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "day: $DAY"
  echo "source_run_dir: runs/$DAY"
  echo "included_files:"
  (
    cd "$STAGING_DIR"
    find . -type f | sort | sed 's#^\./#- #'
  )
} > "$STAGING_DIR/manifest.txt"

tar -czf "$ARCHIVE_PATH" -C "$SUBMISSIONS_DIR" "$PACKAGE_NAME"

cat <<EOF
Created package:
- staging: $STAGING_DIR
- archive: $ARCHIVE_PATH
EOF
