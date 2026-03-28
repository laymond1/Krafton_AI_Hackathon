#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: ./scripts/diff_summary.sh <day1|day2> [relative-path]

Without a relative path, this prints a same-or-different summary for
files shared by runs/<day>/claude and runs/<day>/codex.

With a relative path, this prints a unified diff for that file.
EOF
  exit 1
}

DAY="${1:-}"
REL_PATH="${2:-}"

case "$DAY" in
  day1|day2)
    ;;
  *)
    usage
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLAUDE_DIR="$ROOT/runs/$DAY/claude"
CODEX_DIR="$ROOT/runs/$DAY/codex"

if [[ ! -d "$CLAUDE_DIR" || ! -d "$CODEX_DIR" ]]; then
  echo "Missing comparison directories for $DAY." >&2
  exit 1
fi

if [[ -n "$REL_PATH" ]]; then
  CLAUDE_FILE="$CLAUDE_DIR/$REL_PATH"
  CODEX_FILE="$CODEX_DIR/$REL_PATH"

  if [[ ! -f "$CLAUDE_FILE" ]]; then
    echo "Missing Claude file: $CLAUDE_FILE" >&2
    exit 1
  fi

  if [[ ! -f "$CODEX_FILE" ]]; then
    echo "Missing Codex file: $CODEX_FILE" >&2
    exit 1
  fi

  diff -u "$CLAUDE_FILE" "$CODEX_FILE" || true
  exit 0
fi

CLAUDE_LIST="$(mktemp)"
CODEX_LIST="$(mktemp)"
trap 'rm -f "$CLAUDE_LIST" "$CODEX_LIST"' EXIT

(
  cd "$CLAUDE_DIR"
  find . -type f ! -name '.keep' | sort > "$CLAUDE_LIST"
)

(
  cd "$CODEX_DIR"
  find . -type f ! -name '.keep' | sort > "$CODEX_LIST"
)

echo "== Common files =="
comm -12 "$CLAUDE_LIST" "$CODEX_LIST" | while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue

  if cmp -s "$CLAUDE_DIR/$rel" "$CODEX_DIR/$rel"; then
    echo "SAME  ${rel#./}"
  else
    echo "DIFF  ${rel#./}"
  fi
done

echo
echo "== Only in Claude =="
comm -23 "$CLAUDE_LIST" "$CODEX_LIST" | sed 's#^\./##'

echo
echo "== Only in Codex =="
comm -13 "$CLAUDE_LIST" "$CODEX_LIST" | sed 's#^\./##'
