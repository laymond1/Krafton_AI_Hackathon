#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: ./scripts/sync_outputs.sh <day1|day2>

Syncs outputs from sibling worktrees back into the current repository:
- ../wt-claude/runs/<day>/claude -> runs/<day>/claude
- ../wt-codex/runs/<day>/codex -> runs/<day>/codex
- ../wt-final/runs/<day>/final -> runs/<day>/final
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
PARENT_DIR="$(cd "$ROOT/.." && pwd)"

sync_dir() {
  local src="$1"
  local dest="$2"
  local label="$3"

  if [[ ! -d "$src" ]]; then
    echo "Skipping $label: missing $src"
    return
  fi

  mkdir -p "$dest"

  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$src/" "$dest/"
  else
    cp -R "$src/." "$dest/"
  fi

  echo "Synced $label"
}

mkdir -p "$ROOT/runs/$DAY"

sync_dir "$PARENT_DIR/wt-claude/runs/$DAY/claude" "$ROOT/runs/$DAY/claude" "Claude outputs"
sync_dir "$PARENT_DIR/wt-codex/runs/$DAY/codex" "$ROOT/runs/$DAY/codex" "Codex outputs"
sync_dir "$PARENT_DIR/wt-final/runs/$DAY/final" "$ROOT/runs/$DAY/final" "Final outputs"
