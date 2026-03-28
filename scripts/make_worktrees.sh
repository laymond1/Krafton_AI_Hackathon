#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: ./scripts/make_worktrees.sh [base-ref]

Creates the recommended worktrees:
- ../wt-claude
- ../wt-codex
- ../wt-final

Default base ref: main
EOF
  exit 1
}

BASE_REF="${1:-main}"

if [[ "${BASE_REF}" == "-h" || "${BASE_REF}" == "--help" ]]; then
  usage
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"

if [[ -z "$REPO_ROOT" ]]; then
  echo "This script must be run inside a git repository." >&2
  exit 1
fi

cd "$REPO_ROOT"

if ! git rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
  echo "Base ref does not exist: $BASE_REF" >&2
  exit 1
fi

PARENT_DIR="$(cd "$REPO_ROOT/.." && pwd)"

ensure_branch() {
  local branch="$1"

  if git show-ref --verify --quiet "refs/heads/$branch"; then
    echo "Branch already exists: $branch"
  else
    git branch "$branch" "$BASE_REF"
    echo "Created branch: $branch"
  fi
}

ensure_worktree() {
  local branch="$1"
  local path="$2"

  if git worktree list --porcelain | grep -Fq "worktree $path"; then
    echo "Worktree already registered: $path"
    return
  fi

  if [[ -e "$path" && ! -e "$path/.git" ]]; then
    echo "Path exists and is not a registered worktree: $path" >&2
    exit 1
  fi

  git worktree add "$path" "$branch"
  echo "Created worktree: $path -> $branch"
}

ensure_branch "ops/claude"
ensure_branch "ops/codex"
ensure_branch "ops/final"

ensure_worktree "ops/claude" "$PARENT_DIR/wt-claude"
ensure_worktree "ops/codex" "$PARENT_DIR/wt-codex"
ensure_worktree "ops/final" "$PARENT_DIR/wt-final"
