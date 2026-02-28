#!/usr/bin/env bash
set -euo pipefail

epic="${1:?Usage: run-epic.sh <epic-id> [timeout-minutes]}"
timeout_mins="${2:-15}"
repo_url="${REPO_URL:?REPO_URL required}"
branch="${BRANCH:?BRANCH required}"

# --- Clone and set up ---
echo "Cloning $repo_url (branch: $branch)..."
git clone --branch "$branch" "$repo_url" /workspace
cd /workspace
git config --global --add safe.directory /workspace

logdir="/workspace/logs/epic-runs"
mkdir -p "$logdir"

# --- Initialize litebrite ---
echo "Initializing litebrite..."
lb init

# Verify the epic exists before starting
if ! lb show "$epic" > /dev/null 2>&1; then
  echo "ERROR: Epic $epic not found in litebrite"
  echo "Available items:"
  lb list
  exit 1
fi

# --- Create feature branch from item title ---
item_info="$(lb show "$epic" 2>/dev/null)"
item_title="$(echo "$item_info" | head -1 | sed 's/^[^ ]* //')"
slug="$(echo "$item_title" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g; s/--*/-/g; s/^-//; s/-$//' | cut -c1-50)"
feature_branch="${epic}-${slug}"

echo "Creating feature branch: $feature_branch"
git checkout -b "$feature_branch"

# --- Task loop ---
remaining() {
  lb list --parent "$epic" -s open 2>/dev/null | wc -l || echo 0
}

sync_and_push() {
  lb sync
  git push -u origin "$feature_branch"
}

task_num=0
failures=0
max_failures=3
while [ "$(remaining)" -gt 0 ]; do
  task_num=$((task_num + 1))
  logfile="$logdir/${epic}-task-${task_num}-$(date +%H%M%S).log"

  echo "=== Task $task_num | $(remaining) remaining | log: $logfile ==="

  set +e
  timeout "${timeout_mins}m" claude \
    "Run 'lb list --parent $epic' to see tasks. Pick ONE open child task and complete it. Do NOT work on tasks outside this epic. Commit your changes and close the item when done. Do NOT push — the runner handles pushing." \
    --model opus \
    --dangerously-skip-permissions \
    -p --verbose 2>&1 | tee "$logfile"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [ "$exit_code" -eq 124 ]; then
    echo "--- Task $task_num timed out after ${timeout_mins}m — skipping"
    failures=$((failures + 1))
  elif [ "$exit_code" -ne 0 ]; then
    echo "--- Task $task_num exited with code $exit_code"
    failures=$((failures + 1))
  else
    failures=0
    echo "--- Task $task_num succeeded, pushing to remote..."
    sync_and_push
  fi

  if [ "$failures" -ge "$max_failures" ]; then
    echo "ERROR: $max_failures consecutive failures — aborting"
    break
  fi

  echo ""
  lb list
  echo ""
done

echo ""
echo "All tasks in $epic complete."
echo "Final push to remote..."
sync_and_push
echo "Done. Merge branch '$feature_branch' into '$branch' when ready."
