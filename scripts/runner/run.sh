#!/usr/bin/env bash
set -euo pipefail

repo_url="${REPO_URL:?REPO_URL required}"
branch="${BRANCH:?BRANCH required}"
work_dir="$HOME/workspace"

# --- Clone repo ---
echo "Cloning $repo_url (branch: $branch)..."
git clone --branch "$branch" "$repo_url" "$work_dir"
cd "$work_dir"
git config --global --add safe.directory "$work_dir"

# --- Initialize litebrite (detects remote branch automatically) ---
echo "Initializing litebrite..."
lb init

# --- Restore .claude.json from persisted backup if missing ---
claude_config="$HOME/.claude.json"
if [ ! -f "$claude_config" ] && [ -d "$HOME/.claude/backups" ]; then
  latest_backup=$(ls -t "$HOME/.claude/backups/.claude.json.backup."* 2>/dev/null | head -1)
  if [ -n "$latest_backup" ]; then
    cp "$latest_backup" "$claude_config"
    echo "Restored .claude.json from backup: $(basename "$latest_backup")"
  fi
fi

# --- Run agent ---
echo "Starting agent run..."
claude -p --dangerously-skip-permissions --output-format json --model opus "$(cat <<'PROMPT'
You are ONE agent in a relay. Do ONE task, then stop.

## Steps

1. Read SPEC.md to understand the project.
2. Run `lb list` to check for existing tasks.
3. **If no tasks exist:** create an epic with child tasks from the spec, then pick ONE task.
   **If tasks exist:** pick ONE open task.
4. Claim the task: `lb claim <id>`
5. Implement the task. Commit your code frequently with clear messages.
6. When done, run these commands IN ORDER:
   ```
   lb close <id>
   lb sync
   git push
   ```
7. STOP. Do NOT start another task. Exit immediately.

## Rules
- ONE task per session. Not two. Not "just one more." ONE.
- Every session ends with: lb close, lb sync, git push — in that order.
- The next agent will continue where you left off.
PROMPT
)" > /tmp/agent-run.json

echo "Agent run complete."

# --- Belt-and-suspenders: force sync/push even if agent forgot ---
echo "Post-agent cleanup: forcing lb sync and git push..."
lb sync 2>/dev/null || true
git push 2>/dev/null || true
