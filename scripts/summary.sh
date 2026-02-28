#!/usr/bin/env bash
# Summarize the latest run log using Claude.
# Usage: summary.sh [log-file]
#   If no log-file given, reads logs/latest.jsonl symlink.
set -euo pipefail
unset CLAUDECODE

script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
log_dir="$project_dir/logs"
summary_dir="$log_dir/summaries"
mkdir -p "$summary_dir"

log_file="${1:-$log_dir/latest.jsonl}"
log_name="$(basename "$log_file" .jsonl)"
summary_file="$summary_dir/${log_name}.md"

cat <<EOF | claude -p \
  --model haiku \
  --dangerously-skip-permissions
Read the log file at $log_file. Write a concise markdown summary to $summary_file covering:
- What tasks were worked on
- What was accomplished (files created/modified, commits)
- Whether the run succeeded or failed (and why)
- Any errors or notable events

Also print the summary to stdout.
EOF

echo ""
echo "Summary saved: $summary_file"
