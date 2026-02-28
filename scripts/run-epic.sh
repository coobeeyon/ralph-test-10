#!/usr/bin/env bash
set -euo pipefail

epic="${1:?Usage: run-epic.sh <epic-id> [timeout-minutes]}"
timeout_mins="${2:-15}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
runner_dir="$script_dir/runner"

repo_url="$(git -C "$project_dir" remote get-url origin)"
branch="$(git -C "$project_dir" branch --show-current)"

# Pre-run litebrite sync
lb sync 2>/dev/null || true

# Build the container image (all layers cached unless versions change)
echo "Building runner container..."
docker build -q -t agent-runner \
  --build-arg HOST_UID="$(id -u)" \
  --build-arg HOST_GID="$(id -g)" \
  -f "$runner_dir/Dockerfile" "$project_dir"

container_name="epic-${epic}"

echo "Running epic $epic on branch $branch (${timeout_mins}m timeout per task)..."
echo "Container name: $container_name"

# Remove stale container with same name if it exists
docker rm "$container_name" 2>/dev/null || true

docker run --name "$container_name" \
  --env-file "$project_dir/.env" \
  -e ANTHROPIC_API_KEY \
  -e REPO_URL="$repo_url" \
  -e BRANCH="$branch" \
  -v "${SSH_AUTH_SOCK}:/ssh-agent" \
  -e SSH_AUTH_SOCK=/ssh-agent \
  -v "$runner_dir/run-epic.sh:/run-epic.sh:ro" \
  agent-runner /run-epic.sh "$epic" "$timeout_mins"

echo "Container $container_name finished. Cleaning up..."
docker rm "$container_name"

# Pull changes and sync litebrite
echo "Fetching results from remote..."
git -C "$project_dir" fetch origin
lb sync 2>/dev/null || true
echo "Done. Check remote branches for the feature branch."
