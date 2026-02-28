# AI Agent Harness

A GitHub template for autonomous AI coding agents. Docker-isolated runs, litebrite-based task tracking, and iterative implementation from a spec.

## Quick Start

1. **Create a repo from this template** (or clone directly)
2. **Write your spec** in `SPEC.md` — describe what you want built
3. **Set up credentials**:
   ```bash
   cp .env.example .env
   # Add your ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN
   ```
4. **Run the agent**:
   ```bash
   scripts/run.sh
   ```

The agent reads your spec, creates tasks via litebrite, implements them, and pushes code.

## Modes

### Single Run (`scripts/run.sh`)

Runs the agent once. It picks up existing tasks or creates new ones from the spec.

```bash
scripts/run.sh
```

### Loop (`scripts/run-loop.sh`)

Runs the agent in a tight loop — back-to-back runs with no gap by default.

```bash
scripts/run-loop.sh          # No delay between runs
scripts/run-loop.sh 300      # 5-min delay between runs
```

### Epic (`scripts/run-epic.sh`)

Runs a task loop for a specific epic. Creates a feature branch and works through each child task.

```bash
scripts/run-epic.sh <epic-id>              # 15-min timeout per task
scripts/run-epic.sh <epic-id> 30           # 30-min timeout per task
```

## Prerequisites

- Docker
- Git with SSH access to your repo
- `lb` (litebrite) CLI
- An Anthropic API key or Claude Code OAuth token

## Customization

### Adding project dependencies

The agent can modify `scripts/runner/Dockerfile` to add whatever your project needs (Python, Rust, system packages, etc.). You can also edit it manually before running.

### Task tracking

All work is tracked via litebrite (`lb`). The agent creates epics and tasks, marks them in-progress, and closes them on completion. Run `lb list` to see current state.

### Git hooks

Add project-specific checks (linting, tests) as needed.

## How It Works

1. **Host script** builds a Docker container, mounts SSH agent + persistent Claude volume
2. **Container** clones the repo fresh, initializes litebrite, runs Claude Code with a prompt
3. **Claude** reads `SPEC.md`, checks litebrite for tasks, implements work, commits, pushes
4. **Host script** pulls the changes after the container finishes

## Files

| File | Purpose |
|------|---------|
| `SPEC.md` | Your project specification (edit this) |
| `CLAUDE.md` | Agent instructions |
| `AGENTS.md` | Landing the Plane protocol |
| `scripts/run.sh` | Host: single agent run |
| `scripts/run-loop.sh` | Host: repeated runs |
| `scripts/run-epic.sh` | Host: epic task runner |
| `scripts/runner/Dockerfile` | Container image |
| `scripts/runner/run.sh` | Container entrypoint (single run) |
| `scripts/runner/run-epic.sh` | Container entrypoint (epic mode) |
| `.claude/settings.local.json` | Claude Code permissions |
