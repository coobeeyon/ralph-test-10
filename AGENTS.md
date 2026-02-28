# Agent Reference

## lb Commands

```bash
lb list              # See all open tasks
lb ready             # Find available work (unblocked, unclaimed)
lb show <id>         # View task details
lb claim <id>        # Claim a task
lb close <id>        # Mark task complete
lb sync              # Sync task state with remote
```

## Exit Checklist

1. `lb close <id>` — close your task
2. `lb sync` — sync task state to remote
3. `git push` — push code to remote
