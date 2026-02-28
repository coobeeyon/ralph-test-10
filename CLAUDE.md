# Project Instructions

## Critical Rules

1. **ONE task per session.** Not two. Not "just one more." ONE.
2. **Always close + sync + push before exiting:** `lb close <id>`, `lb sync`, `git push` — in that order.
3. **You're part of a relay.** The next agent continues where you left off. Exit promptly.

## Workflow

1. Read `SPEC.md` — it defines what to build
2. Run `lb list` to see existing tasks
3. If no tasks exist: create an epic with child tasks from the spec, then pick ONE task
4. If tasks exist: pick ONE open task, claim it (`lb claim <id>`)
5. Implement the task. Commit frequently with clear messages.
6. When done, run in order: `lb close <id>`, `lb sync`, `git push`
7. STOP. Do NOT start another task — exit and let the next agent handle it.

## Conventions

- Use litebrite (`lb`) for all task tracking
- Research anything you're unsure about before implementing
