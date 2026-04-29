# Research Workflow

This repository uses file-based state so future Codex sessions can recover research context reliably.

## Ownership Model

The files under `research/` are maintained by Codex as part of the working process.

The user should not need to manually keep these documents in sync during normal collaboration.

Codex is responsible for:

- reading them at session start
- updating them during or after meaningful work
- keeping them aligned with code changes, experiment progress, and recent git history

## Session Start

At the beginning of every new session, read these files in order:

1. `research/PROJECT_STATE.md`
2. `research/WORKLOG.md`
3. `research/EXPERIMENT_LOG.md`
4. `research/DECISIONS.md`
5. `README.md`

Then inspect recent git history:

1. `git log --oneline --decorate -n 20`
2. `git log --stat -n 10` if the current direction is still unclear

## Session End

Before ending a session, update the relevant documents so code and docs stay aligned.

Required rule:

- Any meaningful code change must be reflected in documentation during the same session.
- If the session mainly involves analysis, conclusions and next steps should still be written back to `research/`.

## Document Ownership

- `research/PROJECT_STATE.md`: current research objective, active branch, main hypothesis, current best understanding, next step
- `research/WORKLOG.md`: chronological record of what was done in each session
- `research/EXPERIMENT_LOG.md`: structured experiment records with config, results, conclusion, next action
- `research/DECISIONS.md`: durable decisions and their rationale

## Mapping Rules

- Training logic change: update `research/WORKLOG.md` and either `research/DECISIONS.md` or `research/EXPERIMENT_LOG.md`
- Config change: update `research/EXPERIMENT_LOG.md`
- Bug fix: update `research/WORKLOG.md`
- Research focus change: update `research/PROJECT_STATE.md`
- Important conclusion or abandoned direction: update `research/DECISIONS.md`

## Writing Rules

- Keep `research/PROJECT_STATE.md` short and current. Replace stale status instead of appending long notes.
- Keep `research/WORKLOG.md` chronological and concrete.
- Each `research/EXPERIMENT_LOG.md` entry should include:
  - goal
  - changed files or config
  - metrics
  - conclusion
  - next step
- If a session changes code but does not yet run experiments, log that explicitly.
