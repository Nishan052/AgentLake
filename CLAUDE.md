# CLAUDE.md

Conventions for this repo. Read before changing anything.

## This repo is a destination, not a workspace

Everything under `.claude/skills/` and `skill-evals/` is **copied here from a
private working repository** by a promotion step that overwrites it. An edit
made directly to one of those files is lost the next time a skill is promoted,
silently, with no merge and no conflict.

Change the skill upstream and promote it. If you are reading this without access
to that repository, open a pull request instead and it will be applied at the
source.

Files that belong to this repo and are never overwritten:

```
README.md   NOTICE   LICENSE   CLAUDE.md   verify.sh   .gitignore
skill-evals/README.md   .github/workflows/skill-evals.yml
```

## What may live here

Only tools that are useful to somebody who has never heard of me. In practice
that means edge and agent skills that:

- run on the Python standard library alone
- make no network call and execute nothing at install time
- ship their own tests and fixtures, in the same commit
- carry a source URL and a retrieval date for every piece of vendor data

No articles, no videos, no drafts, no publishing pipeline, no social copy, and
no personal workflow tooling. Those stay upstream. A reader wants the tool.

## The rules the tools obey

- **Never ship an operator table or a device specification from memory.** A
  table invented by a language model is worse than no tool, because it is
  confidently wrong in exactly the cases the user cannot check.
- **Do not ship data that sits behind a login.** Say what is missing and why,
  rather than reconstructing it from forum posts.
- **Unknown is a valid answer.** An agent that guesses where it cannot resolve
  is the failure mode these tools exist to fix.
- **No skill without its evals.** An untested skill in a public repo is a
  liability, not a portfolio piece.

## Checks that must pass

```bash
./verify.sh
python3 skill-evals/tools/skill_lint.py .claude/skills/<name> --strict
python3 skill-evals/tools/skill_scanner.py .claude/skills
python3 skill-evals/tools/run_trigger_evals.py
```

Never edit a checker to make input pass. Fix the input.

## No secrets, ever

This repository is public. No `.env`, no API keys, no employer data, no result
that came off hardware whose code cannot be released alongside it.
