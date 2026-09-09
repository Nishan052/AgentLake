# Skill Evals

How this repo checks that its skills actually work, before they ship and on
every change after.

The tier model and the three tools in `tools/` come from Shubham Saboo's
[awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)
(`agent_skills/evals/`), Apache 2.0, which in turn credits
[addyosmani/agent-skills](https://github.com/addyosmani/agent-skills/tree/main/evals)
for tiers 1-3 and Anthropic's skill-creator for the `evals.json` schema. See
[`../NOTICE`](../NOTICE) for attribution and the list of local modifications.

Layout: one folder per skill, mirroring its name. Nothing here ships in an
install. The skill folders under `.claude/skills/` contain only what runs at
runtime, which is why you can copy one out and it works.

| Tier | What it checks | Runs | Cost |
|---|---|---|---|
| 1. Structural | Frontmatter, naming, name equals dir, dead file references, prose-only skills (`tools/skill_lint.py --strict`) | CI | Free |
| 1b. Security | Install lures, undeclared network calls, credential access, obfuscated payloads (`tools/skill_scanner.py`) | CI | Free |
| 2. Trigger & routing | Positive prompts clear near-miss negatives on description vocabulary, and each positive ranks its own skill first (`tools/run_trigger_evals.py`) | CI | Free |
| 2b. Deterministic scripts | Each skill's bundled script does what it claims, against committed fixtures (`<skill>/test_*.py`) | CI | Free, ~2s |
| 3. Figures | Every number quoted about these tools regenerates from the clone (`../verify.sh`) | CI | Free |

## Running

```bash
python3 skill-evals/tools/skill_lint.py .claude/skills/npu-op-compat --strict
python3 skill-evals/tools/skill_scanner.py .claude/skills
python3 skill-evals/tools/run_trigger_evals.py
for t in skill-evals/*/test_*.py; do python3 "$t"; done
./verify.sh
```

All of it is standard library Python, makes no network calls, and never executes
the code it scans. `.github/workflows/skill-evals.yml` runs the same commands.

## Track record

Not theatre. These tools are run against a larger working catalogue than the two
skills published here, and on the first run against it they caught:

- **A skill referencing a script that did not exist** in its own directory
  (tier 1). A broken path a reader would have hit on first use.
- **A prose-only skill** with no bundled tool (tier 1). Writing the missing tool
  then found the next thing.
- **Every scene in a video script too dense to read** at its duration (tier 2b).
  One ran 30 words in 5.5 seconds, about 327 words per minute, well past the
  roughly 200 wpm subtitle standard.
- **Three routing collisions between three skills** (tier 2). "I want a linkedin
  video" routed to the post writer, not the video renderer. All three
  descriptions were rewritten to own disjoint vocabulary.
- **A lesson in writing descriptions**: adding "produces no video, no MP4" to a
  description made routing *worse*, because a lexical scorer reads that as video
  vocabulary. Negative disclaimers do not work. Give each skill its own words
  instead.

## Adding a skill

A new skill needs a `skill-evals/<skill-name>/` with at minimum:

- `trigger-cases.json` — positives plus near-miss negatives. A case marked
  `"lexical": false` is skipped by tier 2 and belongs to a behavioral tier.
- `test_<skill>.py` — self-contained, tempdir fixtures, exit 0/1, stdlib only.

CI picks up `test_*.py` automatically.

## What tier 2 cannot do

It is a lexical approximation of routing, not a semantic one. It catches the two
failure modes that dominate real trigger bugs: a description missing the words
users actually say, and an over-broad description that outranks the right skill.
It cannot weight a single discriminating word inside two otherwise identical
prompts. Where a near-miss case is worded to avoid that limit, the case file
says so and explains why the routing distinction under test is unchanged.
