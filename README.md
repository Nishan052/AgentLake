# AgentLake

Where finished agent skills come to rest.

Each one arrives with the evidence that it works and a command you can run to
check that evidence yourself. Nothing lands here while it is still being worked
out.

Two so far, both about putting a neural network on edge hardware.

| Skill | Answers | Backed by |
|---|---|---|
| [`npu-op-compat`](.claude/skills/npu-op-compat/) | Will this model actually use the accelerator, or fall off onto the CPU at the second layer? | Google's published operator table, and a committed ONNX model you can rerun |
| [`edge-feasibility`](.claude/skills/edge-feasibility/) | Will it fit in memory, and roughly how fast, before I own the device? | 16 measured figures published by other people, two platforms held out |

Python 3 standard library only. No network calls, no install-time execution,
nothing to configure. Read them before you run them, as you should with anything
that gets your shell.

```bash
git clone https://github.com/Nishan052/AgentLake
cd AgentLake && ./verify.sh
```

`verify.sh` regenerates every figure quoted below and exits non-zero if one has
changed. If this page says 93 percent and the script says something else, one of
them is wrong and you can see which.

---

## Find where your model stops being accelerated

```bash
python3 .claude/skills/npu-op-compat/scripts/check_ops.py your-model.onnx
```

The Edge TPU compiler partitions a graph **exactly once**, at the first
unsupported operator. Everything after that point runs on CPU, however well
supported it is.

That single fact inverts the usual advice. Counting unsupported operators tells
you "half my activations are unsupported" and sends you off to fix all of them.
Position is what matters:

```
target: Google Coral Edge TPU   source: https://coral.ai/docs/edgetpu/models-intro/ (retrieved 2026-09-06)
15 operations in graph order

  ok   [  0] Conv -> Conv2d
  STOP [  1] LeakyRelu  <-- partition cut
         Not in the table. PReLU is supported, LeakyRelu is not.
  ok   [  2] Conv -> Conv2d   (on CPU: after the cut)
  STOP [  3] LeakyRelu   (on CPU: after the cut)
  ...
  ok   [ 14] Conv -> Conv2d   (on CPU: after the cut)

First unsupported operation: LeakyRelu at index 1.
1 operations accelerate. 14 run on CPU (93% of the graph).
```

A detection backbone strands 93% of its graph on one activation function,
because that activation is second. Fix that one and the rest follows.

It parses ONNX with **no modelling library installed**, so it runs before you
have set up a conversion toolchain, which is when the answer is worth the most.
Ships 40 Edge TPU operators with their documented limitations, each sourced and
dated, plus a 35-entry ONNX name mapping.

## Work out whether a model fits, and roughly how fast

```bash
python3 .claude/skills/edge-feasibility/scripts/feasibility.py --params 7B --quant q4_0 --device m3
python3 .claude/skills/edge-feasibility/scripts/feasibility.py --devices
python3 .claude/skills/edge-feasibility/scripts/feasibility.py --validate
```

Generating one token means reading every weight out of memory, so single-stream
decode is bandwidth bound and the compute rating of the chip barely enters into
it. That makes the question answerable from vendor specifications:

```
Apple M3  (100 GB/s, 24 GB)
7.00B parameters at 4.5 bits per weight, 4096 token context

  weights              3.94 GB
  key-value cache      2.24 GB   (32 layers estimated)
  runtime overhead     1.20 GB   (rule of thumb)
  ----------------------------------
  total                7.38 GB  of 24 GB   -> FITS

  ceiling              16.2 tokens/sec   (bandwidth / bytes per token)
  expected              6.2 to 13.8 tokens/sec   (38%-85% of ceiling)
```

**It refuses to give a point estimate.** An efficiency curve fitted to fourteen
Apple measurements scored 14 of 14 within 30% in sample, then missed two
held-out platforms by about 49% each, in the same direction. Achieved efficiency
is a property of a memory subsystem, not a function of a bandwidth number.

So what ships is the pair of things the data supports: a ceiling not exceeded by
any of the 16 published measurements, across three platform families, and a band
that contains all of them. The band is wide, and the width is the finding.
`--validate` prints the scoring, every row, including the held-out ones.

---

## Install

Each skill is a folder with a `SKILL.md` and its scripts. Copy the folder into
your agent's skills directory:

| Agent | Directory |
|---|---|
| Claude Code | `~/.claude/skills/` |
| Codex | `~/.codex/skills/` |
| Cursor | `~/.cursor/skills/` |
| Shared in a repository | `.agents/skills/` |

```bash
cp -r AgentLake/.claude/skills/npu-op-compat ~/.claude/skills/
cp -r AgentLake/.claude/skills/edge-feasibility ~/.claude/skills/
```

They work standalone and depend on nothing else in this repository. The scripts
also run straight from the command line, with no agent involved.

Ask your agent "why is my model slow on the Coral" or "will a 7B fit on a Pi 5"
and it picks the right one on its own. The routing is tested, not assumed.

## How these are tested

[`skill-evals/`](skill-evals/) holds five tiers of checks, all free, all in CI:
structure, security scan, trigger routing, script behaviour, and the figures on
this page. Run them from the clone before you install anything:

```bash
python3 skill-evals/tools/skill_scanner.py .claude/skills     # what it would do to your machine
python3 skill-evals/npu-op-compat/test_npu_op_compat.py
python3 skill-evals/edge-feasibility/test_edge_feasibility.py
```

`skill-evals/README.md` lists what these caught rather than claiming they work.

## What gets in

A skill lands here once it is finished. In practice that means it runs on the
standard library alone, makes no network call, executes nothing at install time,
ships its own tests in the same commit, and carries a source and a date for
every piece of vendor data it restates.

Everything here is copied in from a private working repository by a gated step,
so an edit made directly to a skill in this repo is lost rather than merged.
Open a pull request and it is applied at the source. See
[CLAUDE.md](CLAUDE.md) for which files this repo owns outright.

## What is not here

**No benchmark I did not run.** There is no hardware in this repository and no
measurements taken on any. Where a figure comes from someone else's device it
says so, with the source. Where a result is arithmetic over a published table it
says that instead.

**No operator table written from memory.** A support table invented by a
language model is worse than no tool, because it is confidently wrong in exactly
the cases you cannot check. Every entry has a source URL and a retrieval date.
Hailo's full operator list sits behind a login, which is why this ships Coral and
says so rather than reconstructing it from forum posts.

**No answer where there is no evidence.** Unknown is a valid output. An agent
that guesses where it cannot resolve is the failure mode these tools exist to
fix.

## Licence

MIT, see [LICENSE](LICENSE). Third-party attribution in [NOTICE](NOTICE).
Written by [Nishan Chandrashekar Poojary](https://nishanpoojary.com).
