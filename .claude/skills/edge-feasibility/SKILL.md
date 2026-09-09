---
name: edge-feasibility
description: >-
  Work out whether a language model will fit on a given piece of hardware and
  roughly how many tokens per second it will generate, from vendor memory
  bandwidth and model size alone. Use when choosing hardware for local
  inference, choosing a model size or quantisation for a machine you already
  have, checking whether a quoted tokens-per-second figure is physically
  possible, or asking whether a long context window will still fit in memory.
  Reports a hard bandwidth ceiling and an empirical range, both scored against
  16 independent measured figures it ships.
license: MIT
metadata:
  author: "Nishan Poojary"
  version: "1.0.0"
  source: "https://github.com/Nishan052/AgentLake"
---

# Feasibility

Generating one token requires reading every weight, plus the key-value cache, out
of memory. The arithmetic per byte read is tiny, so **single-stream decode is
bandwidth bound**: throughput is bytes per second divided by bytes per token, and
the compute rating of the chip is close to irrelevant.

That makes the question answerable before you own the hardware.

```bash
python3 .claude/skills/edge-feasibility/scripts/feasibility.py --params 7B --quant q4_0 --device m3
python3 .claude/skills/edge-feasibility/scripts/feasibility.py --devices
python3 .claude/skills/edge-feasibility/scripts/feasibility.py --validate
```

## Three answers, in decreasing order of confidence

**1. Does it fit.** Weights plus key-value cache plus runtime overhead against
installed memory. Arithmetic. Trust it.

**2. The ceiling.** `bandwidth / bytes per token`. A physical upper bound. **Not
exceeded by any of the 16 published measurements this skill ships**, across
three platform families. This is the most useful output: it is how you tell that
a claimed figure is impossible without owning the device.

**3. The expected range.** 38% to 85% of the ceiling. That band contains all 16
measurements. It is wide, and the width is the finding, not a shortcoming.

## Why the range is wide, and why a narrower one would be a lie

I fitted an efficiency curve to the fourteen Apple measurements. In-sample it was
excellent: 14 of 14 within 30%, mean error 7%.

Then I ran it against two held-out platforms — a Raspberry Pi 5 and a Jetson Orin
Nano Super, both a different model size and a different quantisation scheme.
**It missed both by about +49%, in the same direction.**

So achieved efficiency is a property of a memory subsystem, not a function of the
bandwidth number. Apple's own parts range from 83% on a base M2 down to 40% on an
M1 Ultra, because a single decode stream cannot saturate two fused dies. A Pi and
a Jetson both land near 56%, nowhere near what the Apple curve predicts for their
bandwidth.

A point estimate would look more useful and be wrong about half the time. The
band is what the data supports.

**Run `--validate` before trusting any number this produces.** It prints the
scoring, every row, including the held-out ones.

## What decides the answer

**Bytes per token, which sets the ceiling:**

- **Parameter count times bits per weight.** The dominant term. Halving the bits
  roughly doubles the ceiling.
- **The key-value cache**, which grows with context length. For a 32-layer 7B
  model at 16-bit it costs about 0.5 MB per token, so a 4096-token context adds
  about 2 GB — both to the memory budget and to what must be read.
- **Grouped-query attention**, which shares key-value heads across query heads
  and divides the cache by 4 or 8. Pass `--gqa 0.25` when the model uses it. It
  is the single biggest factor in whether a long context fits, and it is easy to
  forget.

**Memory, which decides whether it runs at all:** weights, plus the cache, plus
about 1.2 GB for the runtime and the operating system. If it does not fit, the
throughput numbers are irrelevant, because swapping makes decode disk-bound and
costs far more than any of this arithmetic.

## Adding a device

`references/devices.json` carries bandwidth, memory options and a source. Adding
one means finding a real specification.

**Never add a device from memory.** A bandwidth figure invented by a language
model is worse than no tool, because it is confidently wrong in exactly the case
the user cannot check. The same rule governs the operator tables in
`npu-op-compat`, for the same reason.

If a device has a published measurement, add it to
`references/measurements.json` too, and re-run `--validate`. The validation set
is the only thing making any of this falsifiable.

## What this does not cover

- **Prompt processing.** Reading a prompt is compute bound and parallel, so it
  behaves nothing like decode. These numbers are for generation only.
- **Batching.** With several concurrent streams the weight read is amortised and
  the workload moves toward compute bound. Single stream only.
- **Accelerators that are not general processors.** A Hailo part on a Pi runs
  compiled vision networks, not language models. `npu-op-compat` is the tool for
  that hardware.
- **Sustained versus burst.** A fanless device throttles. Every measurement here
  is a benchmark figure, which is a good hour rather than a hot afternoon.

## What this skill will not do

- Give a point estimate it cannot support.
- Add a device specification that has no source.
- Claim a number is achievable because it is under the ceiling. The ceiling is an
  upper bound, not a prediction.
