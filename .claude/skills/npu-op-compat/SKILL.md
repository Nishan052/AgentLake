---
name: npu-op-compat
description: >-
  Find where a neural network stops being accelerated on an edge NPU, by
  checking every operator in graph order against the accelerator's published
  supported-operator table. Use when someone asks why their model is slow on an
  Edge TPU, Coral, or NPU accelerator, whether a model will compile or convert
  for one, which layers fall back to CPU, what an unsupported operator costs, or
  before porting an ONNX or TFLite network to edge hardware. Reports the
  partition cut point and the fraction of the graph stranded on CPU behind it.
license: MIT
metadata:
  author: "Nishan Poojary"
  version: "1.0.0"
  source: "https://github.com/Nishan052/AgentLake"
---

# NPU Operator Compatibility

The usual question is "how many of my operators are unsupported". For the Edge
TPU that question is close to meaningless, because of one line in Google's
documentation:

> At the first point in the model graph where an unsupported operation occurs,
> the compiler partitions the graph into two parts... Currently, the Edge TPU
> compiler cannot partition the model more than once.

Everything after the first unsupported operator runs on the CPU, however well
supported those later operators are. **Position dominates count.**

A YOLO-shaped backbone with LeakyReLU activations puts 93% of its graph on the
CPU, because the second operator in the graph has no Edge TPU equivalent. One
substitution recovers almost all of it. Counting unsupported operators would
have told you "half my activations are unsupported" and sent you off to fix all
of them, which is the wrong day's work.

## Steps

### 1. Get the operator list

```bash
python3 .claude/skills/npu-op-compat/scripts/check_ops.py model.onnx
```

ONNX files are parsed directly, with a minimal protobuf wire-format walk. No
`onnx` package, no `protobuf`, no `numpy`. The file is read, never executed.

For a graph from anywhere else (a TFLite dump, a Netron export, a compiler log),
pass a whitespace-separated or JSON list of operator names in graph order and
add `--format list`.

Add `--runtime 13` if you know the target runtime version, since a few
operators are version gated. Add `--json` to pipe the result somewhere. Add
`--verbose` to print the limitations attached to supported operators too.

### 2. Read the cut, not the count

The output marks one operator `<-- partition cut`. That is the whole finding.
Report to the user, in this order:

1. Which operator cuts the graph, and at what index.
2. What fraction of the graph is stranded on CPU behind it.
3. What that one operator could be replaced with.

Do not lead with a list of every unsupported operator. That is the framing this
tool exists to correct.

### 3. Recommend the substitution

Common cut points and what actually fixes them:

| Cut operator | Why | Usual fix |
|---|---|---|
| `LeakyRelu` | Not in the table. PReLU is, LeakyRelu is not. | Retrain or fine-tune with ReLU6, or convert to PReLU with a 1-D alpha |
| `Div` | Add, Sub and Mul are supported, Div is not | Multiply by a reciprocal constant folded at export |
| `Gather` | Not in the table. Embedding lookups, NMS post-processing | Move the post-processing out of the graph and run it in host code |
| `NonMaxSuppression`, `TopK` | Not in the table | Split the detection head off. Let the NPU run the backbone and do NMS on CPU deliberately, rather than by accident |
| `BatchNormalization` | Should fold into the preceding Conv | It survived folding. Fix the export, do not fight the compiler |
| `Sqrt` | Rsqrt is supported, Sqrt is not | Reciprocal of Rsqrt, or fold into an adjacent scale |

When the cut is in post-processing, the answer is usually to **cut the graph
there on purpose**: export the backbone alone, let it map fully, and do the
head in host code. That is a different and much better outcome than discovering
the same split by accident with everything after it on CPU.

### 4. State the caveats, every time

This is a pre-conversion estimate, not the compiler's verdict. Say so.

- **ONNX is not TFLite.** The converter fuses, folds and rewrites, so the final
  graph is not a one-to-one image of the ONNX graph. Treat the cut as a strong
  prediction, not a guarantee.
- **Operators reported `unknown` are not guesses.** `Resize`, `Clip` and
  `BatchNormalization` cannot be resolved from an operator name alone. If one
  sits before the reported cut, it may be the real cut point. Say that plainly.
- **The count is operators, not FLOPs.** "93% of the graph" means 93% of
  operators, not 93% of the compute. A cut before the heavy convolutions is
  worse than the number suggests; a cut after them is better.
- **The Edge TPU needs full integer quantisation.** A float32 model maps
  nothing at all, whatever this tool says about its operators.
- **The data has a date.** `references/edgetpu-ops.json` records what Google's
  table said on the date in the file. Vendor support changes with compiler
  releases. If the answer matters, re-fetch the source and diff it.

## Data

| File | Contents | Source |
|---|---|---|
| `references/edgetpu-ops.json` | 40 operators with runtime requirements and verbatim limitations, plus the single-partition rule | [coral.ai/docs/edgetpu/models-intro](https://coral.ai/docs/edgetpu/models-intro/), Table 1 |
| `references/onnx-to-tflite.json` | 35 ONNX to TFLite name mappings, 3 unresolvable, 11 known-unsupported with reasons | Derived from the above plus the ONNX operator spec |

Every entry carries a retrieval date. Adding an accelerator means adding a
`<target>-ops.json` in the same shape, with a real source URL. **Do not add a
target from memory.** An operator table invented by a language model is worse
than no tool, because it is confidently wrong in exactly the cases the user
cannot check.

## What this skill will not do

- Claim to be the compiler. It predicts; `edgetpu_compiler` decides.
- Guess at an operator it cannot resolve, or silently treat unknown as supported.
- Report an operator table for hardware whose documentation is not publicly
  available. Hailo's full operator list sits behind the Developer Zone login, so
  it is not shipped here rather than reconstructed from forum posts.
- Lead with a count of unsupported operators.
