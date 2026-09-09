#!/usr/bin/env bash
# Check every number these two tools are quoted as producing.
#
#   ./verify.sh
#
# Runs from a clean clone with Python 3 and nothing else. Each section prints a
# figure and says where it is quoted, so any claim made about these tools can be
# checked without taking anyone's word for it.
#
# Steps needing an optional package say so and skip rather than failing.

set -u
cd "$(dirname "$0")"
pass=0; fail=0; skip=0

hr() { printf '%s\n' "------------------------------------------------------------------------"; }
ok() { printf '  PASS  %s\n' "$1"; pass=$((pass+1)); }
no() { printf '  FAIL  %s\n' "$1"; fail=$((fail+1)); }
sk() { printf '  SKIP  %s\n' "$1"; skip=$((skip+1)); }

echo "CHECKING EVERY PUBLISHED FIGURE"
echo "repository: $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')   date: $(date -u +%Y-%m-%d)"

# ── Where a graph stops being accelerated ────────────────────────────────────
hr
echo "ONE LAYER DECIDES WHETHER YOUR MODEL USES THE ACCELERATOR"
echo
echo "  A detection network with LeakyReLU activations, exported to ONNX with"
echo "  PyTorch. The figure quoted is 93 percent."
echo
OUT=$(python3 .claude/skills/npu-op-compat/scripts/check_ops.py \
        skill-evals/npu-op-compat/fixtures/leakyrelu_backbone.onnx 2>&1)
echo "$OUT" | grep -E "First unsupported|operations accelerate" | sed 's/^/  /'
echo
echo "$OUT" | grep -q "93% of the graph" && ok "93 percent reproduces" || no "93 percent did NOT reproduce"
echo "$OUT" | grep -q "LeakyRelu" && ok "the cut lands on LeakyRelu" || no "cut operation changed"

# ── Reading a model file with nothing installed ──────────────────────────────
hr
echo "READING A MODEL FILE WITH NOTHING INSTALLED"
echo
echo "  The claim is that a model's layer list can be read with no third-party"
echo "  package, matching the official library exactly. Comparing the two needs"
echo "  the official library, which is the only place it is used."
echo
if python3 -c "import onnx" 2>/dev/null; then
  python3 - <<'PY'
import onnx, importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("c", ".claude/skills/npu-op-compat/scripts/check_ops.py")
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
bad = 0
for name in ("clean_cnn", "leakyrelu_backbone"):
    p = Path(f"skill-evals/npu-op-compat/fixtures/{name}.onnx")
    mine = c.onnx_ops(p)
    theirs = [n.op_type for n in onnx.load(p).graph.node]
    print(f"    {name:<22} stdlib {len(mine):>2} nodes   onnx {len(theirs):>2} nodes   "
          f"{'identical' if mine == theirs else 'DIFFER'}")
    bad += mine != theirs
sys.exit(1 if bad else 0)
PY
  [ $? -eq 0 ] && ok "the zero-dependency reader matches onnx.load exactly" \
                || no "the readers disagree"
else
  sk "needs the optional onnx package: pip install onnx"
fi

# ── The bandwidth ceiling ────────────────────────────────────────────────────
hr
echo "ON-DEVICE SPEED IS A BANDWIDTH PROBLEM"
echo
echo "  Scored against 16 measured figures published by other people, across"
echo "  three platform families, two of them held out entirely."
echo
OUT=$(python3 .claude/skills/edge-feasibility/scripts/feasibility.py --validate 2>&1)
echo "$OUT" | grep -E "ceiling never exceeded|inside the band|efficiency range" | sed 's/^/  /'
echo
echo "$OUT" | grep -q "ceiling never exceeded:      16/16" && ok "16/16 under the ceiling" \
                                                           || no "a measurement broke the ceiling"
echo "$OUT" | grep -q "inside the band: 16/16" && ok "16/16 inside the band" \
                                               || no "a measurement fell outside the band"

# ── The tools themselves ─────────────────────────────────────────────────────
hr
echo "THE TOOLS  every skill's own test suite"
echo
for t in skill-evals/*/test_*.py; do
  [ -e "$t" ] || continue
  skill=$(basename "$(dirname "$t")")
  stem=$(basename "$t" .py)
  if [ "$stem" = "test_${skill//-/_}" ]; then name="$skill"; else name="$skill / ${stem#test_}"; fi
  if python3 "$t" >/dev/null 2>&1; then ok "$name"; else no "$name"; fi
done

# ── Nothing here reaches the network or your credentials ─────────────────────
hr
echo "THE SKILLS  scanned before you install them"
echo
if python3 skill-evals/tools/skill_scanner.py .claude/skills >/dev/null 2>&1; then
  ok "no install lures, network calls, or credential access"
else
  no "the security scan found something"
fi

hr
printf 'PASS %d   FAIL %d   SKIP %d\n' "$pass" "$fail" "$skip"
echo
if [ "$fail" -eq 0 ]; then
  echo "Every figure above regenerated from this clone."
else
  echo "Something no longer reproduces. The claim is wrong, or the code is."
fi
exit "$fail"
