#!/usr/bin/env python3
"""
Deterministic eval for the npu-op-compat skill.

Exercises the protobuf parser against real ONNX binaries, the support
resolution against the shipped tables, and the partition arithmetic that is the
whole point of the tool. Stdlib only, no network.

    python3 test_npu_op_compat.py      # exit 0 pass, 1 fail
"""
from __future__ import annotations
import importlib.util, json, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOL = ROOT / ".claude/skills/npu-op-compat/scripts/check_ops.py"
REF = ROOT / ".claude/skills/npu-op-compat/references"
FIX = HERE / "fixtures"

spec = importlib.util.spec_from_file_location("check_ops", TOOL)
co = importlib.util.module_from_spec(spec)
spec.loader.exec_module(co)

TARGET = json.loads((REF / "edgetpu-ops.json").read_text())
MAPPING = json.loads((REF / "onnx-to-tflite.json").read_text())

# Ground truth from onnx.load(...).graph.node on the fixture. See fixtures/README.md.
CLEAN_OPS = ["Conv", "Relu", "Conv", "GlobalAveragePool", "Flatten", "Gemm", "Softmax"]


def run(args: list[str]):
    r = subprocess.run([sys.executable, str(TOOL)] + args, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main() -> int:
    f: list[str] = []

    # ── Parser: real binary in, exact op order out ────────────────────────
    got = co.onnx_ops(FIX / "clean_cnn.onnx")
    if got != CLEAN_OPS:
        f.append(f"ONNX parser returned {got}, expected {CLEAN_OPS}")

    ops2 = co.onnx_ops(FIX / "leakyrelu_backbone.onnx")
    if ops2.count("LeakyRelu") != 7 or ops2[0] != "Conv":
        f.append(f"ONNX parser mis-read the backbone fixture: {ops2}")

    # A file that is not ONNX must raise, not return nonsense.
    with tempfile.TemporaryDirectory() as td:
        junk = Path(td) / "j.onnx"
        junk.write_bytes(b"not a protobuf at all, not even close")
        try:
            co.onnx_ops(junk)
            f.append("a non-ONNX file did not raise")
        except Exception:
            pass

    # ── Support resolution ────────────────────────────────────────────────
    checks = [
        ("Conv", "supported", "Conv2d"),
        ("Relu", "supported", "ReLU"),
        ("LeakyRelu", "unsupported", None),      # known-unsupported, with a reason
        ("Div", "unsupported", None),
        ("Gather", "unsupported", None),
        ("Resize", "unknown", None),             # mode-dependent, must not guess
        ("Clip", "unknown", None),               # bounds-dependent
        ("BatchNormalization", "unknown", None), # folding-dependent
        ("Conv2d", "supported", "Conv2d"),       # a native TFLite name
        ("ThisOpDoesNotExist", "unknown", None),
    ]
    for op, want, as_ in checks:
        r = co.resolve(op, TARGET, MAPPING, None)
        if r["verdict"] != want:
            f.append(f"resolve({op}) = {r['verdict']}, expected {want}")
        elif want == "unsupported" and not r["why"]:
            f.append(f"resolve({op}) is unsupported but gives no reason")
        elif as_ and r["as"] != as_:
            f.append(f"resolve({op}) mapped to {r['as']}, expected {as_}")

    # Version gating: Transpose needs runtime >=14.
    if co.resolve("Transpose", TARGET, MAPPING, 13)["verdict"] != "unsupported":
        f.append("Transpose at runtime 13 should be unsupported (needs >=14)")
    if co.resolve("Transpose", TARGET, MAPPING, 14)["verdict"] != "supported":
        f.append("Transpose at runtime 14 should be supported")

    # ── Partition arithmetic, the actual product ──────────────────────────
    a = co.analyse(CLEAN_OPS, TARGET, MAPPING, None)
    if a["cut_index"] is not None:
        f.append(f"a fully supported graph reported a cut at {a['cut_index']}")
    if a["ops_on_cpu"] != 0 or a["cpu_fraction"] != 0.0:
        f.append("a fully supported graph reported ops on CPU")

    a = co.analyse(ops2, TARGET, MAPPING, None)
    if a["cut_index"] != 1:
        f.append(f"backbone cut at index {a['cut_index']}, expected 1")
    if a["ops_on_cpu"] != len(ops2) - 1:
        f.append(f"backbone stranded {a['ops_on_cpu']} ops, expected {len(ops2) - 1}")
    if not 0.90 <= a["cpu_fraction"] <= 0.95:
        f.append(f"backbone cpu_fraction {a['cpu_fraction']}, expected ~0.93")

    # The cut is the FIRST unsupported op, not the last and not the most common.
    mixed = ["Conv", "Relu", "Div", "Conv", "Gather", "Softmax"]
    a = co.analyse(mixed, TARGET, MAPPING, None)
    if a["cut_index"] != 2 or a["cut_op"] != "Div":
        f.append(f"mixed graph cut at {a['cut_index']}/{a['cut_op']}, expected 2/Div")

    # A single late unsupported op must strand far less than an early one.
    early = co.analyse(["Div"] + ["Conv"] * 9, TARGET, MAPPING, None)
    late = co.analyse(["Conv"] * 9 + ["Div"], TARGET, MAPPING, None)
    if not early["cpu_fraction"] > late["cpu_fraction"]:
        f.append("position does not dominate: an early cut must strand more than a late one")

    # ── CLI contract ──────────────────────────────────────────────────────
    code, out = run([str(FIX / "clean_cnn.onnx")])
    if code != 0:
        f.append("clean fixture should exit 0")
    if "No unsupported operation" not in out:
        f.append("clean fixture output missing the all-clear line")

    code, out = run([str(FIX / "leakyrelu_backbone.onnx")])
    if code != 1:
        f.append("cut fixture should exit 1 so CI can gate on it")
    # Both ends of the published comparison, not just the loud one.
    #
    # The end-position figure shipped as 10% and stayed there through an
    # article, a panel and the ledger, because nothing asserted it. No position
    # in a 15-operator graph produces 10%: a single unsupported operator last
    # strands 1 of 15, which is 6.7%.
    import os as _os
    for pos, want in ((1, 93.3), (14, 6.7)):
        ops = ["Conv"] * 15
        ops[pos] = "LeakyRelu"
        fd, path = tempfile.mkstemp(suffix=".json")
        with _os.fdopen(fd, "w") as fh:
            json.dump(ops, fh)
        _code, _out = run([path, "--format", "list", "--json"])
        _os.unlink(path)
        # run() merges stderr into stdout, so take the object, not the noise.
        _got = round(json.loads(_out[_out.index("{"):])["cpu_fraction"] * 100, 1)
        if _got != want:
            f.append(f"a LeakyRelu at index {pos} of 15 strands {_got}%, "
                     f"but the ledger publishes {want}%")

    for expect in ["partition cut", "LeakyRelu", "93%"]:
        if expect not in out:
            f.append(f"cut fixture output missing {expect!r}")

    code, out = run([str(FIX / "leakyrelu_backbone.onnx"), "--json"])
    if code == 1:
        try:
            j = json.loads(out)
            if j["cut_index"] != 1:
                f.append("--json reported the wrong cut index")
        except json.JSONDecodeError:
            f.append("--json did not emit valid JSON")

    code, out = run([str(FIX / "nope.onnx")])
    if code != 2:
        f.append("a missing file should exit 2")

    # ── Data integrity ────────────────────────────────────────────────────
    if len(TARGET["ops"]) != 40:
        f.append(f"edgetpu table has {len(TARGET['ops'])} ops, expected 40 from Table 1")
    for key in ("source", "retrieved", "partition_rule"):
        if not TARGET.get(key):
            f.append(f"edgetpu table is missing provenance field {key!r}")
    for op, tfl in MAPPING["map"].items():
        if tfl not in TARGET["ops"]:
            f.append(f"mapping sends {op} to {tfl}, which is not in the support table")

    for x in f:
        print(f"FAIL  {x}")
    total = 26
    print(f"\nnpu-op-compat: {total - len(f)}/{total} checks passed")
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())
