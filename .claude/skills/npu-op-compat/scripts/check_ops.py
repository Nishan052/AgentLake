#!/usr/bin/env python3
"""
check_ops.py - find where a model stops being accelerated on an Edge TPU.

    python3 check_ops.py model.onnx
    python3 check_ops.py ops.txt --format list
    python3 check_ops.py model.onnx --runtime 13 --json

The Edge TPU compiler partitions a graph exactly once, at the first unsupported
operation. Everything after that point runs on the CPU no matter how well
supported it is. So the question worth answering is not "how many of my
operations are unsupported" but "where is the first one", and this tool answers
that.

ONNX files are parsed with a minimal protobuf wire-format walk. No onnx, no
protobuf, no numpy: Python 3 stdlib only, no network, and the file is read, not
executed.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REF = Path(__file__).resolve().parent.parent / "references"


# ─── Minimal protobuf reader ─────────────────────────────────────────────────
# Only what is needed to walk ModelProto.graph(7).node(1).op_type(4). Repeated
# fields keep wire order, so the node list comes out in graph order.

def _varint(b: bytes, i: int) -> tuple[int, int]:
    v = s = 0
    while True:
        if i >= len(b):
            raise ValueError("truncated varint")
        x = b[i]; i += 1
        v |= (x & 0x7F) << s
        if not x & 0x80:
            return v, i
        s += 7


def _fields(b: bytes):
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(b, i)
            v, i = b[i:i + ln], i + ln
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield fn, wt, v


def _first(b: bytes, num: int):
    for fn, wt, v in _fields(b):
        if fn == num and wt == 2:
            return v
    return None


def onnx_ops(path: Path) -> list[str]:
    """Op types of every node, in graph order."""
    data = path.read_bytes()
    graph = _first(data, 7)
    if graph is None:
        raise ValueError("no GraphProto (field 7): is this really an ONNX file?")
    ops = []
    for fn, wt, v in _fields(graph):
        if fn == 1 and wt == 2:
            t = _first(v, 4)
            if t is not None:
                ops.append(t.decode("utf-8", "replace"))
    if not ops:
        raise ValueError("parsed an ONNX graph but found no nodes")
    return ops


# ─── Support resolution ──────────────────────────────────────────────────────

class Verdict:
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


def runtime_ok(spec: str, runtime: int | None) -> bool:
    if spec == "all" or runtime is None:
        return True
    if spec.startswith(">="):
        return runtime >= int(spec[2:])
    return True


def resolve(op: str, target: dict, mapping: dict, runtime: int | None) -> dict:
    """Classify one op name. Never guesses: ambiguity resolves to UNKNOWN."""
    ops = target["ops"]

    # A TFLite name already in the table.
    if op in ops:
        entry = ops[op]
        if not runtime_ok(entry["runtime"], runtime):
            return {"verdict": Verdict.UNSUPPORTED, "as": op,
                    "why": f"needs runtime {entry['runtime']}, you specified {runtime}"}
        return {"verdict": Verdict.SUPPORTED, "as": op, "why": entry["limits"]}

    # A known-unsupported ONNX op, with the reason recorded.
    if op in mapping["known_unsupported"]:
        return {"verdict": Verdict.UNSUPPORTED, "as": None,
                "why": mapping["known_unsupported"][op]}

    # An ONNX op that cannot be resolved from its name alone.
    if op in mapping["depends"]:
        return {"verdict": Verdict.UNKNOWN, "as": None,
                "why": mapping["depends"][op]}

    # A mapped ONNX op.
    if op in mapping["map"]:
        tfl = mapping["map"][op]
        entry = ops.get(tfl)
        if entry is None:
            return {"verdict": Verdict.UNSUPPORTED, "as": tfl,
                    "why": f"maps to {tfl}, which is not in the {target['target']} table"}
        if not runtime_ok(entry["runtime"], runtime):
            return {"verdict": Verdict.UNSUPPORTED, "as": tfl,
                    "why": f"maps to {tfl}, which needs runtime {entry['runtime']}"}
        return {"verdict": Verdict.SUPPORTED, "as": tfl, "why": entry["limits"]}

    return {"verdict": Verdict.UNKNOWN, "as": None,
            "why": "not in the support table and not in the ONNX mapping"}


def analyse(ops: list[str], target: dict, mapping: dict, runtime: int | None) -> dict:
    rows = []
    cut = None
    for i, op in enumerate(ops):
        r = resolve(op, target, mapping, runtime)
        r["index"], r["op"] = i, op
        rows.append(r)
        if cut is None and r["verdict"] == Verdict.UNSUPPORTED:
            cut = i

    total = len(ops)
    on_cpu = total - cut if cut is not None else 0
    return {
        "target": target["target"],
        "total_ops": total,
        "cut_index": cut,
        "cut_op": ops[cut] if cut is not None else None,
        "ops_on_tpu": cut if cut is not None else total,
        "ops_on_cpu": on_cpu,
        "cpu_fraction": round(on_cpu / total, 3) if total else 0.0,
        "rows": rows,
    }


# ─── Reporting ───────────────────────────────────────────────────────────────

def report(a: dict, target: dict, verbose: bool) -> None:
    print(f"target: {target['display_name']}   source: {target['source']} "
          f"(retrieved {target['retrieved']})")
    print(f"{a['total_ops']} operations in graph order\n")

    for r in a["rows"]:
        mark = {"supported": "ok  ", "unsupported": "STOP", "unknown": "?   "}[r["verdict"]]
        cut = "  <-- partition cut" if r["index"] == a["cut_index"] else ""
        as_ = f" -> {r['as']}" if r["as"] and r["as"] != r["op"] else ""
        after = a["cut_index"] is not None and r["index"] > a["cut_index"]
        tail = "   (on CPU: after the cut)" if after else ""
        print(f"  {mark} [{r['index']:>3}] {r['op']}{as_}{cut}{tail}")
        if r["why"] and (verbose or r["verdict"] != Verdict.SUPPORTED
                         or r["index"] == a["cut_index"]):
            print(f"         {r['why']}")

    print()
    if a["cut_index"] is None:
        print("No unsupported operation found. The whole graph can map to the accelerator,")
        print("subject to the per-operation limitations printed above.")
    else:
        print(f"First unsupported operation: {a['cut_op']} at index {a['cut_index']}.")
        print(f"{a['ops_on_tpu']} operations accelerate. "
              f"{a['ops_on_cpu']} run on CPU ({a['cpu_fraction'] * 100:.0f}% of the graph).")
        print()
        print(target["partition_rule"]["consequence"])
        print()
        print("Moving or replacing that single operation is worth more than fixing every")
        print("other unsupported operation in the model combined.")

    unknown = [r for r in a["rows"] if r["verdict"] == Verdict.UNKNOWN]
    if unknown:
        print(f"\n{len(unknown)} operation(s) could not be resolved from the op name alone.")
        print("They are reported as unknown rather than guessed. If one sits before the cut,")
        print("it could become the real cut point.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help=".onnx file, or a text/JSON list of op names")
    ap.add_argument("--format", choices=["auto", "onnx", "list"], default="auto")
    ap.add_argument("--target", default="edgetpu")
    ap.add_argument("--runtime", type=int, default=None,
                    help="accelerator runtime version, e.g. 13")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="print limitations for supported ops too")
    args = ap.parse_args()

    path = Path(args.model)
    if not path.exists():
        print(f"ERROR  no file at {path}", file=sys.stderr)
        return 2

    target = json.loads((REF / f"{args.target}-ops.json").read_text())
    mapping = json.loads((REF / "onnx-to-tflite.json").read_text())

    fmt = args.format
    if fmt == "auto":
        fmt = "onnx" if path.suffix.lower() == ".onnx" else "list"

    try:
        if fmt == "onnx":
            ops = onnx_ops(path)
        else:
            raw = path.read_text()
            ops = json.loads(raw) if raw.lstrip().startswith("[") else raw.split()
    except Exception as e:
        print(f"ERROR  could not read operations: {e}", file=sys.stderr)
        return 2

    a = analyse(ops, target, mapping, args.runtime)

    if args.json:
        print(json.dumps(a, indent=2))
    else:
        report(a, target, args.verbose)

    # Exit 1 when the graph is cut, so this can gate CI on a model change.
    return 1 if a["cut_index"] is not None else 0


if __name__ == "__main__":
    sys.exit(main())
