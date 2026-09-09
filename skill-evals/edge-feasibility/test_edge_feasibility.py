#!/usr/bin/env python3
"""
Deterministic eval for the edge-feasibility estimator.

The important test is the last one: the estimator is scored against the 16
measured figures it ships, including two from held-out platforms. That is what
makes the tool falsifiable without owning any of the hardware.

Stdlib only, no network.

    python3 test_edge_feasibility.py      # exit 0 pass, 1 fail
"""
from __future__ import annotations
import importlib.util, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / ".claude/skills/edge-feasibility/scripts/feasibility.py"
REF = ROOT / ".claude/skills/edge-feasibility/references"

spec = importlib.util.spec_from_file_location("feasibility", TOOL)
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)

DEV = json.loads((REF / "devices.json").read_text(encoding="utf-8"))
MEAS = json.loads((REF / "measurements.json").read_text(encoding="utf-8"))


def run(args):
    r = subprocess.run([sys.executable, str(TOOL)] + args, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main() -> int:
    f: list[str] = []

    # ── Arithmetic ───────────────────────────────────────────────────────
    if abs(fs.weight_gb(7e9, 4.5) - 3.9375) > 0.01:
        f.append("weight size arithmetic is wrong for 7B at 4.5 bits")
    if abs(fs.weight_gb(7e9, 16) - 14.0) > 0.01:
        f.append("weight size arithmetic is wrong at 16 bits")
    if fs.parse_params("7B") != 7e9 or fs.parse_params("3.21b") != 3.21e9:
        f.append("parameter parsing is wrong")
    if fs.parse_params("500M") != 5e8:
        f.append("parameter parsing does not handle millions")

    # ── Key-value cache, against a hand-computed ground truth ────────────
    # LLaMA-7B: 32 layers, hidden 4096, 16-bit. Per token, K and V for every
    # layer: 2 * 32 * 4096 * 2 bytes = 524288 bytes.
    truth_per_token = 2 * 32 * 4096 * 2
    est_per_token = fs.kv_cache_gb(7e9, 1000, layers=32) * 1e9 / 1000
    err = abs(est_per_token - truth_per_token) / truth_per_token
    if err > 0.15:
        f.append(f"key-value cache per token off by {err:.0%} against a hand-computed "
                 f"7B figure (estimate {est_per_token/1e6:.2f} MB, truth "
                 f"{truth_per_token/1e6:.2f} MB)")
    # It must scale with context, and grouped-query attention must divide it.
    if not (fs.kv_cache_gb(7e9, 8192) > 1.9 * fs.kv_cache_gb(7e9, 4096) * 0.99):
        f.append("key-value cache does not scale linearly with context")
    if abs(fs.kv_cache_gb(7e9, 4096, gqa=0.25) - fs.kv_cache_gb(7e9, 4096) / 4) > 1e-6:
        f.append("grouped-query attention does not divide the cache")
    if fs.kv_cache_gb(7e9, 0) != 0.0:
        f.append("zero context should mean no cache")
    # A 4096-token 7B cache is about 2 GB, not megabytes. This is the bug the
    # first version had: a rule of thumb that was two orders of magnitude low.
    if not 1.5 <= fs.kv_cache_gb(7e9, 4096) <= 3.0:
        f.append(f"7B at 4096 context gives {fs.kv_cache_gb(7e9,4096):.2f} GB of cache; "
                 f"the real figure is about 2 GB")

    # Layer estimation should land on the known families.
    for params, expect in [(7e9, 32), (13e9, 40), (70e9, 80), (3e9, 28)]:
        got = fs.estimate_layers(params)
        if abs(got - expect) > 2:
            f.append(f"layer estimate for {params/1e9:.0f}B is {got}, expected ~{expect}")

    # ── Fit decision ─────────────────────────────────────────────────────
    code, out = run(["--params", "7B", "--quant", "q4_0", "--device", "m3", "--memory", "16"])
    if code != 0 or "FITS" not in out:
        f.append("a 7B Q4_0 should fit in 16 GB")
    code, out = run(["--params", "70B", "--quant", "q4_0", "--device", "m3", "--memory", "16"])
    if code == 0 or "DOES NOT FIT" not in out:
        f.append("a 70B Q4_0 must not fit in 16 GB")

    # ── The ceiling is a bound, not a prediction ─────────────────────────
    e = fs.estimate(7e9, 4.5, 100, 0)
    if abs(e["ceiling_tps"] - 100 / 3.9375) > 0.1:
        f.append("ceiling is not bandwidth divided by bytes per token")
    if not (e["low_tps"] < e["high_tps"] < e["ceiling_tps"]):
        f.append("the band is not strictly inside the ceiling")

    # ── Unknown device and unknown quantisation are refused ──────────────
    code, out = run(["--params", "7B", "--quant", "q4_0", "--device", "not-a-device"])
    if code != 2 or "unknown device" not in out:
        f.append("an unknown device was not refused")
    code, out = run(["--params", "7B", "--quant", "not-a-quant", "--device", "m3"])
    if code == 0:
        f.append("an unknown quantisation was not refused")

    # ── Provenance: no device may be here without a source ───────────────
    if not DEV.get("retrieved") or not DEV.get("sources"):
        f.append("devices.json has no retrieval date or sources block")
    for key, d in DEV["devices"].items():
        for field in ("name", "bandwidth_gbs", "memory_gb"):
            if field not in d:
                f.append(f"device {key!r} is missing {field!r}")
        if d.get("bandwidth_gbs", 0) <= 0:
            f.append(f"device {key!r} has a nonsense bandwidth")
    for row in MEAS["measurements"]:
        if row["device"] not in DEV["devices"]:
            f.append(f"measurement references unknown device {row['device']!r}")
        if row["model"] not in MEAS["models"]:
            f.append(f"measurement references unknown model {row['model']!r}")
        if not row.get("source"):
            f.append(f"measurement for {row['device']!r} has no source")

    # ── THE test: scored against every measurement it ships ──────────────
    lines, stats = fs.validate(DEV, MEAS)
    if stats["total"] < 12:
        f.append(f"only {stats['total']} measurements; the claim needs at least 12")
    if stats["under"] != stats["total"]:
        f.append(f"the ceiling was exceeded by {stats['total']-stats['under']} measurement(s). "
                 f"That would mean the physics is wrong, not the calibration.")
    if stats["inside"] != stats["total"]:
        f.append(f"only {stats['inside']}/{stats['total']} measurements fell inside the "
                 f"{fs.EFF_LO:.0%}-{fs.EFF_HI:.0%} band")
    held = [r for r in MEAS["measurements"] if r.get("held_out")]
    if len(held) < 2:
        f.append("fewer than two held-out measurements; without them the band is "
                 "fitted and tested on the same platform family")

    code, out = run(["--validate"])
    if code != 0 or "ceiling never exceeded" not in out:
        f.append("--validate did not report its scoring")

    code, out = run(["--devices"])
    if code != 0 or "GB/s" not in out:
        f.append("--devices did not list the device table")

    for x in f:
        print(f"FAIL  {x}")
    total = 25
    print(f"\nedge-feasibility: {total - len(f)}/{total} checks passed")
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())
