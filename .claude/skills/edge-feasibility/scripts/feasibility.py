#!/usr/bin/env python3
"""
feasibility.py - will this model fit on this device, and roughly how fast.

    python3 feasibility.py --params 7B --quant q4_0 --device m3
    python3 feasibility.py --params 3.21B --bits 4.85 --device pi5 --context 4096
    python3 feasibility.py --devices                # list what is known
    python3 feasibility.py --validate               # score against published data

Single-stream decode is memory-bandwidth bound: every generated token requires
reading the whole weight set plus the key-value cache, and the arithmetic per
byte read is tiny. So throughput is bytes-per-second divided by bytes-per-token,
and the compute rating of the chip barely matters.

The tool reports three things, in decreasing order of how much they can be
trusted:

  1. Does it fit.        Arithmetic. Either it does or it does not.
  2. The ceiling.        bandwidth / bytes-per-token. A physical upper bound,
                         not exceeded by any of the 16 published measurements
                         shipped in references/measurements.json.
  3. An expected range.  38% to 85% of the ceiling. That band contains all 16.
                         It is wide because achieved efficiency is a property of
                         a memory subsystem, and no single curve fitted to one
                         platform transferred to another.

Python 3 stdlib only. No network. Exit 0 clean, 1 if the model does not fit.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

REF = Path(__file__).resolve().parent.parent / "references"

# Validated band: the observed efficiency across all 16 published measurements
# ran 39.7% (M1 Ultra) to 83.1% (M2). Rounded outwards to contain all of them.
EFF_LO, EFF_HI = 0.38, 0.85

# Bits per weight for the common schemes. Q4_0 packs 32 weights into 18 bytes.
QUANTS = {
    "f32": 32.0, "f16": 16.0, "bf16": 16.0, "q8_0": 8.5,
    "q6_k": 6.6, "q5_k_m": 5.7, "q5_0": 5.5,
    "q4_k_m": 4.85, "q4_0": 4.5, "q3_k_m": 3.9, "q2_k": 3.0,
}

# Working memory beyond the weights: activations, the runtime, the operating
# system. A rule of thumb, and flagged as one.
OVERHEAD_GB = 1.2


def parse_params(s: str) -> float:
    m = re.fullmatch(r"([\d.]+)\s*([bmBM])?", s.strip())
    if not m:
        raise ValueError(f"cannot read a parameter count from {s!r}")
    n = float(m.group(1))
    suffix = (m.group(2) or "b").lower()
    return n * (1e9 if suffix == "b" else 1e6)


def weight_gb(params: float, bits: float) -> float:
    return params * bits / 8 / 1e9


# Layer counts for the common decoder families, used to estimate hidden width
# from a parameter count. Interpolated on a log scale between these anchors.
LAYER_ANCHORS = [(1e9, 16), (3e9, 28), (7e9, 32), (8e9, 32),
                 (13e9, 40), (30e9, 60), (70e9, 80), (180e9, 96)]


def estimate_layers(params: float) -> int:
    """Layer count for a parameter count, interpolated between known families."""
    if params <= LAYER_ANCHORS[0][0]:
        return LAYER_ANCHORS[0][1]
    if params >= LAYER_ANCHORS[-1][0]:
        return LAYER_ANCHORS[-1][1]
    for (p0, l0), (p1, l1) in zip(LAYER_ANCHORS, LAYER_ANCHORS[1:]):
        if p0 <= params <= p1:
            import math
            t = (math.log(params) - math.log(p0)) / (math.log(p1) - math.log(p0))
            return round(l0 + t * (l1 - l0))
    return 32


def kv_cache_gb(params: float, context: int, bits: float = 16.0,
                layers: int | None = None, gqa: float = 1.0) -> float:
    """
    Key-value cache size.

    Per token the cache holds a key and a value vector for every layer:

        bytes/token = 2 * layers * hidden * bytes_per_element

    Hidden width is recovered from the parameter count using the standard
    transformer relation params ~= 12 * layers * hidden^2, which gives

        bytes/token = (4 / sqrt(12)) * sqrt(layers * params)   at 16-bit

    That returns about 0.52 MB per token for a 32-layer 7B model, which is the
    right order: 4096 tokens of context costs roughly 2 GB, not the megabytes a
    cruder rule of thumb suggests.

    `gqa` is the ratio of key-value heads to query heads. Modern models share
    key-value heads across query heads, which divides the cache by 4 or 8 and is
    the single biggest factor in whether a long context fits.
    """
    import math
    if context <= 0:
        return 0.0
    L = layers if layers else estimate_layers(params)
    bytes_per_token_16 = (4 / math.sqrt(12)) * math.sqrt(L * params)
    return bytes_per_token_16 * (bits / 16.0) * gqa * context / 1e9


def estimate(params: float, bits: float, bandwidth: float, context: int,
             layers: int | None = None, gqa: float = 1.0):
    w = weight_gb(params, bits)
    kv = kv_cache_gb(params, context, layers=layers, gqa=gqa)
    per_token = w + kv
    ceiling = bandwidth / per_token
    return {
        "weights_gb": w,
        "kv_gb": kv,
        "bytes_per_token_gb": per_token,
        "ceiling_tps": ceiling,
        "low_tps": ceiling * EFF_LO,
        "high_tps": ceiling * EFF_HI,
    }


# ─── Validation ──────────────────────────────────────────────────────────────

def validate(devices: dict, meas: dict) -> tuple[list[str], dict]:
    """Score the estimator against every published measurement it ships."""
    lines, inside, under, total = [], 0, 0, 0
    effs = []

    lines.append(f"{'device':<18} {'model':<22} {'meas':>7} {'ceiling':>8} {'eff':>6}  {'in band':>7}")
    for row in meas["measurements"]:
        d = devices["devices"][row["device"]]
        m = meas["models"][row["model"]]
        est = estimate(m["params"], m["bits_per_weight"], d["bandwidth_gbs"], 0)
        eff = row["tokens_per_sec"] / est["ceiling_tps"]
        ok_band = EFF_LO <= eff <= EFF_HI
        ok_ceil = row["tokens_per_sec"] <= est["ceiling_tps"]
        inside += ok_band
        under += ok_ceil
        total += 1
        effs.append(eff)
        tag = "held out" if row.get("held_out") else ""
        lines.append(f"{d['name']:<18} {row['model']:<22} {row['tokens_per_sec']:>7.2f} "
                     f"{est['ceiling_tps']:>8.1f} {eff:>5.0%}  {'yes' if ok_band else 'NO':>7}  {tag}")

    return lines, {"inside": inside, "under": under, "total": total,
                   "eff_min": min(effs), "eff_max": max(effs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", help="parameter count, e.g. 7B or 3.21B")
    ap.add_argument("--quant", help=f"one of: {', '.join(QUANTS)}")
    ap.add_argument("--bits", type=float, help="bits per weight, if not a named scheme")
    ap.add_argument("--device", help="device key; see --devices")
    ap.add_argument("--memory", type=float, help="device memory in GB, if not the default")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--layers", type=int, help="layer count, if known. Otherwise estimated.")
    ap.add_argument("--gqa", type=float, default=1.0,
                    help="ratio of key-value heads to query heads, e.g. 0.25 for "
                         "8 key-value heads against 32 query heads. Divides the cache.")
    ap.add_argument("--devices", action="store_true", help="list known devices")
    ap.add_argument("--validate", action="store_true", help="score against published data")
    args = ap.parse_args()

    devices = json.loads((REF / "devices.json").read_text(encoding="utf-8"))
    meas = json.loads((REF / "measurements.json").read_text(encoding="utf-8"))

    if args.devices:
        print(f"{'key':<18} {'device':<30} {'GB/s':>7}  memory GB")
        for k, d in devices["devices"].items():
            print(f"{k:<18} {d['name']:<30} {d['bandwidth_gbs']:>7} "
                  f" {', '.join(str(x) for x in d['memory_gb'])}")
        print(f"\nSpecs retrieved {devices['retrieved']}. Adding a device needs a real source.")
        return 0

    if args.validate:
        lines, s = validate(devices, meas)
        print("VALIDATION against published measurements")
        print("=" * 78)
        for l in lines:
            print(l)
        print("=" * 78)
        print(f"\n  ceiling never exceeded:      {s['under']}/{s['total']}")
        print(f"  measurement inside the band: {s['inside']}/{s['total']}"
              f"   (band {EFF_LO:.0%}-{EFF_HI:.0%} of ceiling)")
        print(f"  observed efficiency range:   {s['eff_min']:.0%} to {s['eff_max']:.0%}")
        print("\n  The band is wide on purpose. An efficiency curve fitted to the Apple")
        print("  rows missed both held-out platforms by about +49%, so achieved")
        print("  efficiency is a property of a memory subsystem rather than a function")
        print("  of the bandwidth number. A narrower band would be a nicer answer and")
        print("  a false one.")
        return 0 if s["under"] == s["total"] else 1

    if not (args.params and args.device):
        ap.error("need --params and --device (or --devices / --validate)")

    if args.device not in devices["devices"]:
        print(f"ERROR  unknown device {args.device!r}. Run --devices to list them, and add "
              f"new ones with a real source.", file=sys.stderr)
        return 2

    bits = args.bits
    if bits is None:
        if not args.quant:
            ap.error("need --quant or --bits")
        if args.quant.lower() not in QUANTS:
            ap.error(f"unknown quantisation {args.quant!r}. Known: {', '.join(QUANTS)}")
        bits = QUANTS[args.quant.lower()]

    params = parse_params(args.params)
    d = devices["devices"][args.device]
    mem = args.memory if args.memory else max(d["memory_gb"])
    est = estimate(params, bits, d["bandwidth_gbs"], args.context,
                   layers=args.layers, gqa=args.gqa)
    layers = args.layers or estimate_layers(params)

    print(f"{d['name']}  ({d['bandwidth_gbs']} GB/s, {mem} GB)")
    print(f"{params/1e9:.2f}B parameters at {bits} bits per weight, {args.context} token context\n")

    # 1. Fit.
    need = est["weights_gb"] + est["kv_gb"] + OVERHEAD_GB
    fits = need <= mem
    print(f"  weights           {est['weights_gb']:>7.2f} GB")
    gqa_note = f", key-value head ratio {args.gqa:g}" if args.gqa != 1.0 else ""
    print(f"  key-value cache   {est['kv_gb']:>7.2f} GB   "
          f"({layers} layers{' estimated' if not args.layers else ''}{gqa_note})")
    print(f"  runtime overhead  {OVERHEAD_GB:>7.2f} GB   (rule of thumb)")
    print(f"  {'-'*34}")
    print(f"  total             {need:>7.2f} GB  of {mem} GB   -> {'FITS' if fits else 'DOES NOT FIT'}\n")

    # 2. Ceiling, and 3. band.
    print(f"  ceiling           {est['ceiling_tps']:>7.1f} tokens/sec   "
          f"(bandwidth / bytes per token)")
    print(f"  expected          {est['low_tps']:>7.1f} to {est['high_tps']:.1f} tokens/sec"
          f"   ({EFF_LO:.0%}-{EFF_HI:.0%} of ceiling)\n")

    if d.get("caution"):
        print(f"  CAUTION  {d['caution']}\n")

    if not fits:
        over = need - mem
        print(f"  Short by {over:.2f} GB. Options, cheapest first: quantise further, "
              f"cut the context\n  window, or use a smaller model. Swapping to disk "
              f"makes decode disk-bound and\n  costs far more than the numbers above.")

    print("  The ceiling is a physical bound and was not exceeded by any of the 16")
    print("  published measurements this tool ships. The range is empirical and wide:")
    print("  run --validate to see the scoring, including where it fails.")

    return 0 if fits else 1


if __name__ == "__main__":
    sys.exit(main())
