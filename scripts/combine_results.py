"""Combine per-method result CSVs into one comparison table.

Reads every results/**/baselines_*.csv, skips sample-limited check runs (tagged
_n<k>), and recomputes perplexity relative to the uncompressed `full` run of the
same task and context length. The per-file delta columns cannot be trusted here,
because the full reference usually lives in a different file.

Run:  python scripts/combine_results.py [results_dir]
Writes results_dir/combined.csv.
"""

import glob
import os
import re
import sys

import pandas as pd

CORE_METHODS = ["full", "snapkv", "pyramidkv", "h2o", "hotcold"]
# ablations sort after the method they vary; they are optional, so not "missing"
METHOD_ORDER = CORE_METHODS + ["hotcold_k0", "hotcold_hold"]
SHOW = ["task", "ctx", "budget", "method", "num_samples", "avg_ppl",
        "ppl_pct_vs_full", "avg_latency_s", "max_memory_gb", "hc_injection_used"]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    files = sorted(glob.glob(os.path.join(root, "**", "baselines_*.csv"), recursive=True))
    files = [f for f in files if not re.search(r"_n\d+\.csv$", f)]
    if not files:
        sys.exit(f"no full-run result files under {root}/")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop(columns=[c for c in ("ppl_delta_vs_full", "ppl_pct_vs_full") if c in df])

    ref = (df[df["method"] == "full"]
           .groupby(["task", "ctx"])["avg_ppl"].first()
           .rename("full_ppl").reset_index())
    df = df.merge(ref, on=["task", "ctx"], how="left")
    df["ppl_delta_vs_full"] = df["avg_ppl"] - df["full_ppl"]
    df["ppl_pct_vs_full"] = 100.0 * df["ppl_delta_vs_full"] / df["full_ppl"]

    df["_m"] = df["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    df["_b"] = pd.to_numeric(df["budget"], errors="coerce").fillna(-1)
    df = df.sort_values(["task", "ctx", "_b", "_m"]).drop(columns=["_m", "_b"])

    missing = sorted(set(CORE_METHODS) - set(df["method"]))
    print(f"{len(files)} result files" + (f" | missing methods: {missing}" if missing else ""))
    if df["full_ppl"].isna().any():
        print("warning: no `full` reference for some rows; their % column is blank")
    print()
    print(df[[c for c in SHOW if c in df]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    out = os.path.join(root, "combined.csv")
    df.drop(columns=["full_ppl"]).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
