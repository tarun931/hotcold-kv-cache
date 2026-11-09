"""Unified LongBench comparison harness.

Runs every method through one identical protocol so the numbers are comparable:

    full       uncompressed reference (plain DynamicCache, no patch)
    snapkv     observation-window importance + pooling
    pyramidkv  SnapKV importance with a per-layer pyramidal budget
    h2o        Heavy-Hitter Oracle, online eviction
    hotcold    this project (GPU hot tier + FAISS CPU cold tier + async retrieval)

For each method and task it reports:
  - Perplexity of a compressed-context continuation (the long prompt is
    prefilled, then the last EVAL_TOKENS are scored token-by-token against the
    resulting cache),
  - Wall-clock latency of generating MAX_NEW_TOKENS_LAT tokens,
  - Peak GPU memory allocated.

Scoring a fixed-length continuation after a full prefill is what isolates the
damage done by compression. Scoring every token from a cold start instead would
average in thousands of early positions where no compression has happened yet,
which measures position-in-sequence far more than it measures the method.

Only one attention patch may be installed at a time, so the patch is swapped at
the start of each method. Decoding is driven manually with explicit
cache_position: positions stay absolute while the physical cache holds only
`budget` tokens.

Run:  python benchmarks/run_baselines.py
Expects LongBench JSONL files at LONGBENCH_LOCAL_PATH (Kaggle dataset layout).
"""

import os, json, time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from hotcold_kv.baselines import (
    PressConfig,
    build_press_cache,
    set_active_press_cache,
    install_press_patch,
    uninstall_press_patch,
)
from hotcold_kv import (
    KVController,
    set_active_kvctrl,
    set_active_rotary_emb,
    install_monkeypatch,
    uninstall_monkeypatch,
    build_hotcold_cache,
)

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-1B")

CUDA_OK = torch.cuda.is_available()
MAIN_DEVICE = torch.device("cuda:0" if CUDA_OK else "cpu")

LONGBENCH_LOCAL_PATH = os.environ.get("LONGBENCH_PATH", "/kaggle/input/longbench-json")

# One LongBench task per run, so jobs can be sharded one per GPU.
DATASET = os.environ.get("DATASET", "multifieldqa_en")
TASK_FILE_MAP = {DATASET: f"{DATASET}.jsonl"}

MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_LENGTH", 8192))
MAX_NEW_TOKENS_LAT = int(os.environ.get("MAX_NEW_TOKENS_LAT", 64))
PPL_MAX_TOKENS = MAX_INPUT_LENGTH
EVAL_TOKENS = int(os.environ.get("EVAL_TOKENS", 256))   # continuation scored for PPL

# Shared budget: every compressing method is capped at the same peak KV
# footprint, which is what makes the comparison meaningful. "full" ignores it.
MAX_CAPACITY = int(os.environ.get("KV_BUDGET", 1024))
WINDOW_SIZE = 32          # SnapKV/PyramidKV/H2O observation window; fixed across budgets
KERNEL_SIZE = 7
PYRAMID_BETA = 20

# Hot/cold settings, mirroring run_benchmark.py so the method behaves identically
# in both harnesses.
RECENT_SIZE = int(os.environ.get("RECENT_SIZE", 128))
TOPK_RATIO = max(0.05, 0.75 - RECENT_SIZE / MAX_CAPACITY)
CPU_BUFFER_SIZE = int(os.environ.get("CPU_BUFFER_SIZE", 32768))
FAISS_DIM = 128           # = head_dim of Llama-3.1-8B; a smaller value crops the key vector
RETRIEVAL_WINDOW = 16     # launch an async retrieval every N decode steps
QUERY_POOL_SIZE = 4
DEFAULT_LOOKAHEAD_K = 64
LOOKAHEAD_K = int(os.environ.get("LOOKAHEAD_K", DEFAULT_LOOKAHEAD_K))
# 1 = keep retrieved tokens in attention until the next retrieval replaces
# them; 0 (default) = attend to them only at the step their copy lands.
HOLD_RETRIEVED = os.environ.get("HOLD_RETRIEVED", "0") == "1"
# A non-default lookahead (LOOKAHEAD_K=0 switches retrieval off) or held
# retrieval is a separate hot/cold variant: give it its own method label and
# file suffix so its rows never mix with the main hot/cold rows.
HOTCOLD_SUFFIX = ((f"_k{LOOKAHEAD_K}" if LOOKAHEAD_K != DEFAULT_LOOKAHEAD_K else "")
                  + ("_hold" if HOLD_RETRIEVED else ""))
HOTCOLD_LABEL = "hotcold" + HOTCOLD_SUFFIX

# "full" first: the uncompressed reference everything else is measured against.
METHODS = os.environ.get("METHODS", "full,snapkv,pyramidkv,h2o,hotcold").split(",")

# 0 = every sample. Set small for a quick end-to-end check before a full run.
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", 0))

# KVController.dbg counters reported for hotcold. injection_used > 0 is what
# shows retrieved cold tokens actually reached attention.
HC_COUNTERS = ("evicted_tokens", "retrieval_launched", "prefetch_launched",
               "prefetch_ready", "injection_used")

# Per-run output, so parallel jobs on different GPUs never overwrite each other.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RUN_TAG = f"{DATASET}_b{MAX_CAPACITY}_L{MAX_INPUT_LENGTH}"
# nor may a hot/cold variant
RUN_TAG += HOTCOLD_SUFFIX
if MAX_SAMPLES > 0:
    # A sample-limited check run must never share a filename with the full run.
    RUN_TAG += f"_n{MAX_SAMPLES}"


# -------------------------
# DATA LOADING
# -------------------------
def load_longbench_tasks() -> Dict[str, List[dict]]:
    datasets = {}
    for task, filename in TASK_FILE_MAP.items():
        path = os.path.join(LONGBENCH_LOCAL_PATH, filename)
        print(f"Loading task {task} from {path}")
        if not os.path.exists(path):
            print("  File not found -> skipping")
            continue
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        datasets[task] = records
        print(f"  Loaded {len(records)} samples.")
    return datasets


def prepare_prompt(sample, tokenizer, max_length=MAX_INPUT_LENGTH):
    context = sample.get("context", "")
    question = sample.get("input", sample.get("question", ""))
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(MAIN_DEVICE)
    return prompt, input_ids


def _dev(model):
    return next(model.parameters()).device


def make_config(method: str) -> PressConfig:
    return PressConfig(
        method=method,
        max_capacity=MAX_CAPACITY,
        window_size=WINDOW_SIZE,
        kernel_size=KERNEL_SIZE,
        pooling="maxpool",
        pyramid_beta=PYRAMID_BETA,
    )


def install_patch_for(method: str, model):
    """Swap the attention patch. Only one may be installed at a time, and each
    install must see the true original forward, so always uninstall both first."""
    for uninstall in (uninstall_press_patch, uninstall_monkeypatch):
        try:
            uninstall()
        except Exception:
            pass
    set_active_press_cache(None)
    set_active_kvctrl(None)

    if method == "full":
        return                       # stock attention, nothing patched
    if method == "hotcold":
        set_active_rotary_emb(model.model.rotary_emb)
        install_monkeypatch(enable_debug=False)
    else:
        install_press_patch()


def build_cache_for(method: str, model):
    """Fresh cache per sample. Returns (cache, kvctrl_or_None)."""
    if method == "full":
        return DynamicCache(), None
    if method == "hotcold":
        kvctrl = KVController(
            max_capacity_prompt=MAX_CAPACITY,
            topk_ratio=TOPK_RATIO,
            recent_size=RECENT_SIZE,
            cpu_buffer_size=CPU_BUFFER_SIZE,
            faiss_dim=FAISS_DIM,
            window_size=RETRIEVAL_WINDOW,
            query_pool_size=QUERY_POOL_SIZE,
            lookahead_k=LOOKAHEAD_K,
            hold_retrieved=HOLD_RETRIEVED,
            enable_asserts=False,
            profile_overlap=False,
            force_wait_for_prefetch_diagnostic=False,
        )
        return build_hotcold_cache(model, kvctrl), kvctrl
    return build_press_cache(model, make_config(method)), None


def activate(method: str, cache, kvctrl):
    """Point the installed patch at this sample's cache."""
    if method == "full":
        return
    if method == "hotcold":
        set_active_kvctrl(kvctrl)
    else:
        set_active_press_cache(cache)


# -------------------------
# GENERATION (latency + memory)
# -------------------------
@torch.inference_mode()
def generate_with_press(model, input_ids, max_new_tokens, cache, method, kvctrl=None):
    dev = _dev(model)
    model.eval()
    activate(method, cache, kvctrl)

    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=dev)

    input_ids = input_ids.to(dev)
    prompt_len = int(input_ids.shape[1])

    # Prefill (compresses the prompt inside the patched attention).
    out = model(input_ids=input_ids, use_cache=True, past_key_values=cache)
    cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    times = []
    for step in range(1, max_new_tokens):
        cache_position = torch.tensor([prompt_len + step - 1], device=dev, dtype=torch.long)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.time()
        out = model(
            input_ids=cur,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        times.append(time.time() - t0)
        cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    max_mem = (torch.cuda.max_memory_allocated(device=dev) / (1024 ** 3)) if dev.type == "cuda" else 0.0
    total_lat = float(sum(times))
    return total_lat, max_mem


# -------------------------
# PERPLEXITY (compressed-context continuation)
# -------------------------
@torch.inference_mode()
def compute_ppl_press(model, tokenizer, text, cache, method, kvctrl=None,
                      eval_tokens=EVAL_TOKENS, max_tokens=PPL_MAX_TOKENS):
    dev = _dev(model)
    model.eval()
    activate(method, cache, kvctrl)

    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)["input_ids"].to(dev)
    T = int(ids.shape[1])
    if T < 4:
        return float("nan")

    eval_n = min(eval_tokens, T - 2)
    split = T - eval_n                       # prefill on ids[:split], score ids[split:]

    out = model(input_ids=ids[:, :split], use_cache=True, past_key_values=cache)
    nll_sum = 0.0
    count = 0

    # First eval token scored from the (full-context) prefill logits.
    logits = out.logits[:, -1, :].float()
    lp = F.log_softmax(logits, dim=-1)
    nll_sum += -lp.gather(-1, ids[:, split].unsqueeze(-1)).squeeze(-1).item()
    count += 1

    # Remaining eval tokens scored under the compressed cache.
    for t in range(split, T - 1):
        cur = ids[:, t:t + 1]
        cache_position = torch.tensor([t], device=dev, dtype=torch.long)
        out = model(
            input_ids=cur,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )
        logits = out.logits[:, -1, :].float()
        lp = F.log_softmax(logits, dim=-1)
        nll_sum += -lp.gather(-1, ids[:, t + 1].unsqueeze(-1)).squeeze(-1).item()
        count += 1

    mean_nll = nll_sum / max(count, 1)
    return float(np.exp(mean_nll))


# -------------------------
# MAIN
# -------------------------
def main():
    print(f"[cfg] methods={','.join(METHODS)} dataset={DATASET} budget={MAX_CAPACITY} "
          f"ctx={MAX_INPUT_LENGTH} eval_tokens={EVAL_TOKENS} press_window={WINDOW_SIZE} "
          f"hotcold(topk={int(MAX_CAPACITY * TOPK_RATIO)} recent={RECENT_SIZE} "
          f"retained={int(MAX_CAPACITY * TOPK_RATIO) + RECENT_SIZE} lookahead={LOOKAHEAD_K} "
          f"hold={int(HOLD_RETRIEVED)})")
    print("CUDA available:", CUDA_OK)
    if CUDA_OK:
        print("CUDA device count:", torch.cuda.device_count())
        torch.cuda.set_device(0)

    print("\nLoading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if CUDA_OK else torch.float32,
        device_map={"": 0} if CUDA_OK else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print("\nLoading datasets...")
    datasets = load_longbench_tasks()

    results = []
    for method in METHODS:
        install_patch_for(method, model)
        for task, data in datasets.items():
            if MAX_SAMPLES > 0:
                data = data[:MAX_SAMPLES]
            print(f"\n{'=' * 70}\nMethod: {method} | Task: {task} | {len(data)} samples\n{'=' * 70}")
            ppl_list, lat_list, mem_list = [], [], []
            hc = {k: 0 for k in HC_COUNTERS}

            for i, sample in enumerate(data):
                prompt, input_ids = prepare_prompt(sample, tokenizer)

                cache_ppl, kv_ppl = build_cache_for(method, model)
                ppl = compute_ppl_press(model, tokenizer, prompt, cache_ppl, method, kv_ppl)
                ppl_list.append(ppl)
                if kv_ppl is not None:
                    for k in HC_COUNTERS:
                        hc[k] += int(kv_ppl.dbg.get(k, 0))

                cache_lat, kv_lat = build_cache_for(method, model)
                lat, mem = generate_with_press(model, input_ids, MAX_NEW_TOKENS_LAT,
                                               cache_lat, method, kv_lat)
                lat_list.append(lat)
                mem_list.append(mem)

                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{len(data)}")

            results.append({
                "method": HOTCOLD_LABEL if method == "hotcold" else method,
                "task": task,
                "budget": "full" if method == "full" else MAX_CAPACITY,
                "ctx": MAX_INPUT_LENGTH,
                "num_samples": len(data),
                "avg_ppl": float(np.nanmean(np.array(ppl_list, dtype=np.float64))),
                "std_ppl": float(np.nanstd(np.array(ppl_list, dtype=np.float64))),
                "avg_latency_s": float(np.mean(lat_list)) if lat_list else float("nan"),
                "max_memory_gb": float(np.max(mem_list)) if mem_list else 0.0,
                # hot/cold only: evidence that eviction and retrieval actually ran
                # during the scored PPL passes. Blank for the other methods.
                **({f"hc_{k}": v for k, v in hc.items()} if method == "hotcold" else {}),
            })
            if method == "hotcold":
                print("  hot/cold counters (PPL passes): "
                      + ", ".join(f"{k}={v}" for k, v in hc.items()))

    install_patch_for("full", model)   # leave attention unpatched on exit

    df = pd.DataFrame(results)
    # Report quality relative to the uncompressed reference, which is the number
    # that actually says whether compression hurt.
    base = df.loc[df["method"] == "full", "avg_ppl"]
    if len(base) and np.isfinite(base.iloc[0]):
        df["ppl_delta_vs_full"] = df["avg_ppl"] - base.iloc[0]
        df["ppl_pct_vs_full"] = 100.0 * (df["avg_ppl"] - base.iloc[0]) / base.iloc[0]

    print(f"\n{'=' * 70}\nFINAL RESULTS — all methods, identical protocol\n{'=' * 70}")
    print(df.to_string(index=False))
    csv_path = os.path.join(RESULTS_DIR, f"baselines_{RUN_TAG}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
