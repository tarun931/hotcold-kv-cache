"""LongBench harness for the compression baselines (SnapKV / PyramidKV / H2O).

Mirrors run_benchmark.py but swaps the hot/cold cache for the press baselines.
For each method it reports, per task:
  - Perplexity of a compressed-context continuation (last EVAL_TOKENS scored
    token-by-token against the compressed prompt cache),
  - Wall-clock latency of generating MAX_NEW_TOKENS_LAT tokens,
  - Peak GPU memory allocated.

The press caches compress the whole prompt in a single prefill, so decoding is
driven manually with explicit cache_position (positions stay absolute while the
physical cache holds only `budget` tokens).

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

from hotcold_kv.baselines import (
    PressConfig,
    build_press_cache,
    set_active_press_cache,
    install_press_patch,
    uninstall_press_patch,
)

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "meta-llama/Llama-3.2-1B"

CUDA_OK = torch.cuda.is_available()
MAIN_DEVICE = torch.device("cuda:0" if CUDA_OK else "cpu")

LONGBENCH_LOCAL_PATH = "/kaggle/input/longbench-json"
TASK_FILE_MAP = {
    "multifieldqa_en": "multifieldqa_en.jsonl",
}

MAX_INPUT_LENGTH = 4096
MAX_NEW_TOKENS_LAT = 64
PPL_MAX_TOKENS = 4096
EVAL_TOKENS = 256                 # continuation length scored for PPL

# Shared budget so methods are compared at the same memory footprint.
MAX_CAPACITY = 512
WINDOW_SIZE = 32
KERNEL_SIZE = 7
PYRAMID_BETA = 20

METHODS = ["snapkv", "pyramidkv", "h2o"]


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


# -------------------------
# GENERATION (latency + memory)
# -------------------------
@torch.inference_mode()
def generate_with_press(model, input_ids, max_new_tokens, cache):
    dev = _dev(model)
    model.eval()
    set_active_press_cache(cache)

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
def compute_ppl_press(model, tokenizer, text, cache, eval_tokens=EVAL_TOKENS, max_tokens=PPL_MAX_TOKENS):
    dev = _dev(model)
    model.eval()
    set_active_press_cache(cache)

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

    print("\nInstalling baseline attention patch...")
    try:
        uninstall_press_patch()
    except Exception:
        pass
    install_press_patch()

    print("\nLoading datasets...")
    datasets = load_longbench_tasks()

    results = []
    for method in METHODS:
        for task, data in datasets.items():
            print(f"\n{'=' * 70}\nMethod: {method} | Task: {task} | {len(data)} samples\n{'=' * 70}")
            ppl_list, lat_list, mem_list = [], [], []

            for i, sample in enumerate(data):
                prompt, input_ids = prepare_prompt(sample, tokenizer)

                cache_ppl = build_press_cache(model, make_config(method))
                ppl = compute_ppl_press(model, tokenizer, prompt, cache_ppl)
                ppl_list.append(ppl)

                cache_lat = build_press_cache(model, make_config(method))
                lat, mem = generate_with_press(model, input_ids, MAX_NEW_TOKENS_LAT, cache_lat)
                lat_list.append(lat)
                mem_list.append(mem)

                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{len(data)}")

            results.append({
                "method": method,
                "task": task,
                "num_samples": len(data),
                "avg_ppl": float(np.nanmean(np.array(ppl_list, dtype=np.float64))),
                "std_ppl": float(np.nanstd(np.array(ppl_list, dtype=np.float64))),
                "avg_latency_s": float(np.mean(lat_list)) if lat_list else float("nan"),
                "max_memory_gb": float(np.max(mem_list)) if mem_list else 0.0,
            })

    uninstall_press_patch()

    df = pd.DataFrame(results)
    print(f"\n{'=' * 70}\nFINAL RESULTS (COMPRESSION BASELINES)\n{'=' * 70}")
    print(df.to_string(index=False))
    csv_path = "baseline_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
