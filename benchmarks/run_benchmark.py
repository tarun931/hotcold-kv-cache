"""LongBench harness for the hot/cold + Faiss cache (token-by-token decode).

Measures perplexity, wall-clock latency and peak GPU memory on LongBench tasks.
Expects the LongBench JSONL files at LONGBENCH_LOCAL_PATH (Kaggle dataset layout).

Run:  python benchmarks/run_benchmark.py
"""
import os, json, math, time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

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

MAX_INPUT_LENGTH   = int(os.environ.get("MAX_INPUT_LENGTH", 8192))
MAX_NEW_TOKENS_LAT = int(os.environ.get("MAX_NEW_TOKENS_LAT", 64))
PPL_MAX_TOKENS     = MAX_INPUT_LENGTH
EVAL_TOKENS        = int(os.environ.get("EVAL_TOKENS", 256))

# KV budget = peak hot-cache tokens. The design doc targets 15-20% of the full
# cache, which at 8k is ~1229-1638 tokens; 1024 and 2048 are the budgets SnapKV
# and H2O report, so those are the points worth sweeping.
KV_BUDGET    = int(os.environ.get("KV_BUDGET", 1024))

HOT_CAPACITY = KV_BUDGET
RECENT_SIZE  = int(os.environ.get("RECENT_SIZE", 128))
# Compression retains topk + recent, i.e. ~75% of the budget, leaving headroom
# so compression does not fire on every decode step.
TOPK_RATIO   = max(0.05, 0.75 - RECENT_SIZE / HOT_CAPACITY)

# Must exceed the total number of evicted tokens. If the ring buffer wraps while
# the FAISS index keeps growing, index positions no longer match ring slots and
# retrieval returns the wrong KV pairs.
CPU_BUFFER_SIZE = int(os.environ.get("CPU_BUFFER_SIZE", 32768))
FAISS_DIM       = 128     # = head_dim of Llama-3.1-8B; a smaller value crops the key vector
WINDOW_SIZE     = 16
QUERY_POOL_SIZE = 4
# Retrieval budget: the design doc targets ~1% of tokens; 64 is 0.8% at 8k.
LOOKAHEAD_K     = int(os.environ.get("LOOKAHEAD_K", 64))

# Per-run output, so parallel jobs on different GPUs never overwrite each other.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RUN_TAG = f"{DATASET}_b{KV_BUDGET}_L{MAX_INPUT_LENGTH}"


# -------------------------
# DATA LOADING
# -------------------------

def load_longbench_tasks() -> Dict[str, List[dict]]:
    datasets = {}
    for task, filename in TASK_FILE_MAP.items():
        path = os.path.join(LONGBENCH_LOCAL_PATH, filename)
        print(f"Loading task {task} from {path}")
        if not os.path.exists(path):
            print(f"  File not found → skipping")
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


def prepare_sample(sample, tokenizer, max_length=MAX_INPUT_LENGTH):
    context  = sample.get("context", "")
    question = sample.get("input", sample.get("question", ""))
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(MAIN_DEVICE)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(MAIN_DEVICE)

    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


# -------------------------
# METRICS
# -------------------------

@dataclass
class GenerationMetrics:
    total_latency_s: float
    avg_latency_per_token_s: float
    max_memory_gb: float


def _dev(model):
    return next(model.parameters()).device


def _make_cache_position(attn_2d: torch.Tensor) -> torch.LongTensor:
    # cache_position is the absolute position of *this* token (0-indexed) in the original sequence
    pos = int(attn_2d.shape[1] - 1)
    return torch.tensor([pos], device=attn_2d.device, dtype=torch.long)


# ========================================================================================
# GENERATION: Token-by-token greedy decode
# ========================================================================================

@torch.inference_mode()
def generate_with_hotcold(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    kvctrl,
):
    device = _dev(model)
    model.eval()

    set_active_kvctrl(kvctrl)
    past_key_values = build_hotcold_cache(model, kvctrl)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)

    input_ids = input_ids.to(device)
    attn = attention_mask.to(device)

    times = []
    generated = []

    # --- Prefill with prompt ---
    out = model(
        input_ids=input_ids,
        attention_mask=attn,
        use_cache=True,
        past_key_values=past_key_values,
    )
    past = out.past_key_values
    cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(cur)

    # --- Decode loop ---
    for _step in range(1, max_new_tokens):
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
        cache_position = _make_cache_position(attn)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()

        out = model(
            input_ids=cur,
            attention_mask=attn,          
            cache_position=cache_position,
            past_key_values=past,
            use_cache=True,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.time() - t0)

        past = out.past_key_values
        cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(cur)

    gen = torch.cat(generated, dim=1)
    full = torch.cat([input_ids, gen], dim=1)
    text = tokenizer.decode(full[0], skip_special_tokens=True)

    max_mem = (torch.cuda.max_memory_allocated(device=device) / (1024**3)) if device.type == "cuda" else 0.0
    total_lat = float(sum(times))
    avg_lat = total_lat / len(times) if times else 0.0

    # print(
    # "verify:",
    # "retrieval_launched=", kvctrl.dbg["retrieval_launched"],
    # "prefetch_launched=",  kvctrl.dbg["prefetch_launched"],
    # "injection_used=",     kvctrl.dbg["injection_used"],
    # flush=True,
    # )
    return text, GenerationMetrics(total_lat, avg_lat, max_mem)


    
# ========================================================================================
# PERPLEXITY: compressed-context continuation
# ========================================================================================
# The long prompt is prefilled in one pass (which is where compression happens),
# then the last EVAL_TOKENS are scored token-by-token against the resulting
# cache. This isolates the damage compression does.
#
# Scoring every token from a cold start instead would average in thousands of
# early positions where the cache is still tiny and no compression has occurred,
# measuring position-in-sequence rather than the method. It must stay identical
# to compute_ppl_press in run_baselines.py or the two are not comparable.

@torch.inference_mode()
def compute_ppl_like_nll(
    model,
    tokenizer,
    text: str,
    kvctrl,
    eval_tokens=EVAL_TOKENS,
    max_tokens=PPL_MAX_TOKENS,
):
    device = _dev(model)
    model.eval()

    set_active_kvctrl(kvctrl)
    past_key_values = build_hotcold_cache(model, kvctrl)

    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_tokens)["input_ids"].to(device)
    T = int(ids.shape[1])
    if T < 4:
        return float("nan")

    eval_n = min(eval_tokens, T - 2)
    split = T - eval_n                       # prefill on ids[:split], score ids[split:]

    nll_sum = 0.0
    count = 0

    # --- Prefill the context; compression fires here ---
    out = model(input_ids=ids[:, :split], use_cache=True, past_key_values=past_key_values)
    past = out.past_key_values

    # First eval token scored from the full-context prefill logits.
    logits = out.logits[:, -1, :].float()
    lp = F.log_softmax(logits, dim=-1)
    nll_sum += -lp.gather(-1, ids[:, split].unsqueeze(-1)).squeeze(-1).item()
    count += 1

    # --- Remaining eval tokens scored under the compressed cache ---
    for t in range(split, T - 1):
        cache_position = torch.tensor([t], device=device, dtype=torch.long)
        out = model(
            input_ids=ids[:, t : t + 1],
            cache_position=cache_position,
            past_key_values=past,
            use_cache=True,
        )
        logits = out.logits[:, -1, :].float()
        lp = F.log_softmax(logits, dim=-1)
        nll_sum += -lp.gather(-1, ids[:, t + 1].unsqueeze(-1)).squeeze(-1).item()
        count += 1
        past = out.past_key_values

    mean_nll = nll_sum / max(count, 1)
    ppl = math.exp(mean_nll)
    return ppl


# -------------------------
# MAIN
# -------------------------

print(f"[cfg] method=hotcold dataset={DATASET} budget={KV_BUDGET} ctx={MAX_INPUT_LENGTH} "
      f"topk={int(HOT_CAPACITY * TOPK_RATIO)} recent={RECENT_SIZE} "
      f"retained={int(HOT_CAPACITY * TOPK_RATIO) + RECENT_SIZE} "
      f"lookahead={LOOKAHEAD_K} cpu_buf={CPU_BUFFER_SIZE}")
print("CUDA available:", CUDA_OK)
if CUDA_OK:
    print("CUDA device count:", torch.cuda.device_count())
    torch.cuda.set_device(0)

print("\nLoading tokenizer/model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model_load_kwargs = dict(
    torch_dtype=torch.float16 if CUDA_OK else torch.float32,
    device_map={"": 0} if CUDA_OK else None,
    low_cpu_mem_usage=True,
)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_load_kwargs)
model.eval()

# REQUIRED for the updated hot/cold code (re-rotating retrieved K_pre)
set_active_rotary_emb(model.model.rotary_emb)

print("\nInstalling monkeypatch once...")
try:
    uninstall_monkeypatch()
except Exception:
    pass

install_monkeypatch(enable_debug=False)

print("\nLoading datasets...")
datasets = load_longbench_tasks()

results = []

for task, data in datasets.items():
    print(f"\n{'='*70}")
    print(f"Task: {task} | {len(data)} samples")
    print(f"{'='*70}")

    ppl_list = []
    lat_list = []
    mem_list = []

    for i, sample in enumerate(data):
        prepared = prepare_sample(sample, tokenizer)

        kvctrl_ppl = KVController(
            max_capacity_prompt=HOT_CAPACITY,
            topk_ratio=TOPK_RATIO,
            recent_size=RECENT_SIZE,
            cpu_buffer_size=CPU_BUFFER_SIZE,
            faiss_dim=FAISS_DIM,
            window_size=WINDOW_SIZE,
            query_pool_size=QUERY_POOL_SIZE,
            lookahead_k=LOOKAHEAD_K,
            enable_asserts=False,
            profile_overlap=False,
            force_wait_for_prefetch_diagnostic=False,
        )
        ppl = compute_ppl_like_nll(
            model,
            tokenizer,
            prepared["prompt"],
            kvctrl_ppl,
            max_tokens=PPL_MAX_TOKENS,
        )
        ppl_list.append(ppl)

        kvctrl_lat = KVController(
            max_capacity_prompt=HOT_CAPACITY,
            topk_ratio=TOPK_RATIO,
            recent_size=RECENT_SIZE,
            cpu_buffer_size=CPU_BUFFER_SIZE,
            faiss_dim=FAISS_DIM,
            window_size=WINDOW_SIZE,
            query_pool_size=QUERY_POOL_SIZE,
            lookahead_k=LOOKAHEAD_K,
            enable_asserts=False,
            profile_overlap=False,
            force_wait_for_prefetch_diagnostic=False,
        )
        _txt, metrics = generate_with_hotcold(
            model,
            tokenizer,
            prepared["input_ids"],
            prepared["attention_mask"],
            MAX_NEW_TOKENS_LAT,
            kvctrl_lat,
        )
        lat_list.append(metrics.total_latency_s)
        mem_list.append(metrics.max_memory_gb)

        print("dbg:", kvctrl_lat.dbg["compress_calls"], kvctrl_lat.dbg["evicted_tokens"], kvctrl_lat.dbg["faiss_add"])


        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(data)}")

    avg_ppl = float(np.nanmean(np.array(ppl_list, dtype=np.float64)))
    avg_lat = float(np.mean(lat_list)) if lat_list else float("nan")
    max_mem = float(np.max(mem_list)) if mem_list else 0.0

    results.append({
        "task": task,
        "num_samples": len(data),
        "avg_ppl": avg_ppl,
        "std_ppl": float(np.nanstd(np.array(ppl_list, dtype=np.float64))),
        "avg_latency_s": avg_lat,
        "std_latency_s": float(np.std(lat_list)) if lat_list else 0.0,
        "max_memory_gb": max_mem,
    })

df = pd.DataFrame(results)
print(f"\n{'='*70}")
print("FINAL RESULTS (HOT/COLD CACHE)")
print(f"{'='*70}")
print(df.to_string(index=False))

csv_path = os.path.join(RESULTS_DIR, f"hotcold_{RUN_TAG}.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved → {csv_path}")
