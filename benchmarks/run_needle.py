"""Needle-in-a-Haystack over the hot/cold cache.

Hides one fact at a known depth in a long filler document, asks for it back, and
records whether the model reproduces it. Sweeping context length x needle depth
gives the grid plotted by `scripts/plot_needle.py`.

The point here is not "does compression break retrieval" — that is what the
SnapKV figure shows, and it is green almost everywhere. The point is what the
CPU tier buys, so the default is to run two hot/cold variants over the same grid:

    k0      LOOKAHEAD_K=0, evict to the CPU tier but never fetch anything back
    hold    HOLD_RETRIEVED=1, fetch and keep the retrieved tokens in attention

A cell that fails under k0 and passes under hold is a needle the cold tier won
back. If both variants pass everywhere the test does not bite: the needle is
never leaving the hot tier, so lower KV_BUDGET until it does.

The needle is a random 7-digit number, so the answer cannot be guessed from
world knowledge and scoring is an exact substring match rather than a judge
model. That also means this runs on a base model: no instruction tuning is
needed to continue "The access code is ".

Run:  python benchmarks/run_needle.py
Expects LongBench JSONL files at LONGBENCH_PATH; their contexts are concatenated
into the haystack, so no network access is needed.
"""

import gc
import json
import os
import random
from typing import List

import pandas as pd
import torch

from hotcold_kv import (
    KVController,
    build_hotcold_cache,
    install_monkeypatch,
    set_active_kvctrl,
    set_active_rotary_emb,
    uninstall_monkeypatch,
)

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B")
LONGBENCH_PATH = os.environ.get("LONGBENCH_PATH", "/kaggle/input/longbench-json")
HAYSTACK_TASKS = os.environ.get("HAYSTACK_TASKS", "gov_report,multi_news").split(",")

# Context lengths to sweep, in tokens. Job 1 is the short end; the long end
# needs the prefill-logits change and a larger CPU_BUFFER_SIZE.
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "1024,2048,4096,8192,16384").split(",")]
# Needle position as a fraction of the haystack, matching the SnapKV figure's
# 0-100% depth axis.
DEPTHS = [float(x) for x in os.environ.get("DEPTHS", "0,11,22,33,44,56,67,78,89,100").split(",")]
# Repeats per cell, each with a different needle value and filler offset. The
# score in a cell is the fraction passed.
REPEATS = int(os.environ.get("REPEATS", 3))

VARIANTS = os.environ.get("NEEDLE_VARIANTS", "k0,hold").split(",")
KV_BUDGET = int(os.environ.get("KV_BUDGET", 1024))
RECENT_SIZE = int(os.environ.get("RECENT_SIZE", 128))
TOPK_RATIO = max(0.05, 0.75 - RECENT_SIZE / KV_BUDGET)
CPU_BUFFER_SIZE = int(os.environ.get("CPU_BUFFER_SIZE", 32768))
LOOKAHEAD_K = int(os.environ.get("LOOKAHEAD_K", 64))
RETRIEVAL_WINDOW = 16
QUERY_POOL_SIZE = 4
FAISS_DIM = 128

MAX_NEW_TOKENS = int(os.environ.get("NEEDLE_MAX_NEW_TOKENS", 16))
SEED = int(os.environ.get("SEED", 0))
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NEEDLE_TEMPLATE = "The access code for the {city} vault is {code}."
QUESTION = "\n\nQuestion: What is the access code for the {city} vault?\nAnswer: The access code is"
CITY = "Brindlewood"

CUDA_OK = torch.cuda.is_available()


def load_haystack_text() -> str:
    """One long filler string, built from LongBench contexts so nothing is downloaded."""
    parts: List[str] = []
    for task in HAYSTACK_TASKS:
        path = os.path.join(LONGBENCH_PATH, f"{task}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                ctx = json.loads(line).get("context", "")
                if ctx:
                    parts.append(ctx)
    if not parts:
        raise SystemExit(f"no haystack text found under {LONGBENCH_PATH} for {HAYSTACK_TASKS}")
    return "\n\n".join(parts)


def build_prompt(tokenizer, filler_ids, length, depth_pct, code, rng):
    """Filler cut to `length` tokens, with the needle spliced in at `depth_pct`.

    The needle is inserted by token index, then the whole thing is re-encoded, so
    the final prompt is the text a reader would see rather than a token splice
    that may not decode cleanly.
    """
    needle = NEEDLE_TEMPLATE.format(city=CITY, code=code)
    question = QUESTION.format(city=CITY)
    n_needle = len(tokenizer(needle, add_special_tokens=False)["input_ids"])
    n_question = len(tokenizer(question, add_special_tokens=False)["input_ids"])

    budget = length - n_needle - n_question
    if budget < 64:
        raise ValueError(f"length {length} too short for needle + question")

    # A random window of the filler, so repeats of the same cell differ.
    start = rng.randrange(0, max(1, len(filler_ids) - budget))
    window = filler_ids[start:start + budget]
    if len(window) < budget:                      # wrap around the end
        window = window + filler_ids[:budget - len(window)]

    cut = int(len(window) * depth_pct / 100.0)
    before = tokenizer.decode(window[:cut])
    after = tokenizer.decode(window[cut:])
    text = f"{before}\n{needle}\n{after}{question}"

    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    return ids, needle


def make_controller():
    return KVController(
        max_capacity_prompt=KV_BUDGET,
        topk_ratio=TOPK_RATIO,
        recent_size=RECENT_SIZE,
        cpu_buffer_size=CPU_BUFFER_SIZE,
        faiss_dim=FAISS_DIM,
        window_size=RETRIEVAL_WINDOW,
        query_pool_size=QUERY_POOL_SIZE,
        lookahead_k=LOOKAHEAD_K,
        hold_retrieved=True,
        enable_asserts=False,
        profile_overlap=False,
        force_wait_for_prefetch_diagnostic=False,
    )


@torch.inference_mode()
def generate(model, input_ids, cache, kvctrl, max_new_tokens):
    """Greedy decode with explicit cache_position: positions stay absolute while
    the physical cache holds only `budget` tokens."""
    set_active_kvctrl(kvctrl)
    dev = next(model.parameters()).device
    input_ids = input_ids.to(dev)
    out = model(input_ids=input_ids, use_cache=True, past_key_values=cache)
    pos = int(input_ids.shape[1])
    produced = []
    for _ in range(max_new_tokens):
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        produced.append(int(nxt.item()))
        out = model(
            input_ids=nxt,
            cache_position=torch.tensor([pos], device=dev, dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
        )
        pos += 1
    return produced


def main():
    if CUDA_OK:
        torch.cuda.set_device(0)
    print(f"[cfg] variants={','.join(VARIANTS)} budget={KV_BUDGET} lengths={LENGTHS} "
          f"depths={DEPTHS} repeats={REPEATS} cpu_buffer={CPU_BUFFER_SIZE} "
          f"lookahead={LOOKAHEAD_K} model={MODEL_NAME}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
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
    set_active_rotary_emb(model.model.rotary_emb)
    install_monkeypatch(enable_debug=False)

    print("Building haystack...")
    filler_ids = tokenizer(load_haystack_text(), add_special_tokens=False)["input_ids"]
    print(f"  filler pool: {len(filler_ids):,} tokens")

    rows = []
    try:
        for variant in VARIANTS:
            lookahead = 0 if variant == "k0" else LOOKAHEAD_K
            for length in LENGTHS:
                for depth in DEPTHS:
                    hits = 0
                    for r in range(REPEATS):
                        # Same seed per cell across variants, so both variants
                        # see identical prompts and the panels are comparable.
                        rng = random.Random((SEED, length, depth, r).__hash__())
                        code = f"{rng.randrange(1_000_000, 9_999_999)}"
                        ids, needle = build_prompt(tokenizer, filler_ids, length, depth, code, rng)

                        kvctrl = make_controller()
                        kvctrl.lookahead_k = lookahead
                        cache = build_hotcold_cache(model, kvctrl)
                        toks = generate(model, ids, cache, kvctrl, MAX_NEW_TOKENS)
                        text = tokenizer.decode(toks)
                        ok = code in text.replace(",", "").replace(" ", "")
                        hits += int(ok)

                        rows.append({
                            "variant": variant, "length": length, "depth": depth,
                            "repeat": r, "prompt_tokens": int(ids.shape[1]),
                            "code": code, "generated": text.strip()[:120], "hit": int(ok),
                            "evicted_tokens": int(kvctrl.dbg["evicted_tokens"]),
                            "injection_used": int(kvctrl.dbg["injection_used"]),
                        })
                        del cache, kvctrl
                        gc.collect()
                        if CUDA_OK:
                            torch.cuda.empty_cache()
                    print(f"  {variant} len={length:>7} depth={depth:>5}  {hits}/{REPEATS}")
    finally:
        uninstall_monkeypatch()
        set_active_kvctrl(None)

    df = pd.DataFrame(rows)
    tag = f"b{KV_BUDGET}_L{max(LENGTHS)}"
    path = os.path.join(RESULTS_DIR, f"needle_{tag}.csv")
    df.to_csv(path, index=False)

    print(f"\nScore by variant and length (fraction of cells passed):")
    print(df.groupby(["variant", "length"])["hit"].mean().unstack().round(3).to_string())
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
