"""Correctness checks for the attention patches.

With compression and retrieval switched off, a patched model must behave like
the unpatched one. If it does not, every benchmark number produced afterwards
measures a bug rather than a method, so run this before any sweep.

Hot/cold patch:
  1. prefill causality   - position k's logits must not change when the tokens
                            after k are removed. Catches bidirectional prefill.
  2. prefill parity      - patched vs stock last-position logits.
  3. decode parity       - patched vs stock logits over several decode steps,
                            teacher-forced: both runs are fed the same tokens,
                            so one near-tied greedy choice cannot cascade.
  4. compression off     - with a budget larger than the sequence, nothing may
                            be evicted and nothing retrieved.
  5. prefill vs budget   - prefill output must be identical with compression on
                            and off. Compression changes what is *stored* for
                            later steps, never the prompt's own attention.

Baseline patch (SnapKV / PyramidKV / H2O):
  6. baseline parity     - each baseline with a budget larger than the sequence
                            must match stock, teacher-forced.

Tolerances: Llama logits run to roughly +/-30 and fp16 attention kernels tile
differently, so small differences are expected. A real bug is far larger: a
bidirectional prefill measured 5.6 and a broken decode 44.7.

Run:  PYTHONPATH=src python tests/test_parity.py
Needs a CUDA GPU (the hot/cold decode path is a Triton kernel).
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from hotcold_kv import (
    KVController,
    set_active_kvctrl,
    set_active_rotary_emb,
    install_monkeypatch,
    uninstall_monkeypatch,
    build_hotcold_cache,
)
from hotcold_kv.baselines import (
    PressConfig,
    build_press_cache,
    set_active_press_cache,
    install_press_patch,
    uninstall_press_patch,
)

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B")
N_DECODE = int(os.environ.get("N_DECODE", 16))

TEXT = (
    "The history of computing hardware spans several distinct eras. Early "
    "mechanical calculators gave way to electromechanical relays, then to "
    "vacuum tubes, transistors, and finally integrated circuits. Each "
    "transition brought order-of-magnitude improvements in speed, reliability "
    "and cost. The invention of the stored-program architecture meant that a "
    "single machine could be repurposed by changing its software rather than "
    "rewiring its hardware, which is the idea that made general-purpose "
    "computing practical. Memory hierarchies emerged because fast storage was "
    "expensive and cheap storage was slow, a tension that still shapes system "
    "design today. Caches sit between the processor and main memory precisely "
    "because locality of reference makes recently used data likely to be used "
    "again soon. "
) * 3

TOL_LOGIT = 0.5           # same-kernel comparisons (prefill)
TOL_DECODE_LOGIT = 1.0    # Triton kernel vs SDPA
MAX_TOP1_MISMATCH = 1     # decode steps allowed to disagree on the top token


def decode_trace(model, ids, cache, n_steps, device, forced=None):
    """Prefill `ids`, then decode n_steps with explicit absolute positions.

    With forced=None the run feeds its own greedy choice each step; with a list
    it feeds those tokens instead (teacher forcing).
    Returns (prefill_logits, [step_logits], [fed_tokens])."""
    out = model(input_ids=ids, use_cache=True, past_key_values=cache)
    prefill = out.logits[:, -1, :].float().cpu()
    past = out.past_key_values
    nxt = int(out.logits[0, -1].argmax())

    T = int(ids.shape[1])
    logits, fed = [], []
    for i in range(n_steps):
        tok = forced[i] if forced is not None else nxt
        fed.append(tok)
        cur = torch.tensor([[tok]], device=device, dtype=torch.long)
        cp = torch.tensor([T + i], device=device, dtype=torch.long)
        out = model(input_ids=cur, cache_position=cp, past_key_values=past, use_cache=True)
        lg = out.logits[:, -1, :].float()
        logits.append(lg.cpu())
        nxt = int(lg[0].argmax())
        past = out.past_key_values
    return prefill, logits, fed


def make_kvctrl(budget=10 ** 9):
    """Retrieval is always disabled here; the budget controls compression."""
    return KVController(
        max_capacity_prompt=budget,
        topk_ratio=0.15,
        recent_size=min(128, max(budget // 2, 1)),
        cpu_buffer_size=4096,
        faiss_dim=64,
        window_size=16,
        query_pool_size=4,
        lookahead_k=0,
        enable_asserts=True,
        profile_overlap=False,
        force_wait_for_prefetch_diagnostic=False,
    )


def compare_decode(ref_logits, got_logits):
    max_diff = max((a - b).abs().max().item() for a, b in zip(ref_logits, got_logits))
    agree = sum(int(a.argmax() == b.argmax()) for a, b in zip(ref_logits, got_logits))
    return max_diff, agree


def run_checks(model, ids, device, n_decode=N_DECODE, log=print):
    """Run every check against an already-loaded model. Returns failure strings."""
    failures = []

    def report(ok, label, detail, why=None):
        log(f"[{'PASS' if ok else 'FAIL'}] {label:<24} {detail}")
        if not ok and why:
            failures.append(why)

    n_layers = int(model.config.num_hidden_layers)

    # ---- stock reference, nothing patched ---------------------------------
    with torch.inference_mode():
        ref_prefill, ref_logits, ref_fed = decode_trace(
            model, ids, DynamicCache(), n_decode, device)

    # ---- hot/cold patch ---------------------------------------------------
    set_active_rotary_emb(model.model.rotary_emb)
    try:
        uninstall_monkeypatch()
    except Exception:
        pass
    install_monkeypatch(enable_debug=False)
    try:
        # 1. prefill causality
        k = int(ids.shape[1]) // 2
        with torch.inference_mode():
            set_active_kvctrl(make_kvctrl())
            full_k = model(input_ids=ids, use_cache=False).logits[0, k].float().cpu()
            set_active_kvctrl(make_kvctrl())
            trunc_k = model(input_ids=ids[:, : k + 1], use_cache=False).logits[0, -1].float().cpu()
        d = (full_k - trunc_k).abs().max().item()
        report(d < TOL_LOGIT, "1. prefill causality", f"max|d| = {d:.4f}  (tol {TOL_LOGIT})",
               "prefill is not causal: position k's logits change when later tokens are removed")

        # 2 & 3. parity against stock
        kv = make_kvctrl()
        set_active_kvctrl(kv)
        with torch.inference_mode():
            got_prefill, got_logits, _ = decode_trace(
                model, ids, build_hotcold_cache(model, kv), n_decode, device, forced=ref_fed)
        d = (ref_prefill - got_prefill).abs().max().item()
        report(d < TOL_LOGIT, "2. prefill parity", f"max|d| = {d:.4f}  (tol {TOL_LOGIT})",
               "patched prefill logits differ from stock beyond fp16 noise")

        d, agree = compare_decode(ref_logits, got_logits)
        ok = agree >= n_decode - MAX_TOP1_MISMATCH and d < TOL_DECODE_LOGIT
        report(ok, "3. decode parity",
               f"top-1 {agree}/{n_decode} agree, max|d| = {d:.4f}  (tol {TOL_DECODE_LOGIT})",
               f"decode diverges from stock: top-1 {agree}/{n_decode}, max|d| {d:.4f}")

        # 4. nothing compressed, nothing retrieved
        cold = sum(kv.cold_size(i) for i in range(n_layers))
        ev, inj = kv.dbg["evicted_tokens"], kv.dbg["injection_used"]
        report(cold == 0 and ev == 0 and inj == 0, "4. compression off",
               f"evicted={ev} cold={cold} injected={inj}  (all must be 0)",
               "compression or retrieval fired although it was disabled")
        log(f"       decode calls: triton={kv.dbg['triton_decode_calls']} "
            f"prefill={kv.dbg['triton_prefill_fallback']}")

        # 5. compression must not change the prompt's own attention
        small = max(8, int(ids.shape[1]) // 4)
        kv_c = make_kvctrl(budget=small)
        set_active_kvctrl(kv_c)
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=True,
                        past_key_values=build_hotcold_cache(model, kv_c))
        comp_prefill = out.logits[:, -1, :].float().cpu()
        evicted = kv_c.dbg["evicted_tokens"]
        d = (got_prefill - comp_prefill).abs().max().item()
        report(d < TOL_LOGIT and evicted > 0, "5. prefill vs budget",
               f"max|d| = {d:.4f} with budget {small} ({evicted} tokens evicted)",
               "prefill output changes when compression is on: prompt attention is "
               "running over the compressed cache instead of the full prompt"
               if evicted > 0 else "compression did not fire, so check 5 tested nothing")
    finally:
        uninstall_monkeypatch()
        set_active_kvctrl(None)

    # ---- baseline patch ---------------------------------------------------
    install_press_patch()
    try:
        for method in ("snapkv", "pyramidkv", "h2o"):
            cache = build_press_cache(model, PressConfig(method=method, max_capacity=10 ** 9))
            set_active_press_cache(cache)
            with torch.inference_mode():
                b_prefill, b_logits, _ = decode_trace(
                    model, ids, cache, n_decode, device, forced=ref_fed)
            dp = (ref_prefill - b_prefill).abs().max().item()
            dd, agree = compare_decode(ref_logits, b_logits)
            ok = dp < TOL_LOGIT and agree >= n_decode - MAX_TOP1_MISMATCH and dd < TOL_DECODE_LOGIT
            report(ok, f"6. baseline {method}",
                   f"prefill max|d| = {dp:.4f}, decode top-1 {agree}/{n_decode}, max|d| = {dd:.4f}",
                   f"{method} patch diverges from stock with compression off")
    finally:
        uninstall_press_patch()
        set_active_press_cache(None)

    return failures


def main():
    if not torch.cuda.is_available():
        print("SKIP: needs a CUDA GPU (the hot/cold decode path is a Triton kernel).")
        return 0

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print(f"model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map={"": 0}, low_cpu_mem_usage=True)
    model.eval()

    ids = tok(TEXT, return_tensors="pt").to(device)["input_ids"]
    print(f"prompt tokens: {int(ids.shape[1])} | decode steps: {N_DECODE}\n")

    failures = run_checks(model, ids, device)

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED\n")
        for f in failures:
            print(f"  - {f}")
        print("\nDo not trust benchmark numbers until these pass.")
        return 1
    print("All parity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
