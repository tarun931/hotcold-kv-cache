"""Minimal end-to-end demo: forces KV eviction with a small hot capacity and a
long prompt, then generates with the hot/cold cache installed.

Run:  python scripts/demo.py
Requires a CUDA GPU and access to meta-llama/Llama-3.2-1B on the HF Hub.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hotcold_kv import (
    KVController,
    set_active_kvctrl,
    set_active_rotary_emb,
    install_monkeypatch,
    build_hotcold_cache,
)


def main():
    MODEL_NAME = "meta-llama/Llama-3.2-1B"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # required for  re-rotation
    set_active_rotary_emb(model.model.rotary_emb)

    # Force eviction: small hot capacity + long prompt
    kvctrl = KVController(
        max_capacity_prompt=128,   
        topk_ratio=0.25,
        recent_size=32,

        cpu_buffer_size=8192,
        faiss_dim=64,
        window_size=8,
        query_pool_size=4,
        lookahead_k=32,

        enable_asserts=False,
        profile_overlap=True,
        force_wait_for_prefetch_diagnostic=False,
    )
    set_active_kvctrl(kvctrl)

    install_monkeypatch(enable_debug=True)

    past_key_values = build_hotcold_cache(model, kvctrl)
    prompt = ("The quick brown fox jumps over the lazy dog. " * 600) + "\nExplain KV cache briefly.\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,   
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.05,
            use_cache=True,
            past_key_values=past_key_values,
        )

    print(tokenizer.decode(out[0], skip_special_tokens=True))
    print("hot layer0 len:", past_key_values.get_seq_length(0))
    print("cold layer0 size:", kvctrl.cold_size(0))
    print("FINAL dbg:", kvctrl.dbg)

if __name__ == "__main__":
    main()
