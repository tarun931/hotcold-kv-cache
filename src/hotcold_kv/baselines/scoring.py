"""Pure-tensor scoring / selection primitives shared by the compression
baselines. These depend only on torch so they can be unit-tested without a
model or transformers installed."""

import torch
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand (B, Hk, S, D) -> (B, Hk*n_rep, S, D) for grouped-query attention."""
    if n_rep == 1:
        return x
    B, Hk, S, D = x.shape
    return x[:, :, None, :, :].expand(B, Hk, n_rep, S, D).reshape(B, Hk * n_rep, S, D)


def _group_query_to_kv(q: torch.Tensor, Hk: int) -> torch.Tensor:
    """Reduce query heads to key/value heads by averaging within each GQA group.
    q: (B, Hq, T, D) -> (B, Hk, T, D)."""
    B, Hq, T, D = q.shape
    if Hq == Hk:
        return q
    if Hq % Hk == 0:
        g = Hq // Hk
        return q.view(B, Hk, g, T, D).mean(dim=2)
    return q.mean(dim=1, keepdim=True).expand(B, Hk, T, D)


def _reduce_probs_to_kv(probs: torch.Tensor, Hk: int) -> torch.Tensor:
    """Reduce (B, Hq, T, S) attention probs to per-kv-head (B, Hk, T, S)."""
    B, Hq, T, S = probs.shape
    if Hq == Hk:
        return probs
    if Hq % Hk == 0:
        g = Hq // Hk
        return probs.view(B, Hk, g, T, S).mean(dim=2)
    return probs.mean(dim=1, keepdim=True).expand(B, Hk, T, S)


def pool1d(x: torch.Tensor, kernel_size: int, pooling: str) -> torch.Tensor:
    """1-D pooling over the last dim of (B, H, L), preserving length for odd kernels."""
    if kernel_size is None or kernel_size <= 1:
        return x
    pad = kernel_size // 2
    if pooling == "avgpool":
        return F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=pad)
    return F.max_pool1d(x, kernel_size=kernel_size, stride=1, padding=pad)


def snapkv_importance(
    q_post: torch.Tensor,     # (B, Hq, S, D) post-RoPE queries for the whole prompt
    keys: torch.Tensor,       # (B, Hk, S, D) post-RoPE keys
    window_size: int,
    scaling: float,
    kernel_size: int,
    pooling: str,
) -> torch.Tensor:
    """SnapKV per-kv-head token importance (B, Hk, S).

    Uses the last `window_size` query positions (the observation window) as probes,
    softmax-attends over the prompt (causally masked for the window rows), sums the
    resulting attention over the window and pools it."""
    B, Hq, S, D = q_post.shape
    Hk = keys.shape[1]
    W = min(window_size, S)

    q_tail = q_post[:, :, S - W:, :]                       # (B, Hq, W, D)
    q_for = _group_query_to_kv(q_tail, Hk)                 # (B, Hk, W, D)

    logits = torch.matmul(q_for, keys.transpose(2, 3)) * scaling   # (B, Hk, W, S)

    qpos = torch.arange(S - W, S, device=keys.device).view(1, 1, W, 1)
    kpos = torch.arange(S, device=keys.device).view(1, 1, 1, S)
    logits = logits.masked_fill(kpos > qpos, float("-inf"))

    probs = torch.softmax(logits.float(), dim=-1)          # (B, Hk, W, S)
    imp = probs.sum(dim=2)                                 # (B, Hk, S)
    imp = pool1d(imp, kernel_size, pooling)
    return imp


def h2o_prefill_scores(
    q_post: torch.Tensor,     # (B, Hq, S, D)
    keys: torch.Tensor,       # (B, Hk, S, D)
    scaling: float,
    chunk: int = 512,
) -> torch.Tensor:
    """H2O accumulated attention over the whole prefill, per-kv-head (B, Hk, S).
    Chunked over the query dim so the full (S, S) attention matrix is never
    materialised."""
    B, Hq, S, D = q_post.shape
    Hk = keys.shape[1]
    keys_rep = repeat_kv(keys, Hq // Hk) if Hq != Hk else keys        # (B, Hq, S, D)
    acc = torch.zeros(B, Hq, S, device=keys.device, dtype=torch.float32)
    kpos = torch.arange(S, device=keys.device).view(1, 1, 1, S)
    for st in range(0, S, chunk):
        en = min(st + chunk, S)
        qc = q_post[:, :, st:en, :]                                    # (B, Hq, cs, D)
        logits = torch.matmul(qc, keys_rep.transpose(2, 3)) * scaling  # (B, Hq, cs, S)
        qpos = torch.arange(st, en, device=keys.device).view(1, 1, -1, 1)
        logits = logits.masked_fill(kpos > qpos, float("-inf"))
        probs = torch.softmax(logits.float(), dim=-1)
        acc += probs.sum(dim=2)
    if Hq != Hk:
        g = Hq // Hk
        acc = acc.view(B, Hk, g, S).mean(dim=2)
    return acc                                                         # (B, Hk, S)


def keep_index(
    scores: torch.Tensor,     # (B, Hk, S) importance, higher = keep
    window_size: int,
    budget: int,
) -> torch.Tensor:
    """Select which key positions to retain: the top scoring prefix tokens plus the
    recent window. Returns per-kv-head indices (B, Hk, keep), sorted ascending."""
    B, Hk, S = scores.shape
    window = min(window_size, S)
    prefix_len = S - window
    keep_prefix = max(0, min(budget - window, prefix_len))

    device = scores.device
    recent = torch.arange(prefix_len, S, device=device).view(1, 1, window).expand(B, Hk, window)

    if keep_prefix <= 0:
        return recent.contiguous()

    prefix_scores = scores[:, :, :prefix_len]
    top = torch.topk(prefix_scores, k=keep_prefix, dim=-1).indices     # (B, Hk, keep_prefix)
    top, _ = torch.sort(top, dim=-1)
    keep = torch.cat([top, recent], dim=-1)                            # (B, Hk, keep_prefix+window)
    return keep


def gather_kv(t: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Gather (B, H, S, D) along S using per-head indices keep (B, H, k)."""
    B, H, S, D = t.shape
    idx = keep.unsqueeze(-1).expand(B, H, keep.shape[-1], D)
    return t.gather(2, idx)


def gather_scores(s: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Gather (B, H, S) along S using per-head indices keep (B, H, k)."""
    return s.gather(2, keep)
