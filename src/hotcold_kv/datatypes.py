from dataclasses import dataclass
import torch


# ============================================================
# Shape standardisation
# ============================================================
def _canon_kv_4d(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Ensure x is (B,Hk,R,D).
    Accepts extra singleton dims (e.g., (B,1,Hk,R,D) or (B,Hk,1,R,D)) and squeezes them.
    Never squeezes batch dim (dim=0) to avoid breaking B==1.
    """
    if not isinstance(x, torch.Tensor):
        raise RuntimeError(f"{name} must be a Tensor")
    while x.dim() > 4:
        squeezed = False
        for d in range(1, x.dim()):
            if x.shape[d] == 1:
                x = x.squeeze(d)
                squeezed = True
                break
        if not squeezed:
            break
    if x.dim() != 4:
        raise RuntimeError(f"{name} must be 4D (B,Hk,R,D); got {tuple(x.shape)}")
    return x

def _canon_br_2d(x: torch.Tensor, name: str) -> torch.Tensor:
    """Ensure x is (B,R)."""
    if not isinstance(x, torch.Tensor):
        raise RuntimeError(f"{name} must be a Tensor")
    while x.dim() > 2:
        squeezed = False
        for d in range(1, x.dim()):
            if x.shape[d] == 1:
                x = x.squeeze(d)
                squeezed = True
                break
        if not squeezed:
            break
    if x.dim() != 2:
        raise RuntimeError(f"{name} must be 2D (B,R); got {tuple(x.shape)}")
    return x


# ============================================================
# DTOs
# ============================================================
@dataclass
class RetrievedKV:
    k_pre: torch.Tensor          # (B,Hk,R,D) pre-RoPE keys on GPU
    v: torch.Tensor              # (B,Hk,R,D) values on GPU
    pos: torch.Tensor            # (B,R) int64 positions on GPU
    valid: torch.Tensor          # (B,R) bool
    ready: bool = False

@dataclass
class EvictedKV:
    k_pre_cpu: torch.Tensor      # pinned CPU float32 (B,Hk,Te,D)
    v_cpu: torch.Tensor          # pinned CPU float32 (B,Hk,Te,D)
    pos_cpu: torch.Tensor        # pinned CPU int64   (B,Te)
    evicted_idx: torch.Tensor    # CPU int64 padded with -1 (B,Te)
