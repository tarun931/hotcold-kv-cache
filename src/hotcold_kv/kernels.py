import torch
import triton
import triton.language as tl


# ============================================================
# Triton 
# ============================================================
@triton.jit
def evict_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Mask_ptr,
    AttnOutput_ptr, TempLoss_ptr, L_ptr,
    scaling,
    N_REP,
    N_Q_HEADS,
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_h_kv, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h_kv, stride_v_s, stride_v_d,
    stride_mask_b, stride_mask_s,
    stride_ao_b, stride_ao_h, stride_ao_d,
    stride_loss_b, stride_loss_h, stride_loss_s,
    stride_l_b, stride_l_h, stride_l_s,
    D_HEAD: tl.constexpr,
    SEQ_LEN_KV: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    d_offs = tl.arange(0, D_HEAD)

    q_base_ptr = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q_vec = tl.load(q_base_ptr + d_offs * stride_q_d)

    kv_head_idx = pid_h // N_REP
    k_base = K_ptr + pid_b * stride_k_b + kv_head_idx * stride_k_h_kv
    v_base = V_ptr + pid_b * stride_v_b + kv_head_idx * stride_v_h_kv
    m_base = Mask_ptr + pid_b * stride_mask_b
    loss_base = TempLoss_ptr + pid_b * stride_loss_b + pid_h * stride_loss_h

    acc_o = tl.zeros((D_HEAD,), dtype=tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)

    for s_start in range(0, SEQ_LEN_KV, BLOCK_S):
        s = s_start + tl.arange(0, BLOCK_S)
        inb = s < SEQ_LEN_KV

        k_ptrs = k_base + s[:, None] * stride_k_s + d_offs[None, :] * stride_k_d
        k_blk = tl.load(k_ptrs, mask=inb[:, None], other=0.0)

        scores = tl.sum(q_vec[None, :].to(tl.float32) * k_blk.to(tl.float32), axis=1) * scaling

        m_ptrs = m_base + s * stride_mask_s
        m = tl.load(m_ptrs, mask=inb, other=False)
        scores = tl.where(m & inb, scores, float(-1e35))

        p = tl.exp(scores)
        l_i += tl.sum(p, axis=0)

        v_ptrs = v_base + s[:, None] * stride_v_s + d_offs[None, :] * stride_v_d
        v_blk = tl.load(v_ptrs, mask=inb[:, None], other=0.0)

        attn_tmp = p[:, None] * v_blk.to(tl.float32)
        acc_o += tl.sum(attn_tmp, axis=0)

        attn_loss = tl.sum(tl.abs(attn_tmp), axis=1) / D_HEAD
        tl.store(loss_base + s * stride_loss_s, attn_loss, mask=inb)

    out = (acc_o / l_i).to(Q_ptr.dtype.element_ty)

    ao_base = AttnOutput_ptr + pid_b * stride_ao_b + pid_h * stride_ao_h
    tl.store(ao_base + d_offs * stride_ao_d, out)

    tl.store(L_ptr + pid_b * stride_l_b + pid_h * stride_l_h + tl.arange(0, 1) * stride_l_s, l_i)


def triton_evict_attention_forward(
    query: torch.Tensor,          # [B, N_Q_HEADS, 1, D_HEAD]
    key: torch.Tensor,            # [B, N_KV_HEADS, S_KV, D_HEAD]
    value: torch.Tensor,          # [B, N_KV_HEADS, S_KV, D_HEAD]
    attention_mask: torch.Tensor, # [B, S_KV], dtype=torch.bool (True means keep) 
    num_key_value_groups: int,    # This is N_REP
    scaling: float,
    dropout: float = 0.0,
    block_s: int = 64,            # Block size for SEQ_LEN_KV, can be tuned
):
    assert query.is_cuda and key.is_cuda and value.is_cuda and attention_mask.is_cuda
    if dropout != 0.0:
        raise ValueError("dropout must be 0.0 , because we are not in training")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    attention_mask = attention_mask.contiguous()

    B, Hq, _, D = query.shape
    _, Hk, S, _ = key.shape

    if Hq // num_key_value_groups != Hk:
        raise ValueError(f"Hq({Hq})//N_REP({num_key_value_groups}) must equal Hk({Hk})")

    attn_output = torch.empty_like(query)
    temp_loss = torch.zeros((B, Hq, S), dtype=torch.float32, device=query.device)
    tmp_l = torch.zeros((B, Hq, 1), dtype=torch.float32, device=query.device)

    stride_q_b, stride_q_h, _, stride_q_d = query.stride()
    stride_k_b, stride_k_h, stride_k_s, stride_k_d = key.stride()
    stride_v_b, stride_v_h, stride_v_s, stride_v_d = value.stride()
    stride_m_b, stride_m_s = attention_mask.stride()
    stride_ao_b, stride_ao_h, _, stride_ao_d = attn_output.stride()
    stride_loss_b, stride_loss_h, stride_loss_s = temp_loss.stride()
    stride_l_b, stride_l_h, stride_l_s = tmp_l.stride()

    grid = (B, Hq)
    num_warps = 8 if D >= 128 else 4

    evict_fwd_kernel[grid](
        Q_ptr=query, K_ptr=key, V_ptr=value, Mask_ptr=attention_mask,
        AttnOutput_ptr=attn_output, TempLoss_ptr=temp_loss, L_ptr=tmp_l,
        scaling=scaling,
        N_REP=num_key_value_groups,
        N_Q_HEADS=Hq,
        stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_d=stride_q_d,
        stride_k_b=stride_k_b, stride_k_h_kv=stride_k_h, stride_k_s=stride_k_s, stride_k_d=stride_k_d,
        stride_v_b=stride_v_b, stride_v_h_kv=stride_v_h, stride_v_s=stride_v_s, stride_v_d=stride_v_d,
        stride_mask_b=stride_m_b, stride_mask_s=stride_m_s,
        stride_ao_b=stride_ao_b, stride_ao_h=stride_ao_h, stride_ao_d=stride_ao_d,
        stride_loss_b=stride_loss_b, stride_loss_h=stride_loss_h, stride_loss_s=stride_loss_s,
        stride_l_b=stride_l_b, stride_l_h=stride_l_h, stride_l_s=stride_l_s,
        D_HEAD=D, SEQ_LEN_KV=S, BLOCK_S=block_s,
        num_warps=num_warps,
    )

    final_loss = (temp_loss / tmp_l).mean(dim=1).to(query.dtype)  # [B, S_KV]
    attn_out_b1hd = attn_output.transpose(1, 2).contiguous()      # [B, 1, N_Q_HEADS, D]
    evict_idx = torch.argmin(final_loss, dim=-1)                  # [B]
    return attn_out_b1hd, evict_idx
