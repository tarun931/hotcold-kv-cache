import threading
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from typing import Deque, List, Optional, Tuple
from .datatypes import EvictedKV, RetrievedKV, _canon_br_2d, _canon_kv_4d


# ============================================================
# KVController (PER-LAYER cold store + retrieval + staging)
# ============================================================
class KVController:
    def __init__(
        self,
        max_capacity_prompt: int = 128,
        topk_ratio: float = 0.15,
        recent_size: int = 32,

        cpu_buffer_size: int = 8192,
        faiss_dim: int = 64,
        normalize_for_ip: bool = True,

        window_size: int = 16,
        query_pool_size: int = 4,
        lookahead_k: int = 32,
        hold_retrieved: bool = False,

        enable_asserts: bool = True,
        profile_overlap: bool = True,
        force_wait_for_prefetch_diagnostic: bool = False,
    ):
        self.max_capacity_prompt = int(max_capacity_prompt)
        self.topk_ratio = float(topk_ratio)
        self.recent_size = int(recent_size)

        self.cpu_buffer_size = int(cpu_buffer_size)
        self.faiss_dim = int(faiss_dim)
        self.normalize_for_ip = bool(normalize_for_ip)

        self.window_size = int(window_size)
        self.query_pool_size = int(query_pool_size)
        self.lookahead_k = int(lookahead_k)
        # False: retrieved tokens are attended at the one decode step their copy
        # lands. True: they stay in attention until the next retrieval replaces
        # them (held per layer on the GPU).
        self.hold_retrieved = bool(hold_retrieved)

        self.enable_asserts = bool(enable_asserts)
        self.profile_overlap = bool(profile_overlap)
        self.force_wait_for_prefetch_diagnostic = bool(force_wait_for_prefetch_diagnostic)

        self._inited = False
        self.B = None
        self.Hk = None
        self.D = None
        self.n_layers = None

        # per-layer: [L][B] rings + indices
        self.cpu_kpre_ring: List[List[Deque[torch.Tensor]]] = []
        self.cpu_v_ring: List[List[Deque[torch.Tensor]]] = []
        self.cpu_pos_ring: List[List[Deque[int]]] = []
        self.index: List[List[faiss.Index]] = []
        self.faiss_lock: List[List[threading.RLock]] = []
        self._num_vecs: List[List[int]] = []

        # per-layer query pooling + scheduling
        self.step_count: List[int] = []
        self.query_window: List[Deque[torch.Tensor]] = []

        # per-layer retrieval worker state
        self._retrieval_thread: List[Optional[threading.Thread]] = []
        self._retrieval_lock: List[threading.Lock] = []
        self._retrieval_ready_cpu: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = []

        # per-layer prefetch staging
        self._stage_lock: List[threading.Lock] = []
        self._staged_gpu: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        self._staged_event: List[Optional[torch.cuda.Event]] = []
        self._prefetch_stream: Optional[torch.cuda.Stream] = None
        self._copy_start_ev: List[Optional[torch.cuda.Event]] = []
        self._copy_end_ev: List[Optional[torch.cuda.Event]] = []

        self.last_injected_len = 0
        self.held_mem: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = []

        self.dbg = {
            "compress_calls": 0,
            "evicted_tokens": 0,
            "faiss_add": 0,

            "retrieval_launched": 0,
            "retrieval_ready_cpu": 0,
            "retrieval_skipped_no_cold": 0,
            "retrieval_ready_empty": 0,

            "prefetch_launched": 0,
            "prefetch_ready": 0,
            "prefetch_skipped_empty": 0,

            "injection_used": 0,       # decode steps that attended retrieved tokens
            "injection_new": 0,        # hold_retrieved: fresh sets that replaced the held one
            "injection_skipped_no_valid": 0,

            "prefetch_copy_ms_sum": 0.0,
            "prefetch_copy_ms_count": 0,
            "prefetch_not_ready_checks": 0,
            "main_waited_for_prefetch": 0,

            "triton_decode_calls": 0,
            "triton_prefill_fallback": 0,
        }

    def _build_index(self) -> faiss.Index:
        return faiss.IndexFlatIP(self.faiss_dim)

    def _lazy_init(self, k_pre: torch.Tensor, n_layers: int):
        if self._inited:
            return

        B, Hk, _, D = k_pre.shape
        self.B, self.Hk, self.D = int(B), int(Hk), int(D)
        self.n_layers = int(n_layers)

        L = self.n_layers
        self.cpu_kpre_ring = [[deque(maxlen=self.cpu_buffer_size) for _ in range(self.B)] for _ in range(L)]
        self.cpu_v_ring    = [[deque(maxlen=self.cpu_buffer_size) for _ in range(self.B)] for _ in range(L)]
        self.cpu_pos_ring  = [[deque(maxlen=self.cpu_buffer_size) for _ in range(self.B)] for _ in range(L)]

        self.index = [[self._build_index() for _ in range(self.B)] for _ in range(L)]
        self.faiss_lock = [[threading.RLock() for _ in range(self.B)] for _ in range(L)]
        self._num_vecs = [[0 for _ in range(self.B)] for _ in range(L)]

        self.step_count = [0 for _ in range(L)]
        self.query_window = [deque(maxlen=self.query_pool_size) for _ in range(L)]

        self._retrieval_thread = [None for _ in range(L)]
        self._retrieval_lock = [threading.Lock() for _ in range(L)]
        self._retrieval_ready_cpu = [None for _ in range(L)]

        self._stage_lock = [threading.Lock() for _ in range(L)]
        self._staged_gpu = [None for _ in range(L)]
        self._staged_event = [None for _ in range(L)]
        self._copy_start_ev = [None for _ in range(L)]
        self._copy_end_ev = [None for _ in range(L)]
        self.held_mem = [None for _ in range(L)]

        if torch.cuda.is_available():
            self._prefetch_stream = torch.cuda.Stream()

        self._inited = True

    def cold_size(self, layer_idx: int) -> int:
        if not self._inited:
            return 0
        li = int(layer_idx)
        return sum(len(self.cpu_kpre_ring[li][b]) for b in range(self.B))

    def _pad_or_crop(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.shape[-1] > dim:
            return x[..., :dim]
        if x.shape[-1] < dim:
            return F.pad(x, (0, dim - x.shape[-1]))
        return x

    def _normalize(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if not self.normalize_for_ip:
            return x
        return F.normalize(x, dim=dim)

    def _k_token_to_vec_np(self, kpre_token_cpu: torch.Tensor) -> np.ndarray:
        v = kpre_token_cpu.mean(dim=0)  # (D)
        v = self._pad_or_crop(v, self.faiss_dim)
        v = self._normalize(v, dim=0)
        return v.numpy().astype("float32", copy=False)

    def _q_to_vecs_np(self, q_pre: torch.Tensor) -> np.ndarray:
        q = q_pre.detach().mean(dim=1).mean(dim=1)  # (B,D)
        q = self._pad_or_crop(q, self.faiss_dim)
        q = self._normalize(q, dim=-1)
        return q.float().cpu().numpy().astype("float32", copy=False)

    def add_evicted_to_cold_store(self, layer_idx: int, ev: EvictedKV):
        if not self._inited:
            raise RuntimeError("KVController not initialized")
        li = int(layer_idx)

        B, Hk, Te, D = ev.k_pre_cpu.shape
        for b in range(B):
            with self.faiss_lock[li][b]:
                new_tokens = []

                for t in range(Te):
                    if int(ev.evicted_idx[b, t].item()) < 0:
                        continue

                    tok_kpre = ev.k_pre_cpu[b, :, t, :].contiguous()
                    tok_pos = int(ev.pos_cpu[b, t].item())

                    # store
                    self.cpu_kpre_ring[li][b].append(tok_kpre)
                    self.cpu_v_ring[li][b].append(ev.v_cpu[b, :, t, :].contiguous())
                    self.cpu_pos_ring[li][b].append(tok_pos)

                    new_tokens.append(tok_kpre)

                if not new_tokens:
                    continue

                # A batch larger than the ring only keeps its last
                # cpu_buffer_size tokens; index just those.
                m = len(new_tokens)
                keep = min(m, self.cpu_buffer_size)
                vecs = np.stack([self._k_token_to_vec_np(tok) for tok in new_tokens[m - keep:]], axis=0)
                index = self.index[li][b]
                index.add(vecs)

                # Once full, the ring drops its oldest tokens on append. Drop
                # the same rows from the index, or every later search returns
                # ids that point at the wrong ring slots. IndexFlat ids are row
                # positions and remove_ids keeps the remaining rows in order,
                # so index row i is ring slot i again.
                ring_len = len(self.cpu_kpre_ring[li][b])
                if index.ntotal > ring_len:
                    index.remove_ids(faiss.IDSelectorRange(0, index.ntotal - ring_len))
                self._num_vecs[li][b] = int(index.ntotal)

                self.dbg["faiss_add"] += keep

                if self.enable_asserts:
                    assert self._num_vecs[li][b] == ring_len

    def update_query_window(self, layer_idx: int, q_pre: torch.Tensor):
        self.query_window[int(layer_idx)].append(q_pre.detach())

    def _pooled_query_np(self, layer_idx: int) -> Optional[np.ndarray]:
        qwin = self.query_window[int(layer_idx)]
        if len(qwin) == 0:
            return None
        q_cat = torch.cat(list(qwin), dim=2)
        return self._q_to_vecs_np(q_cat)

    def _retrieval_worker(self, layer_idx: int, q_vecs: np.ndarray):
        li = int(layer_idx)
        if (not self._inited) or self.B is None or self.Hk is None or self.D is None:
            return

        B, Hk, D = int(self.B), int(self.Hk), int(self.D)
        R = int(self.lookahead_k)
        cpu = torch.device("cpu")

        kpre_cpu = torch.zeros((B, Hk, R, D), device=cpu, pin_memory=True, dtype=torch.float32)
        v_cpu    = torch.zeros((B, Hk, R, D), device=cpu, pin_memory=True, dtype=torch.float32)
        pos_cpu  = torch.zeros((B, R), device=cpu, pin_memory=True, dtype=torch.long)
        valid_cpu= torch.zeros((B, R), device=cpu, pin_memory=True, dtype=torch.bool)

        any_valid = False

        for b in range(B):
            with self.faiss_lock[li][b]:
                n = len(self.cpu_kpre_ring[li][b])
                if n == 0:
                    continue
                r_eff = min(R, n)

                q = q_vecs[b:b+1].astype("float32", copy=False)
                _, I = self.index[li][b].search(q, r_eff)
                I = I.reshape(-1)

                for j in range(r_eff):
                    idx = int(I[j])          # index row == ring slot
                    if idx < 0 or idx >= n:
                        continue
                    kpre_cpu[b, :, j, :].copy_(self.cpu_kpre_ring[li][b][idx], non_blocking=False)
                    v_cpu[b, :, j, :].copy_(self.cpu_v_ring[li][b][idx], non_blocking=False)
                    pos_cpu[b, j] = int(self.cpu_pos_ring[li][b][idx])
                    valid_cpu[b, j] = True
                    any_valid = True

        with self._retrieval_lock[li]:
            if not any_valid:
                self._retrieval_ready_cpu[li] = None
                self.dbg["retrieval_ready_empty"] += 1
                return
            self._retrieval_ready_cpu[li] = (kpre_cpu, v_cpu, pos_cpu, valid_cpu)
            self.dbg["retrieval_ready_cpu"] += 1

    def maybe_launch_async_retrieval(self, layer_idx: int):
        if self.lookahead_k <= 0:
            return
        li = int(layer_idx)
        self.step_count[li] += 1

        if (self.step_count[li] % self.window_size) != 0:
            return
        if not self._inited:
            return
        if self.cold_size(li) == 0:
            self.dbg["retrieval_skipped_no_cold"] += 1
            return

        q_vecs = self._pooled_query_np(li)
        if q_vecs is None:
            return

        with self._retrieval_lock[li]:
            th = self._retrieval_thread[li]
            if th is not None and th.is_alive():
                return
            self._retrieval_thread[li] = threading.Thread(
                target=self._retrieval_worker,
                args=(li, q_vecs),
                daemon=True,
            )
            self._retrieval_thread[li].start()
            self.dbg["retrieval_launched"] += 1

    def maybe_prefetch_to_gpu(self, layer_idx: int, device: torch.device, dtype: torch.dtype):
        li = int(layer_idx)
        if device.type != "cuda" or self._prefetch_stream is None:
            return

        with self._retrieval_lock[li]:
            pack = self._retrieval_ready_cpu[li]
            self._retrieval_ready_cpu[li] = None

        if pack is None:
            return

        kpre_cpu, v_cpu, pos_cpu, valid_cpu = pack
        if not bool(valid_cpu.any().item()):
            self.dbg["prefetch_skipped_empty"] += 1
            return

        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        ready_ev = torch.cuda.Event()

        with torch.cuda.stream(self._prefetch_stream):
            copy_start.record(self._prefetch_stream)
            # pinned CPU -> GPU non_blocking 
            kpre_gpu  = kpre_cpu.to(device=device, dtype=dtype, non_blocking=True)
            v_gpu     = v_cpu.to(device=device, dtype=dtype, non_blocking=True)
            pos_gpu   = pos_cpu.to(device=device, non_blocking=True)
            valid_gpu = valid_cpu.to(device=device, non_blocking=True)
            copy_end.record(self._prefetch_stream)
            ready_ev.record(self._prefetch_stream)

        with self._stage_lock[li]:
            self._staged_gpu[li] = (kpre_gpu, v_gpu, pos_gpu, valid_gpu)
            self._staged_event[li] = ready_ev
            self._copy_start_ev[li] = copy_start
            self._copy_end_ev[li] = copy_end
            self.dbg["prefetch_launched"] += 1

    def poll_staged_gpu(self, layer_idx: int) -> RetrievedKV:
        li = int(layer_idx)
        if not torch.cuda.is_available():
            return RetrievedKV(
                k_pre=torch.empty(0), v=torch.empty(0),
                pos=torch.empty(0, dtype=torch.long),
                valid=torch.empty(0, dtype=torch.bool),
                ready=False,
            )

        with self._stage_lock[li]:
            if self._staged_gpu[li] is None or self._staged_event[li] is None:
                return RetrievedKV(
                    k_pre=torch.empty(0), v=torch.empty(0),
                    pos=torch.empty(0, dtype=torch.long),
                    valid=torch.empty(0, dtype=torch.bool),
                    ready=False,
                )

            if not self._staged_event[li].query():
                self.dbg["prefetch_not_ready_checks"] += 1
                if self.force_wait_for_prefetch_diagnostic:
                    self.dbg["main_waited_for_prefetch"] += 1
                    self._staged_event[li].synchronize()
                else:
                    return RetrievedKV(
                        k_pre=torch.empty(0), v=torch.empty(0),
                        pos=torch.empty(0, dtype=torch.long),
                        valid=torch.empty(0, dtype=torch.bool),
                        ready=False,
                    )

            kpre_gpu, v_gpu, pos_gpu, valid_gpu = self._staged_gpu[li]
            self._staged_gpu[li] = None

            copy_start = self._copy_start_ev[li]
            copy_end = self._copy_end_ev[li]
            self._copy_start_ev[li] = None
            self._copy_end_ev[li] = None

            self._staged_event[li] = None
            self.dbg["prefetch_ready"] += 1

        if self.profile_overlap and copy_start is not None and copy_end is not None:
            torch.cuda.synchronize()
            ms = float(copy_start.elapsed_time(copy_end))
            self.dbg["prefetch_copy_ms_sum"] += ms
            self.dbg["prefetch_copy_ms_count"] += 1

        if kpre_gpu.numel() == 0:
            return RetrievedKV(k_pre=kpre_gpu, v=v_gpu, pos=pos_gpu, valid=valid_gpu, ready=False)

        kpre_gpu  = _canon_kv_4d(kpre_gpu, "staged.k_pre")
        v_gpu     = _canon_kv_4d(v_gpu, "staged.v")
        pos_gpu   = _canon_br_2d(pos_gpu, "staged.pos")
        valid_gpu = _canon_br_2d(valid_gpu, "staged.valid")

        if not bool(valid_gpu.any().item()):
            return RetrievedKV(k_pre=kpre_gpu, v=v_gpu, pos=pos_gpu, valid=valid_gpu, ready=False)

        return RetrievedKV(k_pre=kpre_gpu, v=v_gpu, pos=pos_gpu, valid=valid_gpu, ready=True)
