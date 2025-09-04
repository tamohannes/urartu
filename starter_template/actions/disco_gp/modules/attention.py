import torch
import torch.nn as nn
from torch import Tensor

import einops

from fancy_einsum import einsum
from transformer_lens.components import (
    RMSNorm,
)

class Attention(nn.Module):
    def __init__(self, cfg, attn_type, layer_id):
        super().__init__()
        self.cfg = cfg

        self.is_gqa = cfg.n_key_value_heads is not None

        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        # nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        # nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model)))


        # We need to load a different name of parameters if GQA is enabled
        if self.is_gqa:
            # Key projection
            self._W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
            # Value projection
            self._W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
            # Key bias
            self._b_K = nn.Parameter(torch.empty(cfg.n_heads, self.cfg.d_head))
            # Value bias
            self._b_V = nn.Parameter(torch.empty(cfg.n_heads, self.cfg.d_head))
        else:
            self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
            # nn.init.normal_(self.W_K, std=self.cfg.init_range)
            self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
            self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
            # nn.init.normal_(self.W_V, std=self.cfg.init_range)
            self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))

        self.attn_type = attn_type
        self.layer_id = layer_id

        # Attention mask value preallocated.
        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device="cuda"))

        if cfg.use_attn_scale:
            self.attn_scale = cfg.attn_scale

        if cfg.use_qk_norm:
            self.q_norm = RMSNorm(cfg, length=cfg.d_head)
            self.k_norm = RMSNorm(cfg, length=cfg.d_head)
        else:
            self.q_norm = None
            self.k_norm = None

        if self.cfg.positional_embedding_type == "rotary":
            sin, cos = self.calculate_sin_cos_rotary(
                self.cfg.rotary_dim,
                self.cfg.n_ctx,
                base=self.cfg.rotary_base,
                dtype=self.cfg.dtype,
            )
            self.register_buffer("rotary_sin", sin)
            self.register_buffer("rotary_cos", cos)


    def calculate_sin_cos_rotary(
        self,
        rotary_dim: int,
        n_ctx: int,
        base: int = 10000,
        dtype: torch.dtype = torch.float32,
    ):
        high_precision = torch.float32 if dtype != torch.float64 else torch.float64
        pos = torch.arange(n_ctx, dtype=high_precision)
        dim = torch.arange(rotary_dim // 2, dtype=high_precision)

        # A set of frequencies evenly spaced in log space
        freq = base ** (dim / (rotary_dim / 2))
        if self.cfg.rotary_adjacent_pairs:
            freq = einops.repeat(freq, "d -> (d 2)")
        else:
            freq = einops.repeat(freq, "d -> (2 d)")
        # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
        angles = pos[:, None] / freq[None, :]
        return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)


    def rotate_every_two(self, x):
        rot_x = x.clone()
        if self.cfg.rotary_adjacent_pairs:
            rot_x[..., ::2] = -x[..., 1::2]
            rot_x[..., 1::2] = x[..., ::2]
        else:
            n = x.size(-1) // 2
            rot_x[..., :n] = -x[..., n:]
            rot_x[..., n:] = x[..., :n]

        return rot_x


    def apply_rotary(self, x):
        # Only apply rotary to first rotary_dim dimensions (eg, if rotary_dim=64 and d_head=256, only apply to first 1/4 of dimensions)
        x_pos = x.size(1)
        x_rot = x[..., : self.cfg.rotary_dim]
        x_pass = x[..., self.cfg.rotary_dim :]
        x_flip = self.rotate_every_two(x_rot)

        rotary_cos = self.rotary_cos[
            None, 0 : x_pos, None, :
        ]
        rotary_sin = self.rotary_sin[
            None, 0 : x_pos, None, :
        ]
        x_rotated = x_rot * rotary_cos + x_flip * rotary_sin

        return torch.cat([x_rotated, x_pass], dim=-1)


    def forward(self, normalized_resid_pre_q, normalized_resid_pre_k, normalized_resid_pre_v):
        # normalized_resid_pre: [batch, position, d_model]

        q = einsum("batch query_pos n_heads d_model, n_heads d_model d_head -> batch query_pos n_heads d_head", normalized_resid_pre_q, self.W_Q) + self.b_Q
        k = einsum("batch key_pos n_heads d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre_k, self.get_W_K()) + self.get_b_K()
        v = einsum("batch key_pos n_heads d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre_v, self.get_W_V()) + self.get_b_V()

        if self.cfg.use_qk_norm:
            q = self.apply_qk_norm(q, self.q_norm)
            k = self.apply_qk_norm(k, self.k_norm)

        if self.cfg.positional_embedding_type == "rotary":
            q = self.apply_rotary(q)
            k = self.apply_rotary(k)

        attn_scores = einsum("batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attn_scores = attn_scores / self.attn_scale
      
        if self.cfg.attention_dir == "causal":
            attn_scores = self.apply_causal_mask(attn_scores)

        pattern = attn_scores.softmax(dim=-1) # [batch, n_head, query_pos, key_pos]

        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head", pattern, v)

        attn_out = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos n_heads d_model", z, self.W_O) + (self.b_O / self.cfg.n_heads)
        return attn_out


    def apply_qk_norm(
        self, x: Tensor, # [batch pos head_index d_head],
        norm_module: RMSNorm
    ) -> Tensor: # [batch pos head_index d_head]
        batch, pos, n_heads, d_head = x.shape
        x_reshaped = x.reshape(-1, d_head)
        x_normed = norm_module(x_reshaped)
        return x_normed.reshape(batch, pos, n_heads, d_head)


    def apply_causal_mask(self, attn_scores):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


    def get_b_K(self):
        if self.is_gqa:
            return self._b_K
        else:
            return self.b_K


    def get_W_K(self):
        if self.is_gqa:
            return self._W_K
        else:
            return self.W_K


    def get_b_V(self):
        if self.is_gqa:
            return self._b_V
        else:
            return self.b_V


    def get_W_V(self):
        if self.is_gqa:
            return self._W_V
        else:
            return self.W_V