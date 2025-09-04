"""
DiscoGPTransformerBlock: a single layer with masked attention and MLP routing.

- Accepts a residual tensor that already includes a "prev_head_idx" axis which
  enumerates contributions from earlier nodes (previous heads/MLP/output).
- Applies learnable edge masks to gate which previous nodes feed into the
  current layer's attention (separate masks for Q/K/V) and its MLP.
- Concatenates the new attention and MLP outputs onto the prev_head_idx axis so
  downstream layers can also be gated.
"""

import torch
import torch.nn as nn
import einops
from fancy_einsum import einsum
from transformer_lens.components import (
    RMSNorm,
    RMSNormPre,
    LayerNorm,
    LayerNormPre,
    GatedMLP,
    MLP,
    MoE,
)
import logging

from .attention import Attention


def gumbel_sigmoid(logits, gs_temp=1.0, eps=1e-10):
    """Straight-through Gumbel-sigmoid sampler for binary gates.

    - Samples Gumbel-like noise from two uniforms, adds to logits, divides by
      temperature, passes through sigmoid.
    - Applies a hard threshold at 0.5 in the forward pass but keeps gradients
      via the straight-through estimator.
    """
    uniform = logits.new_empty([2] + list(logits.shape)).uniform_(0, 1)
    noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
    res = torch.sigmoid((logits + noise) / gs_temp)
    res = ((res > 0.5).type_as(res) - res).detach() + res
    return res


class DiscoGPTransformerBlock(nn.Module):
    """One transformer block with per-edge gating for Q/K/V and MLP paths."""

    def __init__(self, cfg, block_index):
        super().__init__()
        self.cfg = cfg

        # --- Normalization layers (pre/post variants supported) ---
        if self.cfg.normalization_type == "LN":
            self.ln1 = LayerNorm(cfg)
            if not self.cfg.attn_only:
                self.ln2 = LayerNorm(cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            self.ln1 = LayerNormPre(cfg)
            if not self.cfg.attn_only:
                self.ln2 = LayerNormPre(cfg)
        elif self.cfg.normalization_type == "RMS":
            self.ln1 = RMSNorm(cfg)
            if not self.cfg.attn_only:
                self.ln2 = RMSNorm(cfg)
        elif self.cfg.normalization_type == "RMSPre":
            self.ln1 = RMSNormPre(cfg)
            if not self.cfg.attn_only:
                self.ln2 = RMSNormPre(cfg)
        elif self.cfg.normalization_type is None:
            self.ln1 = nn.Identity()
            if not self.cfg.attn_only:
                self.ln2 = nn.Identity()
        else:
            logging.warning(f"Invalid normalization_type passed in {self.cfg.normalization_type}")

        # Self-attention sublayer
        self.attn = Attention(cfg, "global", block_index)

        # --- Feed-forward / MLP sublayer ---
        if not self.cfg.attn_only:
            if self.cfg.num_experts:
                self.mlp = MoE(cfg)
            elif self.cfg.gated_mlp:
                self.mlp = GatedMLP(cfg)
            else:
                self.mlp = MLP(cfg)

        # Freeze all *weight* tensors by default; only mask logits should train
        for p in self.parameters():
            p.requires_grad = False

        # --- Edge mask logits (trainable) ---
        # Previous nodes up to this block: for each earlier layer we count its heads
        # (+1 for its MLP), plus the initial input node. Then we add +1 for the
        # initial input at this layer.
        prev_nodes = (cfg.n_heads + 1) * block_index + 1

        # Gate which previous nodes feed into Q/K/V projections of each head.
        # Shapes: [prev_nodes, n_heads]. Each column gates a head's Q/K/V input.
        self.edge_mask_attention_q_logits = nn.Parameter(
            torch.nn.init.normal_(torch.ones((prev_nodes, cfg.n_heads)), mean=self.cfg.logits_e_init, std=0.01),
            requires_grad=True,
        )
        self.edge_mask_attention_k_logits = nn.Parameter(
            torch.nn.init.normal_(torch.ones((prev_nodes, cfg.n_heads)), mean=self.cfg.logits_e_init, std=0.01),
            requires_grad=True,
        )
        self.edge_mask_attention_v_logits = nn.Parameter(
            torch.nn.init.normal_(torch.ones((prev_nodes, cfg.n_heads)), mean=self.cfg.logits_e_init, std=0.01),
            requires_grad=True,
        )

        # Gate which previous nodes feed into the MLP (post-attention concat)
        # Shape: [prev_nodes + n_heads]
        self.edge_mask_mlp_logits = nn.Parameter(
            torch.nn.init.normal_(torch.ones((prev_nodes + cfg.n_heads,)), mean=self.cfg.logits_e_init, std=0.01),
            requires_grad=True,
        )

    def forward(self, resid_pre):
        """Apply masked attention and masked MLP, then append outputs as nodes.

        Parameters
        ---
        resid_pre : Tensor, shape [batch, position, prev_head_idx, d_model]
            Residual stream including contributions from previous nodes.
        Returns
        ---
        residual : Tensor, shape [batch, position, prev_head_idx', d_model]
            Same as input but with attention heads (+ optionally MLP) concatenated
            on the prev_head_idx axis for downstream blocks.
        """
        # --- Sample or threshold masks ---
        if self.cfg.use_edge_masks:
            if self.cfg.use_deterministic_masks:
                sampled_edge_mask_attentions_q = torch.where(self.edge_mask_attention_q_logits > 0.0, 1.0, 0.0).to(self.cfg.device, dtype=self.cfg.dtype)
                sampled_edge_mask_attentions_k = torch.where(self.edge_mask_attention_k_logits > 0.0, 1.0, 0.0).to(self.cfg.device, dtype=self.cfg.dtype)
                sampled_edge_mask_attentions_v = torch.where(self.edge_mask_attention_v_logits > 0.0, 1.0, 0.0).to(self.cfg.device, dtype=self.cfg.dtype)
                sampled_edge_mask_mlp = torch.where(self.edge_mask_mlp_logits > 0.0, 1.0, 0.0).to(self.cfg.device, dtype=self.cfg.dtype)
            else:
                sampled_edge_mask_attentions_q = gumbel_sigmoid(self.edge_mask_attention_q_logits, self.cfg.gs_temp_edge)
                sampled_edge_mask_attentions_k = gumbel_sigmoid(self.edge_mask_attention_k_logits, self.cfg.gs_temp_edge)
                sampled_edge_mask_attentions_v = gumbel_sigmoid(self.edge_mask_attention_v_logits, self.cfg.gs_temp_edge)
                sampled_edge_mask_mlp = gumbel_sigmoid(self.edge_mask_mlp_logits, self.cfg.gs_temp_edge)
            if self.cfg.use_reverse_masks:
                sampled_edge_mask_attentions_q = 1.0 - sampled_edge_mask_attentions_q
                sampled_edge_mask_attentions_k = 1.0 - sampled_edge_mask_attentions_k
                sampled_edge_mask_attentions_v = 1.0 - sampled_edge_mask_attentions_v
                sampled_edge_mask_mlp = 1.0 - sampled_edge_mask_mlp
        else:
            # Disable mask by setting everything to 1 (no gating).
            sampled_edge_mask_attentions_q = torch.ones(self.edge_mask_attention_q_logits.shape).to(resid_pre.device, dtype=self.cfg.dtype)
            sampled_edge_mask_attentions_k = torch.ones(self.edge_mask_attention_k_logits.shape).to(resid_pre.device, dtype=self.cfg.dtype)
            sampled_edge_mask_attentions_v = torch.ones(self.edge_mask_attention_v_logits.shape).to(resid_pre.device, dtype=self.cfg.dtype)
            sampled_edge_mask_mlp = torch.ones(self.edge_mask_mlp_logits.shape).to(resid_pre.device, dtype=self.cfg.dtype)

        # resid_pre: [batch, position, prev_head_idx, d_model]
        # Project gated residuals into per-head Q/K/V inputs
        masked_residuals_q = einsum(
            "batch position prev_head_idx d_model, prev_head_idx n_heads -> batch position n_heads d_model",
            resid_pre,
            sampled_edge_mask_attentions_q,
        )
        masked_residuals_k = einsum(
            "batch position prev_head_idx d_model, prev_head_idx n_heads -> batch position n_heads d_model",
            resid_pre,
            sampled_edge_mask_attentions_k,
        )
        masked_residuals_v = einsum(
            "batch position prev_head_idx d_model, prev_head_idx n_heads -> batch position n_heads d_model",
            resid_pre,
            sampled_edge_mask_attentions_v,
        )

        # Apply normalization before attention
        normalized_resid_pre_q = self.ln1(masked_residuals_q)
        normalized_resid_pre_k = self.ln1(masked_residuals_k)
        normalized_resid_pre_v = self.ln1(masked_residuals_v)

        # Self-attention returns per-head outputs shaped [batch, position, n_heads, d_model]
        attn_out = self.attn(normalized_resid_pre_q, normalized_resid_pre_k, normalized_resid_pre_v)

        # Concatenate new attention nodes onto the prev_head_idx axis
        residual = torch.cat((resid_pre, attn_out), dim=2)

        # Gate which nodes feed the MLP, then (optionally) run MLP
        masked_mlp_residual = einsum(
            "batch position prev_head_idx d_model, prev_head_idx -> batch position d_model",
            residual,
            sampled_edge_mask_mlp,
        )

        normalized_resid_mid = self.ln2(masked_mlp_residual)
        mlp_out = self.mlp(normalized_resid_mid)
        mlp_out = einops.rearrange(mlp_out, "batch position d_model -> batch position 1 d_model")

        # Append MLP node for downstream layers
        residual = torch.cat((residual, mlp_out), dim=2)

        return residual
