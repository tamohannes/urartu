from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast, overload
from typing_extensions import Literal

import os
import gc
from pathlib import Path

from tqdm.auto import tqdm
from pprint import pprint

import numpy as np
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from fancy_einsum import einsum

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformer_lens.components import (
    Embed,
    PosEmbed,
    RMSNorm,
    RMSNormPre,
    LayerNorm,
    LayerNormPre,
    Unembed,
    GatedMLP,
    MLP,
    MoE,
    Attention
)
from transformer_lens import HookedTransformer

from .data import setup_task
from .evaluation import compute_complete_loss, compute_faith_loss
from .utils import schedule_epoch_lambda
from .configs import Config

def gumbel_sigmoid(logits, gs_temp=1., eps=1e-10):
    uniform = logits.new_empty([2]+list(logits.shape)).uniform_(0, 1)
    noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
    res = torch.sigmoid((logits + noise) / gs_temp)
    res = ((res > 0.5).type_as(res) - res).detach() + res
    return res

class Attention(nn.Module):
    def __init__(self, cfg, attn_type, layer_id):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        # nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        # nn.init.normal_(self.W_K, std=self.cfg.init_range)
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        # nn.init.normal_(self.W_V, std=self.cfg.init_range)
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))

        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        # nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model)))

        self.attn_type = attn_type
        self.layer_id = layer_id

        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device="cuda"))

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

        k = einsum("batch key_pos n_heads d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre_k, self.W_K) + self.b_K

        if self.cfg.positional_embedding_type == "rotary":
            q = self.apply_rotary(q)
            k = self.apply_rotary(k)

        attn_scores = einsum("batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attn_scores = attn_scores / math.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        pattern = attn_scores.softmax(dim=-1) # [batch, n_head, query_pos, key_pos]
        v = einsum("batch key_pos n_heads d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre_v, self.W_V) + self.b_V

        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head", pattern, v)

        attn_out = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos n_heads d_model", z, self.W_O) + (self.b_O / self.cfg.n_heads)
        return attn_out

    def apply_causal_mask(self, attn_scores):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

class TransformerBlock(nn.Module):
    def __init__(self, cfg, block_index):
        super().__init__()
        self.cfg = cfg

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

        self.attn = Attention(cfg, "global", block_index)

        if not self.cfg.attn_only:
            if self.cfg.num_experts:
                self.mlp = MoE(cfg)
            elif self.cfg.gated_mlp:
                self.mlp = GatedMLP(cfg)
            else:
                self.mlp = MLP(cfg)

        for p in self.parameters():
            p.requires_grad = False

        prev_nodes = (cfg.n_heads + 1) * block_index + 1
        self.edge_mask_attention_q_logits = torch.nn.Parameter(
            torch.nn.init.normal_(torch.ones((prev_nodes, cfg.n_heads)), mean=self.cfg.logits_e_init, std=0.01), 
            requires_grad=True)
        self.edge_mask_attention_k_logits = torch.nn.Parameter(
            torch.nn.init.normal_(torch.ones((prev_nodes, cfg.n_heads)), mean=self.cfg.logits_e_init, std=0.01), 
            requires_grad=True)
        self.edge_mask_attention_v_logits = torch.nn.Parameter(
            torch.nn.init.normal_(torch.ones((prev_nodes, cfg.n_heads)), mean=self.cfg.logits_e_init, std=0.01), 
            requires_grad=True)
        self.edge_mask_mlp_logits = torch.nn.Parameter(
            torch.nn.init.normal_(torch.ones((prev_nodes + cfg.n_heads, )), mean=self.cfg.logits_e_init, std=0.01), 
            requires_grad=True)

    def forward(self, resid_pre):

        if self.cfg.use_edge_masks:
            if self.cfg.use_deterministic_masks:
                sampled_edge_mask_attentions_q = torch.where(self.edge_mask_attention_q_logits > 0., 1., 0.).to(self.cfg.device, dtype=self.cfg.dtype)
                sampled_edge_mask_attentions_k = torch.where(self.edge_mask_attention_k_logits > 0., 1., 0.).to(self.cfg.device, dtype=self.cfg.dtype)
                sampled_edge_mask_attentions_v = torch.where(self.edge_mask_attention_v_logits > 0., 1., 0.).to(self.cfg.device, dtype=self.cfg.dtype)
                sampled_edge_mask_mlp = torch.where(self.edge_mask_mlp_logits > 0., 1., 0.).to(self.cfg.device, dtype=self.cfg.dtype)
            else:
                sampled_edge_mask_attentions_q = gumbel_sigmoid(self.edge_mask_attention_q_logits, self.cfg.gs_temp_edge)
                sampled_edge_mask_attentions_k = gumbel_sigmoid(self.edge_mask_attention_k_logits, self.cfg.gs_temp_edge)
                sampled_edge_mask_attentions_v = gumbel_sigmoid(self.edge_mask_attention_v_logits, self.cfg.gs_temp_edge)
                sampled_edge_mask_mlp = gumbel_sigmoid(self.edge_mask_mlp_logits, self.cfg.gs_temp_edge)
            if self.cfg.use_reverse_masks:
                sampled_edge_mask_attentions_q = 1. - sampled_edge_mask_attentions_q
                sampled_edge_mask_attentions_k = 1. - sampled_edge_mask_attentions_k
                sampled_edge_mask_attentions_v = 1. - sampled_edge_mask_attentions_v
                sampled_edge_mask_mlp = 1. - sampled_edge_mask_mlp
        else:
            sampled_edge_mask_attentions_q = torch.ones(self.edge_mask_attention_q_logits.shape).to(resid_pre.device, dtype=self.cfg.dtype)
            sampled_edge_mask_attentions_k = torch.ones(self.edge_mask_attention_k_logits.shape).to(resid_pre.device, dtype=self.cfg.dtype)
            sampled_edge_mask_attentions_v = torch.ones(self.edge_mask_attention_v_logits.shape).to(resid_pre.device, dtype=self.cfg.dtype)

            sampled_edge_mask_mlp = torch.ones(self.edge_mask_mlp_logits.shape).to(resid_pre.device, dtype=self.cfg.dtype)

        # print(f'sampled_edge_mask_mlp: {sampled_edge_mask_mlp}')
        # print(f'self.edge_mask_attention_v_logits: {self.edge_mask_attention_v_logits.sum()}')

        # resid_pre [batch, position, d_model, prev_head_idx]
        masked_residuals_q = einsum("batch position prev_head_idx d_model, prev_head_idx n_heads -> batch position n_heads d_model", resid_pre, sampled_edge_mask_attentions_q)
        masked_residuals_k = einsum("batch position prev_head_idx d_model, prev_head_idx n_heads -> batch position n_heads d_model", resid_pre, sampled_edge_mask_attentions_k)
        masked_residuals_v = einsum("batch position prev_head_idx d_model, prev_head_idx n_heads -> batch position n_heads d_model", resid_pre, sampled_edge_mask_attentions_v)

        normalized_resid_pre_q = self.ln1(masked_residuals_q)
        normalized_resid_pre_k = self.ln1(masked_residuals_k)
        normalized_resid_pre_v = self.ln1(masked_residuals_v)

        attn_out = self.attn(normalized_resid_pre_q, normalized_resid_pre_k, normalized_resid_pre_v)

        residual = torch.cat((resid_pre, attn_out), dim=2)

        # print(torch.where(self.edge_mask_mlp_logits > 0., 1., 0.))

        masked_mlp_residual = einsum("batch position prev_head_idx d_model, prev_head_idx -> batch position d_model", residual, sampled_edge_mask_mlp)
        
        normalized_resid_mid = self.ln2(masked_mlp_residual)
        mlp_out = self.mlp(normalized_resid_mid)
        mlp_out = einops.rearrange(mlp_out, "batch position d_model -> batch position 1 d_model")

        residual = torch.cat((residual, mlp_out), dim=2)

        return residual

class CircuitTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.embed = Embed(self.cfg)

        if self.cfg.positional_embedding_type != "rotary":
            self.pos_embed = PosEmbed(self.cfg)

        self.blocks = nn.ModuleList(
            [TransformerBlock(self.cfg, block_index) for block_index in range(self.cfg.n_layers)]
        )
        total_nodes = (cfg.n_heads + 1) * cfg.n_layers + 1
        self.edge_mask_output_logits = torch.nn.Parameter(
            torch.nn.init.normal_(torch.ones((total_nodes,)), mean=self.cfg.logits_e_init, std=0.01), 
            requires_grad=True)

        if self.cfg.normalization_type == "RMS":
            self.ln_final = RMSNorm(self.cfg)
        elif self.cfg.normalization_type == "RMSPre":
            self.ln_final = RMSNormPre(self.cfg)
        elif self.cfg.normalization_type == "LN":
            if self.cfg.final_rms:
                self.ln_final = RMSNorm(self.cfg)
            else:
                self.ln_final = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            if self.cfg.final_rms:
                self.ln_final = RMSNormPre(self.cfg)
            else:
                self.ln_final = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type is None:
            # If it's None, don't create either layer
            pass
        else:
            logging.warning("Invalid normalization_type passed in %s", self.cfg.normalization_type)
        self.unembed = Unembed(self.cfg)

        if self.cfg.init_weights:
            self.init_weights()

        # initialize mask logits
        # self.mask_logits_device = cfg.mask_logits_device
        self.gs_temp_weight = cfg.gs_temp_weight
        self.gs_temp_edge = cfg.gs_temp_edge
        self.unmasked_params = {}
        self.mask_logits_dict_weight = {}
        self.mask_logits_dict_edge = {}
        self.use_weight_masks = cfg.use_weight_masks
        self.use_edge_masks = cfg.use_edge_masks
        self.reference_weight_mask = {}
        self.overlap_weight_or_circuit = {}

        # weight mask logits initialization
        # only register weight mask logits if cfg.use_weight_masks == True to save memory
        # load pretrained mask logits if necessary
        if self.use_weight_masks:
            self.N_weight = 0
            for name, p in self.named_parameters():
                # do not learn masks for:
                # 1) embedding and unembedding layers
                # 2) layernorms
                if 'emb' not in name and 'edge' not in name and 'ln' not in name:  
                    p.grad = None
                    p.requires_grad = False
                    self.unmasked_params[name] = p.clone()

                    masks_logits = nn.Parameter(
                        torch.nn.init.normal_(torch.ones_like(p).to('cuda'), mean=self.cfg.logits_w_init, std=0.01),  
                        requires_grad=True
                    )    # we manually put mask_logits onto cuda here, since using nn.ParameterDict will incur an annoying re-naming issue             
                    self.mask_logits_dict_weight[name] = masks_logits
                    with torch.no_grad():
                        self.N_weight += torch.ones_like(p.view(-1)).sum().cpu()

        # edge mask logits initialization
        if self.use_edge_masks:
            self.N_edge = 0
            for name, p in self.named_parameters():
                if 'edge' in name:
                    self.mask_logits_dict_edge[name] = p
                    with torch.no_grad():
                        self.N_edge += torch.ones_like(p.view(-1)).sum().cpu()

    def forward(self, tokens, return_states=False):

        if self.cfg.positional_embedding_type == "standard":
            # tokens [batch, position]
            embed = self.embed(tokens)
            pos_embed = self.pos_embed(tokens)
            residual = embed + pos_embed
        elif self.cfg.positional_embedding_type == "rotary":
            residual = self.embed(tokens)
        else:
            raise ValueError(
                f"Invalid positional_embedding_type passed in {self.cfg.positional_embedding_type}"
            )

        residual = einops.rearrange(residual, "batch position d_model -> batch position 1 d_model")

        for i, block in enumerate(self.blocks):
            residual = block(residual)

        if return_states:
            return residual


        if self.cfg.use_edge_masks:
            if self.cfg.use_deterministic_masks:
                sampled_output_mask = torch.where(self.edge_mask_output_logits > 0., 1., 0.).to(self.cfg.device, dtype=self.cfg.dtype)
            else:
                sampled_output_mask = gumbel_sigmoid(self.edge_mask_output_logits, gs_temp=self.cfg.gs_temp_edge)
            if self.cfg.use_reverse_masks:
                sampled_output_mask = 1. - sampled_output_mask
        else:
            sampled_output_mask = torch.ones(self.edge_mask_output_logits.shape).to(self.cfg.device, dtype=self.cfg.dtype)


        residual = einsum("batch position prev_head_idx d_model, prev_head_idx -> batch position d_model", residual, sampled_output_mask)
        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        return [logits]


    def turn_on_edge_masks(self, deterministic=False, reverse=False):
        self.cfg.use_edge_masks = True
        self.cfg.use_deterministic_masks = deterministic
        self.cfg.use_reverse_masks = reverse
        for block in self.blocks:
            block.cfg.use_edge_masks = True
            block.cfg.use_deterministic_masks = deterministic
            block.cfg.use_reverse_masks = reverse


    def turn_off_edge_masks(self):
        self.cfg.use_edge_masks = False
        for block in self.blocks:
            block.cfg.use_edge_masks = False


    def turn_on_weight_masks(self, deterministic=False, reverse=False):
        if self.use_weight_masks:
            for name, param in self.named_parameters():
                if name in self.unmasked_params:            
                    unmasked_m = self.unmasked_params[name].to(param.device)
                    mask_logits = self.mask_logits_dict_weight[name]
                    if not deterministic:
                        sampled_masks = gumbel_sigmoid(mask_logits, gs_temp=self.gs_temp_weight)
                    else:
                        with torch.no_grad():
                            sampled_masks = torch.where(mask_logits > 0., 1., 0.).to(self.cfg.device, dtype=self.cfg.dtype)
                    if reverse:
                        sampled_masks = 1. - sampled_masks
                    param.copy_(sampled_masks * unmasked_m)            


    def turn_off_weight_masks(self):
        if self.use_weight_masks:
            for name, param in self.named_parameters():
                if name in self.unmasked_params:            
                    unmasked_m = self.unmasked_params[name]
                    param.copy_(unmasked_m)
                    param.detach_()

    def weight_experiment_loss(self):
        if not bool(self.reference_weight_mask):
            return 0
        experiment_loss = 0
        for key, mask_logits in self.mask_logits_dict_weight.items():
            reference_value = self.reference_weight_mask[key]
            condition = (reference_value == 1) & (mask_logits > 0)
            experiment_loss += 2 * mask_logits[condition].sum()
        return experiment_loss / self.N_weight

    def weight_sparseness_loss(self):
        sparse_loss = 0
        for _, mask_logits in self.mask_logits_dict_weight.items():
            sparse_loss += F.sigmoid(mask_logits).sum()
        return sparse_loss / self.N_weight


    def edge_sparseness_loss(self):
        sparse_loss = 0
        for n, mask_logits in self.mask_logits_dict_edge.items():
            # print(n)
            sparse_loss += F.sigmoid(mask_logits).sum()
        return sparse_loss / self.N_edge

    def overlap_reward_loss(self):
        overlap_reward_loss = 0
        if type(self.overlap_weight_or_circuit) == dict and 'edge_mask' in self.overlap_weight_or_circuit.keys():
            return overlap_reward_loss  # not implemented
        else:
            for key, mask_logits in self.mask_logits_dict_weight.items():
                overlap_value = self.overlap_weight_or_circuit[key]
                condition = (overlap_value == 1) & (mask_logits > 0)
                overlap_reward_loss -= mask_logits[condition].sum()
            return overlap_reward_loss


    def get_edge_masks(self):
        edge_mask_dict = {
            'attn_q': [],
            'attn_k': [],
            'attn_v': [],
            'mlp': [],
            'output': []
        }
        with torch.no_grad():
            edge_mask_dict['output'] = torch.where(self.output_mask_logits > 0., 1., 0.).cpu()
            for i in range(self.cfg.n_layers):
                block_i = self.blocks[i]
                edge_mask_attn_q_i = torch.where(block_i.edge_mask_attention_q_logits > 0., 1., 0.).cpu()
                edge_mask_attn_k_i = torch.where(block_i.edge_mask_attention_k_logits > 0., 1., 0.).cpu()
                edge_mask_attn_v_i = torch.where(block_i.edge_mask_attention_v_logits > 0., 1., 0.).cpu()
                edge_mask_mlps_i = torch.where(block_i.edge_mask_mlp_logits > 0., 1., 0.).cpu()
                edge_mask_dict['attn_q'].append(edge_mask_attn_q_i)
                edge_mask_dict['attn_k'].append(edge_mask_attn_k_i)
                edge_mask_dict['attn_v'].append(edge_mask_attn_v_i)
                edge_mask_dict['mlp'].append(edge_mask_mlps_i)

        return edge_mask_dict

    def get_weight_density(self):
        try:
            N_weight_preserved = 0
            with torch.no_grad():
                for _, mask in self.mask_logits_dict_weight.items():
                    N_weight_preserved += torch.where(mask > 0., 1, 0).sum()

            weight_den = N_weight_preserved / self.N_weight
            # print(f'N_weight_preserved: {N_weight_preserved}')
            # print(f'N_weight: {self.N_weight}')
            return self.N_weight.item(), N_weight_preserved.item(), weight_den.item()
        except Exception as e:
            return -1, -1, 1.0


    def get_edge_density(self):
        N_edge_preserved = 0
        with torch.no_grad():
            for _, mask in self.mask_logits_dict_edge.items():
                N_edge_preserved += torch.where(mask > 0., 1, 0).sum()

        edge_den = N_edge_preserved / self.N_edge
        return self.N_edge.item(), N_edge_preserved.item(), edge_den.item()


    def load_pretrained_weight_mask(self, mask_logits_dict_weight):
        for n, _ in mask_logits_dict_weight.items():
            masks_logits = nn.Parameter(mask_logits_dict_weight[n], requires_grad=True)
            self.mask_logits_dict_weight[n] = masks_logits


    def load_pretrained_edge_mask(self, mask_logits_dict_edge):
        self.load_state_dict(mask_logits_dict_edge, strict=False)

    @classmethod
    def from_pretrained(cls, cfg):
        model = cls(cfg)
        state_dict = HookedTransformer.from_pretrained(cfg.model_name, **{'force_download': True}).state_dict()
        model.load_state_dict(state_dict, strict=False)
        for n, p in state_dict.items():
            if n in model.unmasked_params:
                model.unmasked_params[n] = p.clone()
        del state_dict
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = model.to(model.cfg.device, dtype=model.cfg.dtype)

        model.tokenizer = tokenizer
        model.dls = setup_task(model)

        return model

    @torch.no_grad
    def evaluate(self, dl=None, reverse=False):
        if dl is None:
            dl = self.dls.eval

        self.eval()

        self.turn_on_weight_masks(deterministic=True, reverse=reverse)
        self.turn_on_edge_masks(deterministic=True, reverse=reverse)

        if self.cfg.use_weight_masks:
            _, _, weight_density = self.get_weight_density()
        else:
            weight_density = 'na'
        if self.cfg.use_edge_masks:
            _, _, edge_density = self.get_edge_density()
        else:
            edge_density = 'na'

        if not self.cfg.use_weight_masks:
            self.turn_off_weight_masks()
        if not self.cfg.use_edge_masks:
            self.turn_off_edge_masks()

        total = len(dl.dataset)
        correct = 0
        kls = []
        faith_losses = []

        for i, batch_inputs in enumerate(dl):  # tqdm(dl, desc='Evaluating')

            original_logits = dl.original_output[i]

            batch_logits_masked = self(batch_inputs['input_ids'].to(self.cfg.device))[0]
            eval_results = self.compute_loss(batch_logits_masked, batch_inputs, original_logits)

            correct += eval_results['n_correct']
            kls.append(eval_results['kl_div'].cpu())
            faith_losses.append(eval_results['faith'])

        self.turn_off_weight_masks()
        self.turn_off_edge_masks()

        acc = correct / total

        results = {
            'acc': acc,
            'kl': torch.stack(kls).mean().item(),
            'faith_loss': torch.stack(faith_losses).mean().item(),
            'weight_density': weight_density,
            'edge_density': edge_density,
            'n_correct': correct,
            'total': total
        }
        return results
    
    @torch.no_grad
    def prepare_origin_output(self, dl=None):
        self.turn_off_weight_masks()
        self.turn_off_edge_masks()

        self.eval()

        if dl is None:
            dl = self.dls.eval

        record = {}

        for i, batch_inputs in enumerate(dl):  # tqdm(dl, desc='Evaluating')
            batch_logits_orig = self(batch_inputs['input_ids'].to(self.cfg.device))[0]

            batch_seq_lens = batch_inputs['seq_lens']
            batch_size = batch_logits_orig.shape[0]
            logits_target_good_orig = batch_logits_orig[torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target good']]
            logits_target_bad_orig = batch_logits_orig[torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target bad']]
            logits_gb_orig = torch.stack([logits_target_good_orig, logits_target_bad_orig], -1)  # (B, 2)

            record[i] = logits_gb_orig.cpu()
        dl.original_output = record


    def compute_loss(self, batch_logits_masked, batch_inputs, original_logits=None):
        results = {}

        faith_results = compute_faith_loss(batch_logits_masked, batch_inputs, original_logits)
        results.update(faith_results)

        return faith_results

    def search_circuit(self, modes='we'):
        result = {}
        if 'w' in modes:
            result = self.run_prune(mode='w')

        gc.collect()
        torch.cuda.empty_cache()

        if 'e' in modes:
            result = self.run_prune(mode='e')
        return result

    def run_prune(self, mode):

        if mode == 'w':
            # weight pruning
            mask_logits_dict = self.mask_logits_dict_weight
            hparams = self.cfg.weight
        elif mode == 'e':
            mask_logits_dict = self.mask_logits_dict_edge
            hparams = self.cfg.edge

        mask_logits = [mask for _, mask in mask_logits_dict.items()]
        optimizer = torch.optim.AdamW(mask_logits, lr=hparams.lr)

        if mode == 'w':
            self.turn_off_edge_masks()
        elif mode == 'e':
            self.turn_on_weight_masks(deterministic=True)

        # time.sleep(0.01)
        epoch_loop = tqdm(range(hparams.train_epochs), desc='Number of Epochs', leave=True, dynamic_ncols=True)

        list_result = []

        for i, epoch in enumerate(epoch_loop):
            lambda_sparse = schedule_epoch_lambda(
                epoch,
                lambda_0=hparams.lambda_sparse_init,
                max_times=hparams.max_times_lambda_sparse,
                min_times=hparams.min_times_lambda_sparse,
                n_epoch_warmup=hparams.n_epoch_warmup_lambda_sparse,
                n_epoch_cooldown=hparams.n_epoch_cooldown_lambda_sparse,
            )
            lambda_complete = schedule_epoch_lambda(epoch, hparams.lambda_complete_init)

            for batch_inputs in self.dls.train:

                # weight pruning
                if mode == 'w':
                    self.turn_on_weight_masks(deterministic=False, reverse=False)
                    sparse_loss = self.weight_sparseness_loss()
                elif mode == 'e':
                    self.turn_on_edge_masks(deterministic=False, reverse=False)
                    sparse_loss = self.edge_sparseness_loss()

                batch_logits_masked = self(batch_inputs['input_ids'].to(self.cfg.device))[
                    0]  # (B, seq_len, vocab_size)
                eval_results = compute_faith_loss(batch_logits_masked, batch_inputs)
                faith_loss = eval_results['faith']

                if mode == 'e' and lambda_complete > 0:
                    self.turn_on_edge_masks(deterministic=False, reverse=True)
                    batch_logits = self(batch_inputs['input_ids'].to(self.cfg.device))[0]
                    complete_loss, _ = compute_complete_loss(batch_logits, batch_inputs)
                else:
                    complete_loss = 0.

                if mode == 'w':
                    loss = faith_loss + sparse_loss * lambda_sparse
                elif mode == 'e':
                    loss = faith_loss + sparse_loss * lambda_sparse + complete_loss * lambda_complete

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if mode == 'w':
                    self.turn_off_weight_masks()

                if mode == 'w' and lambda_complete > 0:
                    self.turn_on_weight_masks(deterministic=False, reverse=True)
                    batch_logits = self(batch_inputs['input_ids'].to(self.cfg.device))[0]
                    complete_loss, _ = compute_complete_loss(batch_logits, batch_inputs)
                    loss = complete_loss * lambda_complete
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    self.turn_off_weight_masks()

            if i % self.cfg.evaluate_every == self.cfg.evaluate_every - 1:
                comp = self.evaluate(reverse=True)
                results = self.evaluate()
                results['comp'] = comp['acc']
                list_result.append(results)
                pprint(results)

        weight_mask = self.mask_logits_dict_weight
        edge_mask = self.mask_logits_dict_edge

        # output_dir = Path(self.cfg.output_dir_path) / self.cfg.exp_name
        # output_dir.mkdir(parents=True, exist_ok=True)

        # if mode == 'w':
        #     torch.save(weight_mask, output_dir / f'weight_mask_{mode}.pt')
        # if mode == 'e':
        #     torch.save(edge_mask, output_dir / f'edge_mask_{mode}.pt')

        del mask_logits
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        return list_result
