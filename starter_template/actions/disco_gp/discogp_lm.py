"""
DiscoGPTransformer: maskable transformer for sheaf discovery / pruning.

This module wraps a (HookedTransformer-compatible) architecture with two kinds of
learnable masks:
- **Weight masks**: continuous logits per-parameter that are sampled via a
  straight-through Gumbel-sigmoid to 0/1 during forward passes.
- **Edge masks**: continuous logits per edge/node whose samples gate residual
  contributions across the computational graph (heads/MLP/output).

It provides:
- Utilities to turn masks on/off (optionally deterministically or reversed),
- Sparsity/overlap losses for regularization,
- Evaluation helpers that compare masked vs. original logits,
- A simple pruning loop that optimizes mask logits w.r.t. faithfulness and
  completeness objectives.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast, overload
from typing_extensions import Literal

import os
import gc
from pathlib import Path

from tqdm.auto import tqdm
from pprint import pprint
import wandb

import numpy as np
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from fancy_einsum import einsum

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens.components import (
    PosEmbed,
    RMSNorm,
    RMSNormPre,
    LayerNorm,
    LayerNormPre,
    Unembed,
    GatedMLP,
    MLP,
    MoE,
    Embed,
)
from transformer_lens import HookedTransformer

from .modules.transformer_block import DiscoGPTransformerBlock

from .data import setup_task
from .evaluation import (
    compute_complete_loss_binary_label,
    compute_faith_loss_binary_label,
    compute_complete_loss_multi_label,
    compute_faith_loss_multi_label,
)
from .utils import schedule_epoch_lambda
from .configs import Config
import optuna

def gumbel_sigmoid(logits, gs_temp: float = 1.0, eps: float = 1e-10):
    """Sample a Bernoulli-like gate using a straight-through Gumbel-sigmoid.

    - Adds Gumbel noise to `logits`, divides by temperature `gs_temp`, and applies
      `sigmoid`.
    - Uses a straight-through estimator (round at 0.5 but keep gradients).
    """
    # Draw two uniforms to create a (differenced) Gumbel noise sample
    uniform = logits.new_empty([2] + list(logits.shape)).uniform_(0, 1)
    noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()

    # Relaxed sample
    res = torch.sigmoid((logits + noise) / gs_temp)

    # Straight-through: hard threshold in forward, identity in backward
    res = ((res > 0.5).type_as(res) - res).detach() + res
    return res


class DiscoGPTransformer(nn.Module):
    """Transformer with learnable weight and edge masks for sheaf (circuit) discovery."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Token + positional embeddings (if not rotary)
        self.embed = Embed(self.cfg)
        if self.cfg.positional_embedding_type != "rotary":
            self.pos_embed = PosEmbed(self.cfg)

        # Stack of custom blocks that apply/propagate masks internally
        self.blocks = nn.ModuleList(
            [DiscoGPTransformerBlock(self.cfg, block_index) for block_index in range(self.cfg.n_layers)]
        )

        # Output edge mask logits cover all nodes (heads + mlp per layer + final output)
        total_nodes = (cfg.n_heads + 1) * cfg.n_layers + 1
        self.edge_mask_output_logits = torch.nn.Parameter(
            torch.nn.init.normal_(torch.ones((total_nodes,)), mean=self.cfg.logits_e_init, std=0.01),
            requires_grad=True,
        )

        # Final normalization choice
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

    def setup_mask(self):
        """Initialize mask-related state and (optionally) mask logits parameters.

        Weight masks are registered only if `cfg.use_weight_masks` to save memory.
        Edge masks are taken from existing parameters with 'edge' in their name.
        """
        # Temperatures for Gumbel-sigmoid sampling
        self.gs_temp_weight = self.cfg.gs_temp_weight
        self.gs_temp_edge = self.cfg.gs_temp_edge

        # Containers for mask bookkeeping
        self.unmasked_params = {}
        self.mask_logits_dict_weight = {}
        self.mask_logits_dict_edge = {}
        self.use_weight_masks = self.cfg.use_weight_masks
        self.use_edge_masks = self.cfg.use_edge_masks
        self.reference_weight_mask = {}
        self.overlap_weight_or_circuit = {}

        # ---- Weight mask logits initialization ----
        # Register mask logits per-parameter (excluding emb/ln/edge) and freeze real weights.
        if self.use_weight_masks:
            self.N_weight = 0
            for name, p in self.named_parameters():
                # Skip embeddings/unembeddings, LayerNorms, and any edge-mask parameters
                if 'emb' not in name and 'edge' not in name and 'ln' not in name:
                    p.grad = None
                    p.requires_grad = False
                    self.unmasked_params[name] = p.clone()

                    masks_logits = nn.Parameter(
                        torch.nn.init.normal_(torch.ones_like(p).to(self.cfg.device), mean=self.cfg.logits_w_init, std=0.01),
                        requires_grad=True,
                    )
                    # Note: using a plain dict avoids `nn.ParameterDict` name mangling here.
                    self.mask_logits_dict_weight[name] = masks_logits
                    with torch.no_grad():
                        self.N_weight += torch.ones_like(p.view(-1)).sum().cpu()

        # ---- Edge mask logits initialization ----
        # Collect parameters that are already defined as edge-mask logits.
        if self.use_edge_masks:
            self.N_edge = 0
            for name, p in self.named_parameters():
                if 'edge' in name:
                    self.mask_logits_dict_edge[name] = p
                    with torch.no_grad():
                        self.N_edge += torch.ones_like(p.view(-1)).sum().cpu()

    def forward(self, tokens, return_states: bool = False):
        """Forward pass with optional state return (pre-unembed).

        Input
        ---
        tokens: LongTensor [batch, position]
        return_states: if True, return the residual stream per-node before final ln+unembed.
        """
        if self.cfg.positional_embedding_type == "standard":
            # Standard absolute position embeddings
            embed = self.embed(tokens)
            pos_embed = self.pos_embed(tokens)
            residual = embed + pos_embed
        elif self.cfg.positional_embedding_type == "rotary":
            # Rotary embeddings applied inside attention layers
            residual = self.embed(tokens)
        else:
            raise ValueError(
                f"Invalid positional_embedding_type passed in {self.cfg.positional_embedding_type}"
            )

        # Add explicit prev_head_idx dimension for gated aggregation across nodes
        residual = einops.rearrange(residual, "batch position d_model -> batch position 1 d_model")

        # Pass through transformer blocks; each block may expand the prev_head_idx dim
        for i, block in enumerate(self.blocks):
            residual = block(residual)

        if return_states:
            return residual

        # Sample or threshold the *output* edge mask that combines node streams
        if self.cfg.use_edge_masks:
            if self.cfg.use_deterministic_masks:
                sampled_output_mask = torch.where(self.edge_mask_output_logits > 0., 1., 0.).to(self.cfg.device, dtype=self.cfg.dtype)
            else:
                sampled_output_mask = gumbel_sigmoid(self.edge_mask_output_logits, gs_temp=self.cfg.gs_temp_edge)
            if self.cfg.use_reverse_masks:
                sampled_output_mask = 1. - sampled_output_mask
        else:
            sampled_output_mask = torch.ones(self.edge_mask_output_logits.shape).to(self.cfg.device, dtype=self.cfg.dtype)

        # Collapse the prev_head_idx dimension via a weighted sum by the sampled mask
        residual = einsum(
            "batch position prev_head_idx d_model, prev_head_idx -> batch position d_model",
            residual,
            sampled_output_mask,
        )

        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        return [logits]

    # --- Runtime mask toggles -------------------------------------------------

    def turn_on_edge_masks(self, deterministic: bool = False, reverse: bool = False):
        """Enable edge masks for this model and all blocks.

        - `deterministic=True` means use (logits>0) thresholding instead of sampling.
        - `reverse=True` flips 0/1 decisions to evaluate completeness.
        """
        self.cfg.use_edge_masks = True
        self.cfg.use_deterministic_masks = deterministic
        self.cfg.use_reverse_masks = reverse
        for block in self.blocks:
            block.cfg.use_edge_masks = True
            block.cfg.use_deterministic_masks = deterministic
            block.cfg.use_reverse_masks = reverse

    def turn_off_edge_masks(self):
        """Disable edge masks for this model and all blocks."""
        self.cfg.use_edge_masks = False
        for block in self.blocks:
            block.cfg.use_edge_masks = False

    def turn_on_weight_masks(self, deterministic: bool = False, reverse: bool = False):
        """Materialize masked weights in-place from logits using sampling/thresholding.

        When enabled, parameters in `self.unmasked_params` are overwritten by
        sampled masks multiplied elementwise with the stored unmasked weights.
        """
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
        """Restore original (unmasked) weights in-place and detach."""
        if self.use_weight_masks:
            for name, param in self.named_parameters():
                if name in self.unmasked_params:
                    unmasked_m = self.unmasked_params[name]
                    param.copy_(unmasked_m)
                    param.detach_()

    # --- Mask-related losses --------------------------------------------------

    def weight_experiment_loss(self):
        """Optional loss to penalize disagreements with a reference weight mask.

        Adds positive mass where current mask_logits are >0 but reference is 1.
        Returns 0 if no reference is set.
        """
        if not bool(self.reference_weight_mask):
            return 0
        experiment_loss = 0
        for key, mask_logits in self.mask_logits_dict_weight.items():
            reference_value = self.reference_weight_mask[key]
            condition = (reference_value == 1) & (mask_logits > 0)
            experiment_loss += 2 * mask_logits[condition].sum()
        return experiment_loss / self.N_weight

    def weight_sparseness_loss(self):
        """L1-like sparsity on weight mask probabilities (via sigmoid)."""
        sparse_loss = 0
        for _, mask_logits in self.mask_logits_dict_weight.items():
            sparse_loss += F.sigmoid(mask_logits).sum()
        return sparse_loss / self.N_weight

    def edge_sparseness_loss(self):
        """L1-like sparsity on edge mask probabilities (via sigmoid)."""
        sparse_loss = 0
        for n, mask_logits in self.mask_logits_dict_edge.items():
            sparse_loss += F.sigmoid(mask_logits).sum()
        return sparse_loss / self.N_edge

    # --- Introspection helpers ------------------------------------------------

    def get_edge_masks(self):
        """Return thresholded (0/1) edge masks for attention Q/K/V, MLP, and output."""
        edge_mask_dict = {
            'attn_q': [],
            'attn_k': [],
            'attn_v': [],
            'mlp': [],
            'output': [],
        }
        with torch.no_grad():
            edge_mask_dict['output'] = torch.where(self.edge_mask_output_logits > 0., 1., 0.).cpu()
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
        """Return (#weights, #preserved, density) for thresholded (>0) weight masks."""
        try:
            N_weight_preserved = 0
            with torch.no_grad():
                for _, mask in self.mask_logits_dict_weight.items():
                    N_weight_preserved += torch.where(mask > 0., 1, 0).sum()

            weight_den = N_weight_preserved / self.N_weight
            return self.N_weight.item(), N_weight_preserved.item(), weight_den.item()
        except Exception as e:
            # If masks are uninitialized
            return -1, -1, 1.0

    def get_edge_density(self):
        """Return (#edges, #preserved, density) for thresholded (>0) edge masks."""
        N_edge_preserved = 0
        with torch.no_grad():
            for _, mask in self.mask_logits_dict_edge.items():
                N_edge_preserved += torch.where(mask > 0., 1, 0).sum()

        edge_den = N_edge_preserved / self.N_edge
        return self.N_edge.item(), N_edge_preserved.item(), edge_den.item()

    # --- Loading/stubs --------------------------------------------------------

    def load_pretrained_weight_mask(self, mask_logits_dict_weight):
        """Load externally trained weight mask logits into this model."""
        for n, _ in mask_logits_dict_weight.items():
            masks_logits = nn.Parameter(mask_logits_dict_weight[n], requires_grad=True)
            self.mask_logits_dict_weight[n] = masks_logits

    def load_pretrained_edge_mask(self, mask_logits_dict_edge):
        """Load externally trained edge mask logits via state_dict (non-strict)."""
        self.load_state_dict(mask_logits_dict_edge, strict=False)

    @classmethod
    def from_pretrained(cls, cfg):
        """Instantiate, load base weights from HookedTransformer, and set up masks.

        - Copies the state_dict from a pretrained `HookedTransformer`.
        - Calls `hook_state_dict` first to adapt K/V for GQA before loading.
        - Prepares tokenizer and data loaders via `setup_task`.
        """
        model = cls(cfg)
        print("cfg name:", cfg.full_model_name)
        state_dict = HookedTransformer.from_pretrained(cfg.full_model_name).state_dict()
        model.hook_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model.setup_mask()

        # Keep an untouched copy of original weights for in-place masking
        for n, p in state_dict.items():
            if n in model.unmasked_params:
                model.unmasked_params[n] = p.clone()
        del state_dict
        torch.cuda.empty_cache()

        # Tokenizer setup (ensure padding token exists)
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = model.to(model.cfg.device, dtype=model.cfg.dtype)

        model.tokenizer = tokenizer
        model.dls = setup_task(model)

        return model
    
    def setup_experiment(self):
        """Prepare cached original outputs for all splits and optionally start a Weights & Biases run.

        Caching original (unmasked) logits is required so later evaluations can compare
        masked outputs against the unmodified model outputs (KL/faithfulness metrics).
        """
        # Precompute and store original logits for train/eval/test dataloaders.
        self.prepare_origin_output(self.dls.train)
        self.prepare_origin_output(self.dls.eval)
        self.prepare_origin_output(self.dls.test)

        # Initialize wandb if requested in the config to track experiment metrics.
        if self.cfg.get('use_wandb', False):
            run = wandb.init(
                project=self.cfg.wandb_project_name,
                entity=self.cfg.wandb_entity,
                config=self.cfg.to_dict(collapse=True),
            )
            self.wandb_run = run

    def teardown_experiment(self):
        """Cleanly finish any external experiment tracking (e.g., wandb)."""
        if self.cfg.get('use_wandb', False):
            # Ensure the wandb run is finalized to flush logs and release resources.
            self.wandb_run.finish()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def log_result(self, result):
        """Log a single dictionary of results to the configured tracking backend.

        Currently supports wandb when enabled in the configuration.
        """
        if self.cfg.get('use_wandb', False):
            # Forward the result dict to wandb for visualization/monitoring.
            self.wandb_run.log(result)

        if self.cfg.get('print_results', True):
            # Print to console for immediate feedback.
            pprint(result)

    def hook_state_dict(self, state_dict):
        """Preprocess state_dict before loading (e.g., expand K/V for GQA models).

        If `n_key_value_heads` < `n_heads`, repeat K/V heads to match Q head count
        so that edge masks align with the actual compute graph.
        """
        if self.cfg.n_key_value_heads is not None:
            repeat_kv_heads = self.cfg.n_heads // self.cfg.n_key_value_heads

            for layer in range(self.cfg.n_layers):
                prefix = f"blocks.{layer}.attn."
                state_dict[prefix + "_W_K"] = torch.repeat_interleave(state_dict[prefix + "_W_K"], dim=0, repeats=repeat_kv_heads)
                state_dict[prefix + "_b_K"] = torch.repeat_interleave(state_dict[prefix + "_b_K"], dim=0, repeats=repeat_kv_heads)
                state_dict[prefix + "_W_V"] = torch.repeat_interleave(state_dict[prefix + "_W_V"], dim=0, repeats=repeat_kv_heads)
                state_dict[prefix + "_b_V"] = torch.repeat_interleave(state_dict[prefix + "_b_V"], dim=0, repeats=repeat_kv_heads)

    # --- Evaluation / bookkeeping --------------------------------------------

    @torch.no_grad
    def evaluate(self, dl=None, reverse: bool = False):
        """Evaluate accuracy/KL/faith on a given dataloader with deterministic masks.

        - If `reverse=True`, measures *completeness* by flipping masks.
        - Requires `prepare_origin_output` to have been called to compare KL/faith.
        """
        if dl is None:
            dl = self.dls.eval

        self.eval()

        # Use deterministic masks for evaluation
        self.turn_on_weight_masks(deterministic=True, reverse=reverse)
        self.turn_on_edge_masks(deterministic=True, reverse=reverse)

        # Densities for reporting
        if self.cfg.use_weight_masks:
            _, _, weight_density = self.get_weight_density()
        else:
            weight_density = 'na'
        if self.cfg.use_edge_masks:
            _, _, edge_density = self.get_edge_density()
        else:
            edge_density = 'na'

        # Respect config toggles
        if not self.cfg.use_weight_masks:
            self.turn_off_weight_masks()
        if not self.cfg.use_edge_masks:
            self.turn_off_edge_masks()

        total = len(dl.dataset)
        correct = 0
        kls = []
        faith_losses = []

        for i, batch_inputs in enumerate(dl):
            # Original (unmasked) logits must be precomputed and cached on the loader
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
            'total': total,
        }
        return results

    @torch.no_grad
    def prepare_origin_output(self, dl=None):
        """Cache original (unmasked) logits per batch index onto the dataloader.

        This must be called prior to `evaluate` so that KL/faith can compare
        masked vs. original outputs fairly.
        """
        self.turn_off_weight_masks()
        self.turn_off_edge_masks()

        self.eval()

        if dl is None:
            dl = self.dls.eval

        record = {}

        for i, batch_inputs in enumerate(dl):
            batch_logits_orig = self(batch_inputs['input_ids'].to(self.cfg.device))[0]

            if self.cfg.task_type in ['ioi', 'blimp']:
                batch_seq_lens = batch_inputs['seq_lens']
                batch_size = batch_logits_orig.shape[0]
                logits_target_good_orig = batch_logits_orig[torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target good']]
                logits_target_bad_orig = batch_logits_orig[torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target bad']]
                logits_gb_orig = torch.stack([logits_target_good_orig, logits_target_bad_orig], -1)  # (B, 2)
                record[i] = logits_gb_orig.cpu()

            elif self.cfg.task_type in ['pararel']:
                batch_seq_lens = batch_inputs['seq_lens']
                batch_size = batch_logits_orig.shape[0]
                full_logit = batch_logits_orig[torch.arange(batch_size), batch_seq_lens - 1]  # (B, answer_idx_vocab_size)
                answer_token_logits = full_logit[:, self.answer_idx_vocab]  # (B, answer_idx_vocab_size)
                record[i] = answer_token_logits.cpu()
            else:
                raise NotImplementedError(f"Original output not implemented for task type {self.cfg.task_type}")

        # Attach to the DataLoader instance (okay in Python, albeit a bit unconventional)
        dl.original_output = record

    # --- Loss dispatchers -----------------------------------------------------

    def compute_faith_loss(self, batch_logits_masked, batch_inputs, original_logits=None):
        """Compute faithfulness loss for current task type.

        For PARArel, restrict logits to the known answer vocab.
        """
        if self.cfg.task_type in ['ioi', 'blimp']:
            return compute_faith_loss_binary_label(batch_logits_masked, batch_inputs, original_logits)
        elif self.cfg.task_type in ['pararel']:
            batch_logits_masked = batch_logits_masked[:, :, self.answer_idx_vocab]
            return compute_faith_loss_multi_label(batch_logits_masked, batch_inputs, original_logits)
        else:
            raise NotImplementedError(f"Faith loss not implemented for task type {self.cfg.task_type}")

    def compute_complete_loss(self, batch_logits_masked, batch_inputs):
        """Compute completeness loss for the current task type."""
        if self.cfg.task_type in ['ioi', 'blimp']:
            return compute_complete_loss_binary_label(batch_logits_masked, batch_inputs)
        elif self.cfg.task_type in ['pararel']:
            batch_logits_masked = batch_logits_masked[:, :, self.answer_idx_vocab]
            return compute_complete_loss_multi_label(batch_logits_masked, batch_inputs)
        else:
            raise NotImplementedError(f"Complete loss not implemented for task type {self.cfg.task_type}")

    def compute_loss(self, batch_logits_masked, batch_inputs, original_logits=None):
        """Wrapper used in `evaluate`; currently returns just faithfulness results."""
        results = {}
        faith_results = self.compute_faith_loss(batch_logits_masked, batch_inputs, original_logits)
        results.update(faith_results)
        return results

    # --- Pruning loop ---------------------------------------------------------

    def search(self, modes='we', trial = None):
        """Run pruning for weights ('w'), edges ('e'), or both (default 'we')."""
        result = []
        if 'w' in modes:
            result = result + self.run_prune(mode='w', trial = trial)

        gc.collect()
        torch.cuda.empty_cache()

        if 'e' in modes:
            result = result + self.run_prune(mode='e', trial = trial)
        return result

    def evaluate_and_report(self, epoch=None, mode=None, meta={}, trial = None):
        """Evaluate on train/eval/test splits and pretty-print a summary."""
        full_results = {}
        full_results.update(meta)
        return_result = {}
        return_result.update(meta)

        for split_name, dl in {'train': self.dls.train, 'eval': self.dls.eval, 'test': self.dls.test}.items():
            comp = self.evaluate(dl=dl, reverse=True)
            results = self.evaluate(dl=dl)
            results['comp'] = comp['acc']
            results['prune_mode'] = mode
            results['epoch'] = epoch
            full_results[split_name] = results
            general_keys = ['edge_density', 'weight_density', 'epoch', 'prune_mode']
            for key in general_keys:
                return_result[key] = results[key]
            for key, value in results.items():
                if key in general_keys:
                    continue
                return_result[f'{split_name}_{key}'] = value
            if (split_name == 'train') and (trial is not None):
                if mode == 'w':
                    trial.report(results['acc'], step=epoch)
                else:
                    trial.report(results['acc'], step=epoch + self.cfg.weight.train_epochs)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        self.log_result(full_results)
        return return_result

    def run_prune(self, mode, trial = None):
        """Optimize mask logits for sparsity + (faithfulness [+ completeness]).

        - For weights ('w'): minimize faith + λ_sparse * sparsity.
        - For edges   ('e'): additionally include completeness via reversed masks.
        """
        if mode == 'w':
            # weight pruning
            mask_logits_dict = self.mask_logits_dict_weight
            hparams = self.cfg.weight
        elif mode == 'e':
            # edge pruning
            mask_logits_dict = self.mask_logits_dict_edge
            hparams = self.cfg.edge

        mask_logits = [mask for _, mask in mask_logits_dict.items()]
        optimizer = torch.optim.AdamW(mask_logits, lr=hparams.lr)

        # Ensure we don't double-mask: disable edges when pruning weights, but keep weights fixed when pruning edges
        if mode == 'w':
            self.turn_off_edge_masks()
        elif mode == 'e':
            self.turn_on_weight_masks(deterministic=True)

        epoch_loop = tqdm(
            range(hparams.train_epochs),
            desc='Number of Epochs', leave=True, dynamic_ncols=True,
            disable=self.cfg.get('disable_tqdm', False))

        list_results = []
        for i, epoch in enumerate(epoch_loop):
            # Lambda scheduling (warmup/cooldown, etc.)
            lambda_sparse = schedule_epoch_lambda(
                epoch,
                lambda_0=hparams.lambda_sparse_init,
                max_times=hparams.max_times_lambda_sparse,
                min_times=hparams.min_times_lambda_sparse,
                n_epoch_warmup=hparams.n_epoch_warmup_lambda_sparse,
                n_epoch_cooldown=hparams.n_epoch_cooldown_lambda_sparse,
            )
            lambda_complete = schedule_epoch_lambda(epoch, hparams.lambda_complete_init)

            epoch_loop.set_description(f"Epoch {epoch} {mode} λ_s={lambda_sparse:.3f} λ_c={lambda_complete:.3f}")

            for batch_inputs in self.dls.train:
                batch_input_ids = batch_inputs['input_ids'].to(self.cfg.device)

                # Sample current masks and compute sparsity loss
                if mode == 'w':
                    self.turn_on_weight_masks(deterministic=False, reverse=False)
                    sparse_loss = self.weight_sparseness_loss()
                elif mode == 'e':
                    self.turn_on_edge_masks(deterministic=False, reverse=False)
                    sparse_loss = self.edge_sparseness_loss()

                # Faithfulness on masked model
                batch_logits_masked = self(batch_input_ids)[0]  # (B, seq_len, vocab_size)
                eval_results = self.compute_faith_loss(batch_logits_masked, batch_inputs)
                faith_loss = eval_results['faith']

                # Completeness (evaluate with reversed masks) — only for edge mode during this step
                if mode == 'e' and lambda_complete > 0:
                    self.turn_on_edge_masks(deterministic=False, reverse=True)
                    batch_logits = self(batch_input_ids)[0]
                    complete_loss, _ = self.compute_complete_loss(batch_logits, batch_inputs)
                else:
                    complete_loss = 0.0

                if mode == 'w':
                    loss = faith_loss + sparse_loss * lambda_sparse
                elif mode == 'e':
                    loss = faith_loss + sparse_loss * lambda_sparse + complete_loss * lambda_complete

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if mode == 'w':
                    self.turn_off_weight_masks()

                # For weights, compute completeness in a second pass with reversed masks
                if mode == 'w' and lambda_complete > 0:
                    self.turn_on_weight_masks(deterministic=False, reverse=True)
                    batch_logits = self(batch_input_ids)[0]
                    complete_loss, _ = self.compute_complete_loss(batch_logits, batch_inputs)
                    loss = complete_loss * lambda_complete
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    self.turn_off_weight_masks()

            # Periodic evaluation
            if i % self.cfg.evaluate_every == self.cfg.evaluate_every - 1:
                result = self.evaluate_and_report(
                    epoch=epoch, mode=mode,
                    meta={
                        'lambda_sparse': lambda_sparse,
                        'lambda_complete': lambda_complete
                    }, trial=trial)
                list_results.append(result)

            weight_mask = self.mask_logits_dict_weight
            edge_mask = self.mask_logits_dict_edge

            if self.cfg.has('save_every', 'output_dir_path') and self.cfg.save_every and i % self.cfg.save_every == self.cfg.save_every - 1:
                output_dir = Path(self.cfg.output_dir_path) / self.cfg.exp_name
                output_dir.mkdir(parents=True, exist_ok=True)

                if mode == 'w':
                    torch.save(weight_mask, output_dir / f'weight_mask_{mode}_epoch{epoch}.pt')
                if mode == 'e':
                    torch.save(edge_mask, output_dir / f'edge_mask_{mode}_epoch{epoch}.pt')

        del mask_logits
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        return list_results
