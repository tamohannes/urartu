from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast, overload
from typing_extensions import Literal

import random

import logging
from tqdm.auto import tqdm
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from fancy_einsum import einsum

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformer_lens import HookedTransformer

from .data import setup_task
from .evaluation import compute_complete_loss, compute_faith_loss
from .utils import schedule_epoch_lambda

from .circuit_lm import (
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


class TransformerBlockIntMask(nn.Module):
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

        self.block_index = block_index

        prev_nodes = (self.cfg.n_heads + 1) * self.block_index + 1

        self.edge_mask_attention_q_mask = torch.ones(
            (prev_nodes, self.cfg.n_heads), device=self.cfg.device)
        self.edge_mask_attention_k_mask = torch.ones(
            (prev_nodes, self.cfg.n_heads), device=self.cfg.device)
        self.edge_mask_attention_v_mask = torch.ones(
            (prev_nodes, self.cfg.n_heads), device=self.cfg.device)
        self.edge_mask_mlp_mask = torch.ones(
            (prev_nodes + self.cfg.n_heads,), device=self.cfg.device)

        self.edge_mask_attention_q_index = {'input': 0}
        self.edge_mask_attention_k_index = {'input': 0}
        self.edge_mask_attention_v_index = {'input': 0}
        self.edge_mask_mlp_index = {'input': 0}

        # For attention masks (qkv), the second dim is always going to be attn head
        for i in range(self.block_index):
            for j in range(self.cfg.n_heads):
                self.edge_mask_attention_q_index[f'{i}.attn_{j}'] = i * (self.cfg.n_heads + 1) + j + 1
                self.edge_mask_attention_k_index[f'{i}.attn_{j}'] = i * (self.cfg.n_heads + 1) + j + 1
                self.edge_mask_attention_v_index[f'{i}.attn_{j}'] = i * (self.cfg.n_heads + 1) + j + 1
                self.edge_mask_mlp_index[f'{i}.attn_{j}'] = i * (self.cfg.n_heads + 1) + j + 1
            self.edge_mask_attention_q_index[f'{i}.mlp'] = i * (self.cfg.n_heads + 1) + j + 2
            self.edge_mask_attention_k_index[f'{i}.mlp'] = i * (self.cfg.n_heads + 1) + j + 2
            self.edge_mask_attention_v_index[f'{i}.mlp'] = i * (self.cfg.n_heads + 1) + j + 2
            self.edge_mask_mlp_index[f'{i}.mlp'] = i * (self.cfg.n_heads + 1) + j + 2

        for j in range(self.cfg.n_heads):
            self.edge_mask_mlp_index[f'{self.block_index}.attn_{j}'] = prev_nodes + j

    def forward(self, resid_pre):

        if self.cfg.use_edge_masks:
            edge_mask_q = self.edge_mask_attention_q_mask
            edge_mask_k = self.edge_mask_attention_k_mask
            edge_mask_v = self.edge_mask_attention_v_mask

            masked_residuals_q = einsum("batch position prev_head_idx d_model, prev_head_idx n_heads -> batch position n_heads d_model", resid_pre, edge_mask_q)
            masked_residuals_k = einsum("batch position prev_head_idx d_model, prev_head_idx n_heads -> batch position n_heads d_model", resid_pre, edge_mask_k)
            masked_residuals_v = einsum("batch position prev_head_idx d_model, prev_head_idx n_heads -> batch position n_heads d_model", resid_pre, edge_mask_v)

            if self.cfg.reversed:
                masked_residuals_q = 1 - masked_residuals_q
                masked_residuals_k = 1 - masked_residuals_k
                masked_residuals_v = 1 - masked_residuals_v

        else:
            masked_residuals_q = resid_pre
            masked_residuals_k = resid_pre
            masked_residuals_v = resid_pre

        normalized_resid_pre_q = self.ln1(masked_residuals_q)
        normalized_resid_pre_k = self.ln1(masked_residuals_k)
        normalized_resid_pre_v = self.ln1(masked_residuals_v)


        attn_out = self.attn(normalized_resid_pre_q, normalized_resid_pre_k, normalized_resid_pre_v)

        residual = torch.cat((resid_pre, attn_out), dim=2)

        # print(torch.where(self.edge_mask_mlp_logits > 0., 1., 0.))

        if self.cfg.use_edge_masks:

            sampled_edge_mask_mlp = self.edge_mask_mlp_mask
            if self.cfg.reversed:
                sampled_edge_mask_mlp = 1 - sampled_edge_mask_mlp
            masked_mlp_residual = einsum("batch position prev_head_idx d_model, prev_head_idx -> batch position d_model", residual, sampled_edge_mask_mlp)
        else:
            masked_mlp_residual = residual.sum(dim=2)
        
        normalized_resid_mid = self.ln2(masked_mlp_residual)
        mlp_out = self.mlp(normalized_resid_mid)
        mlp_out = einops.rearrange(mlp_out, "batch position d_model -> batch position 1 d_model")

        residual = torch.cat((residual, mlp_out), dim=2)

        return residual

    def get_weights(self):
        masks = {
            'q': self.edge_mask_attention_q_mask,
            'k': self.edge_mask_attention_k_mask,
            'v': self.edge_mask_attention_v_mask,
            'mlp': self.edge_mask_mlp_mask,
        }

        return masks

    def _set_edge(self, from_node, to_node, value):
        layer, node = from_node.split('.')
        if node == 'mlp':
            index = self.edge_mask_mlp_index[to_node]
            self.edge_mask_mlp_mask[index] = value
        else:
            node, head = node.split('_')
            head = int(head)
            if node == 'q':
                mask = self.edge_mask_attention_q_mask
                index = self.edge_mask_attention_q_index[to_node]
            elif node == 'k':
                mask = self.edge_mask_attention_k_mask
                index = self.edge_mask_attention_k_index[to_node]
            elif node == 'v':
                mask = self.edge_mask_attention_v_mask
                index = self.edge_mask_attention_v_index[to_node]
            else:
                raise ValueError
            mask[index][head] = value

    def get_edge(self, from_node, to_node):
        layer, node = from_node.split('.')
        if node == 'mlp':
            index = self.edge_mask_mlp_index[to_node]
            return self.edge_mask_mlp_mask[index]
        else:
            node, head = node.split('_')
            head = int(head)
            if node == 'q':
                mask = self.edge_mask_attention_q_mask
                index = self.edge_mask_attention_q_index[to_node]
            elif node == 'k':
                mask = self.edge_mask_attention_k_mask
                index = self.edge_mask_attention_k_index[to_node]
            elif node == 'v':
                mask = self.edge_mask_attention_v_mask
                index = self.edge_mask_attention_v_index[to_node]
            else:
                raise ValueError
            return mask[index][head]

    def get_connections(self, node):
        _, _node = node.split('.')
        if _node == 'mlp':
            return list(self.edge_mask_mlp_index.keys())

        _node, head = _node.split('_')
        if _node == 'q':
            return list(self.edge_mask_attention_q_index.keys())
        elif _node == 'k':
            return list(self.edge_mask_attention_k_index.keys())
        elif _node == 'v':
            return list(self.edge_mask_attention_v_index.keys())
        else:
            raise ValueError


class CircuitTransformerIntMask(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.embed = Embed(self.cfg)

        if self.cfg.positional_embedding_type != "rotary":
            self.pos_embed = PosEmbed(self.cfg)

        self.blocks = nn.ModuleList(
            [TransformerBlockIntMask(self.cfg, block_index) for block_index in range(self.cfg.n_layers)]
        )

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

        total_nodes = (self.cfg.n_heads + 1) * self.cfg.n_layers + 1
        self.edge_mask_output_mask = torch.ones((total_nodes,), device=self.cfg.device)
        self.edge_mask_output_index = {'input': 0}
        for i in range(self.cfg.n_layers):
            for j in range(self.cfg.n_heads):
                self.edge_mask_output_index[f'{i}.attn_{j}'] = i*(self.cfg.n_heads + 1) + j + 1
            self.edge_mask_output_index[f'{i}.mlp'] = i*(self.cfg.n_heads + 1) + j + 2

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
            sampled_output_mask = self.edge_mask_output_mask
            if self.cfg.reversed:
                sampled_output_mask = 1 - sampled_output_mask

            residual = einsum("batch position prev_head_idx d_model, prev_head_idx -> batch position d_model", residual, sampled_output_mask)
        else:
            residual = residual.sum(dim=2)

        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        return [logits]

    @classmethod
    def from_pretrained(cls, cfg):
        model = cls(cfg)
        # state_dict = HookedTransformer.from_pretrained_no_processing(model_name, **{'force_download': True}).state_dict()
        state_dict = HookedTransformer.from_pretrained(cfg.model_name).state_dict()
        model.load_state_dict(state_dict, strict=False)
        # for n, p in state_dict.items():
        #     if n in model.unmasked_params:
        #         model.unmasked_params[n] = p.clone()
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
    def prepare_origin_output(self, dl=None):
        # self.turn_off_weight_masks()
        # self.turn_off_edge_masks()

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

    @torch.no_grad
    def evaluate(self, dl=None, reverse=False):
        if dl is None:
            dl = self.dls.eval

        self.eval()

        # self.turn_on_weight_masks(deterministic=True, reverse=reverse)
        # self.turn_on_edge_masks(deterministic=True, reverse=reverse)

        # if self.cfg.use_weight_masks:
        #     _, _, weight_density = self.get_weight_density()
        # else:
        #     weight_density = 'na'
        # if self.cfg.use_edge_masks:
        #     _, _, edge_density = self.get_edge_density()
        # else:
        #     edge_density = 'na'

        # if not self.cfg.use_weight_masks:
        #     self.turn_off_weight_masks()
        # if not self.cfg.use_edge_masks:
        #     self.turn_off_edge_masks()

        self.cfg.reversed = reverse

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

        # self.turn_off_weight_masks()
        # self.turn_off_edge_masks()

        acc = correct / total

        results = {
            'acc': acc,
            'kl': torch.stack(kls).mean().item(),
            'faith_loss': torch.stack(faith_losses).mean().item(),
            # 'weight_density': weight_density,
            # 'edge_density': edge_density,
            'n_correct': correct,
            'total': total
        }
        return results

    def get_masks(self, flat=True):
        masks = {
            'output': self.edge_mask_output_mask
        }
        for i, block in enumerate(self.blocks):
            if flat:
                for k, v in block.get_weights().items():
                    masks[f'block_{i}_{k}'] = v
            else:
                masks[f'block_{i}'] = block.get_weights()
        return masks
    
    def density(self):
        masks = self.get_masks()
        total = int(sum(x.nelement() for x in masks.values()))
        on = int(sum(x.sum() for x in masks.values()))
        return on/total, on, total
    
    def _set_edge(self, to_node, value):
        index = self.edge_mask_output_index[to_node]
        self.edge_mask_output_mask[index] = value

    def set_edge(self, from_node, to_node, value):
        masks = self.get_masks()

        if from_node == 'output':
            self._set_edge(to_node, value)
        else:
            layer, node = from_node.split('.')
            layer = int(layer)
            self.blocks[layer]._set_edge(from_node, to_node, value)

    def get_edge(self, from_node, to_node):
        masks = self.get_masks()

        if from_node == 'output':
            index = self.edge_mask_output_index[to_node]
            return self.edge_mask_output_mask[index]
        else:
            layer, node = from_node.split('.')
            layer = int(layer)
            return self.blocks[layer].get_edge(from_node, to_node)

    def yield_from_nodes(self):
        '''Topo sorted'''

        yield 'output'
        for i in range(self.cfg.n_layers-1, -1, -1):
            yield f'{i}.mlp'
            for attn_type in 'qkv':
                for j in range(self.cfg.n_heads):
                    yield f'{i}.{attn_type}_{j}'

    def get_incoming_edges(self, to_node):
        results = {}
        for from_node in self.yield_from_nodes():
            try:
                edge = self.get_edge(from_node, to_node)
                results[(from_node, to_node)] = edge.item()
            except IndexError:
                pass
            except KeyError:
                pass
        return results

    def get_connections(self, node):
        _, _node = node.split('.')
        if _node == 'mlp':
            return list(self.edge_mask_mlp_index.keys())

        _node, head = _node.split('_')
        if _node == 'q':
            return list(self.edge_mask_attention_q_index.keys())
        elif _node == 'k':
            return list(self.edge_mask_attention_k_index.keys())
        elif _node == 'v':
            return list(self.edge_mask_attention_v_index.keys())
        else:
            raise ValueError

    def get_connections(self, node):
        if node == 'output':
            return list(self.edge_mask_output_index.keys())
        layer, _node = node.split('.')
        layer = int(layer)
        return self.blocks[layer].get_connections(node)


    def search_acdc(self, threshold=1e-4):

        num_examples = 100


        full_ds = self.tokenizer(self.dls.train.dataset['prompt'][:10], padding=True, return_tensors='pt')['input_ids']

        base_logits = self(full_ds)[0][:, -1, :]
        base_logprob = F.log_softmax(base_logits, dim=-1)

        def kl_div(_log_probs):
            return F.kl_div(_log_probs, base_logprob, log_target=True, reduction="none").sum(dim=-1).mean().item()
        
        curr_score = 0.

        for node in self.yield_from_nodes():

            print(self.density())

            comp = self.evaluate(reverse=True)
            results = self.evaluate()
            results['comp'] = comp['acc']

            print(results)


            if node == 'output':
                incoming_edges = {'temp': 1}
            elif 'mlp' in node:
                incoming_edges = self.get_incoming_edges(node)
            else:
                for attn in 'qkv':
                    if attn in node:
                        incoming_edges = self.get_incoming_edges(
                            node.replace(attn, 'attn'))

            if all(x==0. for x in incoming_edges.values()):
                print(f'no incoming edges left, skip node {sum(incoming_edges.values())}/{len(incoming_edges)}')
                for to_node in self.get_connections(node):
                    self.set_edge(node, to_node, 0)
                continue
            else:
                print(f'incoming edges left {int(sum(incoming_edges.values()))}/{len(incoming_edges)}')

            for to_node in self.get_connections(node):

                self.set_edge(node, to_node, 0)
                log_probs = F.log_softmax(self(full_ds)[0][:, -1, :], dim=-1)
                score = kl_div(log_probs)

                if score - curr_score < threshold:
                    # not important, remove the edge and continue
                    print('[remove]', node, to_node, score - curr_score, score, curr_score)
                    curr_score = score
                else:
                    # important, keep the edge and continue
                    print('[keep]', node, to_node, score - curr_score, score, curr_score)
                    self.set_edge(node, to_node, 1)

        print('search done!', threshold)
        print(self.density())

        comp = self.evaluate(reverse=True)
        results = self.evaluate()
        results['comp'] = comp['acc']

        print(results)

        # print(self.density())
        # print(self.evaluate())

    def reset_edges(self):
        for node in self.yield_from_nodes():
            for to_node in self.get_connections(node):
                self.set_edge(node, to_node, 1)

        print(self.density())
        print(self.evaluate())


    def search_random(self, prob=0.1):

        num_examples = 100
        
        curr_score = 0.

        for node in self.yield_from_nodes():

            if node == 'output':
                incoming_edges = {'temp': 1}
            elif 'mlp' in node:
                incoming_edges = self.get_incoming_edges(node)
            else:
                for attn in 'qkv':
                    if attn in node:
                        incoming_edges = self.get_incoming_edges(
                            node.replace(attn, 'attn'))

            if all(x==0. for x in incoming_edges.values()):
                print(f'no incoming edges left, skip node {sum(incoming_edges.values())}/{len(incoming_edges)}')
                for to_node in self.get_connections(node):
                    self.set_edge(node, to_node, 0)
                continue
            else:
                print(f'incoming edges left {int(sum(incoming_edges.values()))}/{len(incoming_edges)}')

            for to_node in self.get_connections(node):

                if random.uniform(0, 1) > prob:
                    self.set_edge(node, to_node, 0)

        print(self.density())

        comp = self.evaluate(reverse=True)
        results = self.evaluate()
        results['comp'] = comp['acc']

        print(results)