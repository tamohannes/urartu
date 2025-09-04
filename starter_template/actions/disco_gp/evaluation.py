import torch
import torch.nn.functional as F

def compute_complete_loss_binary_label(batch_logits, batch_inputs):
    # batch_logits: (B, seq_len, vocab_size)
    batch_seq_lens = batch_inputs['seq_lens']
    batch_size = batch_logits.shape[0]

    logits_target_good = batch_logits[
        torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target good']]
    logits_target_bad = batch_logits[
        torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target bad']]
    logits_gb = torch.stack([logits_target_good, logits_target_bad], -1)  # (B,2)

    batch_probs_uniform = torch.ones(logits_gb.shape).to(logits_gb.device) * 0.5
    batch_complete_loss = F.cross_entropy(logits_gb, batch_probs_uniform)

    return batch_complete_loss, logits_gb


def compute_faith_loss_binary_label(batch_logits_masked, batch_inputs, original_logits=None):
    batch_seq_lens = batch_inputs['seq_lens']
    batch_size = batch_logits_masked.shape[0]

    logits_target_good_masked = batch_logits_masked[
        torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target good']]
    logits_target_bad_masked = batch_logits_masked[
        torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target bad']]

    logits_gb_masked = torch.stack([logits_target_good_masked, logits_target_bad_masked], -1)  # (B, 2)
    log_probs_target_masked = F.log_softmax(logits_gb_masked, -1)  # (B, 2)

    batch_pred = (logits_gb_masked[:, 0] > logits_gb_masked[:, 1]).long()
    batch_faith_loss = F.cross_entropy(logits_gb_masked,
        torch.zeros(batch_size).long().to(logits_gb_masked.device))
    
    if original_logits is not None:
        batch_kl = F.kl_div(
            original_logits.to(log_probs_target_masked.device).log_softmax(dim=-1),
            log_probs_target_masked,
            log_target=True
        ).cpu()
    else:
        batch_kl = None

    batch_n_correct = batch_pred.sum().item()

    results = {
        'faith': batch_faith_loss,
        'kl_div': batch_kl,
        'n_correct': batch_n_correct,
    }

    return results

def compute_faith_loss_multi_label(batch_logits_masked, batch_inputs, original_logits=None):
    # batch_logits: (B, seq_len, answer_idx_vocab_size)
    batch_seq_lens = batch_inputs['seq_lens']
    batch_size = batch_logits_masked.shape[0]
    log_probs_target_unmasked = original_logits  # (B, answer_idx_vocab_size)
    batch_labels = batch_inputs['answer_idx'].to(batch_logits_masked.device)

    batch_logits_masked_target = batch_logits_masked[torch.arange(batch_size), batch_seq_lens - 1]  # (B, answer_idx_vocab_size)
    batch_pred = torch.argsort(batch_logits_masked_target, -1)[:, -1]  # (B)
    
    batch_faith_loss = F.cross_entropy(batch_logits_masked_target, batch_labels)
    if original_logits is not None:
        batch_kl = F.kl_div(
            log_probs_target_unmasked.to(batch_logits_masked.device), 
            F.log_softmax(batch_logits_masked_target, -1),
            log_target=True
        ).cpu()
    else:
        batch_kl = None

    batch_n_correct = (batch_labels == batch_pred).sum().cpu().item()

    results = {
        'faith': batch_faith_loss,
        'kl_div': batch_kl,
        'n_correct': batch_n_correct,
    }

    return results

def compute_complete_loss_multi_label(batch_logits_masked, batch_inputs):
    # batch_logits: (B, seq_len, answer_idx_vocab_size)
    batch_seq_lens = batch_inputs['seq_lens']
    batch_size = batch_logits_masked.shape[0]

    batch_logits_masked_target = batch_logits_masked[torch.arange(batch_size), batch_seq_lens - 1]  # (B, answer_idx_vocab_size)

    batch_probs_uniform = torch.ones(batch_logits_masked_target.shape).to(batch_logits_masked_target.device) * (1. / batch_logits_masked_target.shape[-1])
    batch_complete_loss = F.cross_entropy(batch_logits_masked_target, batch_probs_uniform)
    
    return batch_complete_loss, batch_logits_masked_target