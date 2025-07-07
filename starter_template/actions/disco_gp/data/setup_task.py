import json
from argparse import Namespace

from .ioi_dataset import IOIGeneratorDataset

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

PARAREL_RELS = ['P103', 'P127', 'P136', 'P138', 'P140', 'P159', 'P176', 'P19', 'P20', 'P264', 'P279', 'P30', 'P364', 'P37', 'P407', 'P413', 'P449', 'P495', 'P740', 'P1376', 'P36']

def setup_task(disco_gp):
    if disco_gp.cfg.task_type == 'ioi':
        return setup_ioi(disco_gp)
    elif disco_gp.cfg.task_type == 'blimp':
        return setup_blimp(disco_gp)

def setup_pararel(disco_gp):
    task = disco_gp.cfg.task
    assert task in PARAREL_RELS, f"Task {task} not in {PARAREL_RELS}"

    ds_dict = {
        'prompt': [],
        'answer': [],
    }

    with open(data) as open_file:
        pararel_rel_data = json.load(open_file)
        data = pararel_rel_data[task]

    for entry in data:
        prompt = entry[0][0].replace(' [MASK] .', '')
        prompt = prompt.replace(' [MASK].', '')

        if '[MASK]' not in prompt:
            target = entry[0][1]
            if target:
                ds_dict['prompt'].append(prompt)
                ds_dict['answer'].append(target)


def setup_blimp(disco_gp):
    task = disco_gp.cfg.task
    prompts, targets, targets_good, targets_bad = [], [], [], []

    blimp_ds = load_dataset('blimp', task)
    for row in blimp_ds['train']:
        sg, sb = row['sentence_good'][:-1].split(), row['sentence_bad'][:-1].split()

        combined = []
        target_good, target_bad = None, None
        has_got_full_prefix = False
        for i, (tg, tb) in enumerate(zip(sg, sb)):

            if tg == tb:
                combined.append(tg)
            else:
                has_got_full_prefix = True
                target_good, target_bad = tg, tb

            if not has_got_full_prefix:
                continue

        sent = ' '.join(combined)
        prompts.append(sent)
        targets_good.append(' ' + target_good)
        targets_bad.append(' ' + target_bad)
        targets.append((target_good, target_bad))
    
    data_dict = {}
    data_dict['prompt'] = prompts
    data_dict['targets'] = targets

    tokenized = disco_gp.tokenizer(prompts, return_tensors='pt', padding=True)
    data_dict['input_ids'] = tokenized['input_ids']
    data_dict['seq_lens'] = tokenized['attention_mask'].sum(-1)

    # first_token_idx = 1 if disco_gp.tokenizer.add_bos_token else 0
    first_token_idx = 0

    data_dict['target good'] = [
        token_ids[first_token_idx] for token_ids in
        disco_gp.tokenizer(targets_good, add_special_tokens=False)['input_ids']
    ]
    data_dict['target bad'] = [
        token_ids[first_token_idx] for token_ids in
        disco_gp.tokenizer(targets_bad, add_special_tokens=False)['input_ids']
    ]

    ds = Dataset.from_dict(data_dict).train_test_split(0.3).with_format('torch')

    # data_dict['full_model_target_log_probs'] = full_model_target_log_probs
    # data_dict['full_model_pred_label'] = full_model_pred_labels

    train_dl = DataLoader(
        ds['train'],
        batch_size=disco_gp.cfg.batch_size,
    )
    eval_dl = DataLoader(
        ds['test'],
        batch_size=disco_gp.cfg.batch_size,
        shuffle=False,
    )

    return Namespace(train=train_dl, eval=eval_dl)


def setup_ioi(disco_gp):
    ioi_prompts = IOIGeneratorDataset(prompt_type="ABBA",
        N=disco_gp.cfg.n_ioi_data, tokenizer=disco_gp.tokenizer).ioi_prompts
    ds = setup_ioi_dataset(ioi_prompts, disco_gp).train_test_split(0.3).with_format('torch')

    train_dl = DataLoader(
        ds['train'],
        batch_size=disco_gp.cfg.batch_size,
    )
    eval_dl = DataLoader(
        ds['test'],
        batch_size=disco_gp.cfg.batch_size,
        shuffle=False,
    )

    return Namespace(train=train_dl, eval=eval_dl)

def setup_ioi_dataset(ioi_prompts, disco_gp):
    prompts, targets, io_list, s_list = [], [], [], []
    for item in ioi_prompts:
        prompt_full = item['text']
        prompt = prompt_full[:prompt_full.rfind(' ' + item['IO'])]
        prompts.append(prompt)
        targets.append((item['IO'], item['S']))

        io_list.append(item['IO'])
        s_list.append(item['S'])

    data_dict = {}
    data_dict['prompt'] = prompts
    data_dict['targets'] = targets

    tokenized = disco_gp.tokenizer(prompts, return_tensors='pt', padding=True)
    data_dict['input_ids'] = tokenized['input_ids']
    data_dict['seq_lens'] = tokenized['attention_mask'].sum(-1)

    data_dict['target good'] = [token_ids[0] for token_ids in disco_gp.tokenizer(io_list)['input_ids']]
    data_dict['target bad'] = [token_ids[0] for token_ids in disco_gp.tokenizer(s_list)['input_ids']]

    ds = Dataset.from_dict(data_dict)

    return ds
