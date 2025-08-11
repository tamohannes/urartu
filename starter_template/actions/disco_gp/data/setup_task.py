import json
import os
from argparse import Namespace

from .ioi_dataset import IOIGeneratorDataset

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import random
from collections import defaultdict 

PARAREL_RELS = ['P103', 'P127', 'P136', 'P138', 'P140', 'P159', 'P176', 'P19', 'P20', 'P264', 'P279', 'P30', 'P364', 'P37', 'P407', 'P413', 'P449', 'P495', 'P740', 'P1376', 'P36']

def setup_task(disco_gp):
    data_dict = {}
    if disco_gp.cfg.task_type == 'ioi':
        data_dict = setup_ioi_dataset(disco_gp.cfg, disco_gp.tokenizer)
        #return setup_ioi(disco_gp.cfg, disco_gp.tokenizer)
    elif disco_gp.cfg.task_type == 'blimp':
        data_dict = setup_blimp_dataset(disco_gp.cfg, disco_gp.tokenizer)
        #return setup_blimp(disco_gp.cfg, disco_gp.tokenizer)
    elif disco_gp.cfg.task_type == 'glue':
        data_dict = setup_sst2(disco_gp.cfg, disco_gp.tokenizer)
    elif disco_gp.cfg.task_type == 'coqa':
        data_dict = setup_coqa(disco_gp.cfg, disco_gp.tokenizer)
    return get_dataloader(disco_gp.cfg, data_dict)

def setup_coqa(cfg, tokenizer, max_turns=1, seed=42, safety_buffer=20):
    """
    Keeps full question + answer; trims story tokens precisely so that the tokenized
    prompt length <= tokenizer.model_max_length.
    """
    random.seed(seed)
    ds = load_dataset("stanfordnlp/coqa")
    train = ds["train"]

    # collect all answers for realistic distractors
    all_answers = [ans for conv in train for ans in conv["answers"]["input_text"] if ans.strip()]

    prompts, targets, targets_good, targets_bad = [], [], [], []
    max_len = int(tokenizer.model_max_length)

    too_long_count = 0
    for conv in train:
        story = conv["story"]

        # Pre-encode story once to ids so we can slice by token ids precisely
        story_ids_full = tokenizer(
            "Passage: " + story,
            add_special_tokens=False
        )["input_ids"]

        for i, (q, a) in enumerate(zip(conv["questions"], conv["answers"]["input_text"])):
            if i >= max_turns:
                break

            # question + "Answer:" tail (we keep this whole)
            q_part = f"\nQuestion: {q}\nAnswer:"
            q_part_ids = tokenizer(q_part, add_special_tokens=False)["input_ids"]
            base_len = len(q_part_ids)

            # compute allowed story length in token ids (exact)
            allowed_story_tokens = max_len - base_len - safety_buffer
            if allowed_story_tokens < 0:
                # question alone too long (very rare). fallback: truncate question.
                allowed_story_tokens = 0

            # slice story ids to allowed size
            short_story_ids = story_ids_full[:allowed_story_tokens]
            short_story = tokenizer.decode(
                short_story_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            prompt = short_story + q_part

            # verify exact tokenized length of the constructed prompt
            enc = tokenizer(prompt, add_special_tokens=False)
            if len(enc["input_ids"]) > max_len:
                # fallback: aggressively truncate the encoded prompt to max_len - safety_buffer
                too_long_count += 1
                enc_ids = enc["input_ids"][: max_len - safety_buffer]
                prompt = tokenizer.decode(
                    enc_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

                # final check (should be <= max_len)
                enc = tokenizer(prompt, add_special_tokens=False)
                if len(enc["input_ids"]) > max_len:
                    # as a last resort, truncate to exactly max_len
                    enc_ids = enc["input_ids"][:max_len]
                    prompt = tokenizer.decode(
                        enc_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )

            # Build targets: good is true answer; bad is sampled realistic distractor
            good = " " + a.strip()
            bad = " " + random.choice(all_answers) if all_answers else " I don't know"
            # avoid identical distractor
            retries = 0
            while bad.strip() == a.strip() and retries < 10:
                bad = " " + random.choice(all_answers)
                retries += 1

            prompts.append(prompt)
            targets.append((a.strip(), bad.strip()))
            targets_good.append(good)
            targets_bad.append(bad)

    # final tokenization (batch) with truncation & padding for safety
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )

    # sanity assert (optional) to ensure nothing exceeds max_len
    max_observed = tokenized["input_ids"].shape[1]
    assert max_observed <= max_len, f"Observed {max_observed} > max_len {max_len}"

    data_dict = {
        "prompts": prompts,
        "targets": targets,
        "input_ids": tokenized["input_ids"],
        "seq_lens": tokenized["attention_mask"].sum(-1),
        "target good": [
            token_ids[0] for token_ids in tokenizer(targets_good, add_special_tokens=False)["input_ids"]
        ],
        "target bad": [
            token_ids[0] for token_ids in tokenizer(targets_bad, add_special_tokens=False)["input_ids"]
        ],
    }
    return data_dict


def get_stratified_sst2_subset(n_per_class=500, seed=42):
    random.seed(seed)
    full_dataset = load_dataset("glue", "sst2")
    train_set = full_dataset["train"]

    # Group examples by label (0 = negative, 1 = positive)
    label_buckets = defaultdict(list)
    for ex in train_set:
        label_buckets[ex["label"]].append(ex)

    # Check if enough examples exist for each class
    for label, samples in label_buckets.items():
        if len(samples) < n_per_class:
            raise ValueError(f"Not enough samples for label {label}")

    # Sample n_per_class from each label
    subset_data = []
    for label in [0, 1]:
        sampled = random.sample(label_buckets[label], n_per_class)
        subset_data.extend(sampled)

    # Shuffle the final subset
    random.shuffle(subset_data)

    return Dataset.from_list(subset_data)


def setup_sst2(cfg, tokenizer, n_per_class=500):
    dataset = get_stratified_sst2_subset(n_per_class)

    
    prompts, targets, targets_good, targets_bad = [], [], [], []

    for row in dataset:
        sentence = row['sentence']
        label = row['label']  # 1 = positive, 0 = negative

        prompts.append(sentence)

        good = ' positive' if label == 1 else ' negative'
        bad = ' negative' if label == 1 else ' positive'

        targets_good.append(good)
        targets_bad.append(bad)
        targets.append((good.strip(), bad.strip()))

    tokenized = tokenizer(prompts, return_tensors='pt', padding=True)
    data_dict = {
        'prompts': prompts,
        'targets': targets,
        'input_ids': tokenized['input_ids'],
        'seq_lens': tokenized['attention_mask'].sum(-1),
        'target good': [
            token_ids[0] for token_ids in
            tokenizer(targets_good, add_special_tokens=False)['input_ids']
        ],
        'target bad': [
            token_ids[0] for token_ids in
            tokenizer(targets_bad, add_special_tokens=False)['input_ids']
        ]
    }
    return data_dict

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

def setup_blimp_dataset(cfg, tokenizer):
    task = cfg.task
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

    tokenized = tokenizer(prompts, return_tensors='pt', padding=True)
    data_dict['input_ids'] = tokenized['input_ids']
    data_dict['seq_lens'] = tokenized['attention_mask'].sum(-1)

    # first_token_idx = 1 if disco_gp.tokenizer.add_bos_token else 0
    first_token_idx = 0

    data_dict['target good'] = [
        token_ids[first_token_idx] for token_ids in
        tokenizer(targets_good, add_special_tokens=False)['input_ids']
    ]
    data_dict['target bad'] = [
        token_ids[first_token_idx] for token_ids in
        tokenizer(targets_bad, add_special_tokens=False)['input_ids']
    ]
    return data_dict

def get_dataloader(cfg, data_dict):
    ds = Dataset.from_dict(data_dict).train_test_split(0.3).with_format('torch')
    train_dl = DataLoader(
        ds['train'],
        batch_size=cfg.batch_size,
    )
    eval_dl = DataLoader(
        ds['test'],
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    return Namespace(train=train_dl, eval=eval_dl)


def setup_blimp(cfg, tokenizer):
    data_dict = setup_blimp_dataset(cfg, tokenizer)
    ds = Dataset.from_dict(data_dict).train_test_split(0.3).with_format('torch')

    # data_dict['full_model_target_log_probs'] = full_model_target_log_probs
    # data_dict['full_model_pred_label'] = full_model_pred_labels

    train_dl = DataLoader(
        ds['train'],
        batch_size=cfg.batch_size,
    )
    eval_dl = DataLoader(
        ds['test'],
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    return Namespace(train=train_dl, eval=eval_dl)


def setup_ioi(cfg, tokenizer):
    data_dict = setup_ioi_dataset(cfg, tokenizer)
    ds = Dataset.from_dict(data_dict).train_test_split(0.3).with_format('torch')
    train_dl = DataLoader(
        ds['train'],
        batch_size=cfg.batch_size,
    )
    eval_dl = DataLoader(
        ds['test'],
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    return Namespace(train=train_dl, eval=eval_dl)

def setup_ioi_dataset(cfg, tokenizer):
    ioi_prompts = IOIGeneratorDataset(prompt_type="ABBA",
        N=cfg.n_ioi_data, tokenizer=tokenizer).ioi_prompts
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

    tokenized = tokenizer(prompts, return_tensors='pt', padding=True)
    data_dict['input_ids'] = tokenized['input_ids']
    data_dict['seq_lens'] = tokenized['attention_mask'].sum(-1)

    data_dict['target good'] = [token_ids[0] for token_ids in tokenizer(io_list)['input_ids']]
    data_dict['target bad'] = [token_ids[0] for token_ids in tokenizer(s_list)['input_ids']]

    return data_dict


def get_data_as_dict(cfg, tokenizer):
    data_dict = {}
    if cfg.task_type == 'ioi':
        data_dict = setup_ioi_dataset(cfg, tokenizer)
    elif cfg.task_type == 'blimp':
        data_dict = setup_blimp_dataset(cfg, tokenizer)
    return data_dict