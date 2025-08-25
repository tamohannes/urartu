import json
import os
from argparse import Namespace

from .ioi_dataset import IOIGeneratorDataset

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import random
import pandas as pd
from collections import defaultdict 

PARAREL_RELS = ['P103', 'P127', 'P136', 'P138', 'P140', 'P159', 'P176', 'P19', 'P20', 'P264', 'P279', 'P30', 'P364', 'P37', 'P407', 'P413', 'P449', 'P495', 'P740', 'P1376', 'P36']

def setup_task(disco_gp):
    data_dict = {}
    if disco_gp.cfg.task_type == 'ioi':
        data_dict = setup_ioi_dataset(disco_gp.cfg, disco_gp.tokenizer)
        return get_dataloader(disco_gp.cfg, data_dict)
    elif disco_gp.cfg.task_type == 'blimp':
        data_dict = setup_blimp_dataset(disco_gp.cfg, disco_gp.tokenizer)
        return get_dataloader(disco_gp.cfg, data_dict)
    elif disco_gp.cfg.task_type == 'glue':
        data_dict = setup_sst2(disco_gp.cfg, disco_gp.tokenizer)
        return get_dataloader(disco_gp.cfg, data_dict)
    elif disco_gp.cfg.task_type == 'boolq':
        data_dict = setup_boolq(disco_gp.cfg, disco_gp.tokenizer)
    elif disco_gp.cfg.task_type == 'winogrande':
        data_dict = setup_winogrande(disco_gp.cfg, disco_gp.tokenizer)
    elif disco_gp.cfg.task_type == 'copa':
        data_dict = setup_copa(disco_gp.cfg, disco_gp.tokenizer)
    elif disco_gp.cfg.task_type == 'snli':
        data_dict = setup_snli(disco_gp.cfg, disco_gp.tokenizer)
    return get_dataloader_big_dataset(disco_gp.cfg, data_dict, disco_gp.tokenizer)

def get_dataloader_big_dataset(cfg, data_dict, tokenizer, split_ratio=0.3):
    """
    General dataloader builder for large NLP datasets
    """
    ds = Dataset.from_dict(data_dict).train_test_split(split_ratio, seed=42)

    # Collate function: tokenize batch dynamically
    def collate_fn(batch):
        prompts = [item["prompts"] for item in batch]

        # Tokenize only the prompts for input_ids + attention_mask
        tokenized = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Prepare batch dict
        batch_dict = {
            "prompts": prompts,
            "targets": [item["targets"] for item in batch],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "seq_lens": tokenized["attention_mask"].sum(-1),
            "target good": [item["target good"] for item in batch],
            "target bad": [item["target bad"] for item in batch],
        }
        return batch_dict

    train_dl = DataLoader(
        ds["train"],
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers if hasattr(cfg, "num_workers") else 0,
        pin_memory=True
    )

    eval_dl = DataLoader(
        ds["test"],
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers if hasattr(cfg, "num_workers") else 0,
        pin_memory=True
    )

    return Namespace(train=train_dl, eval=eval_dl)
 

def balanced_subset(dataset, n_per_class):
    """Return balanced subset with equal yes/no answers."""
    df = pd.DataFrame(dataset)
    yes_df = df[df["answer"] == True].sample(n_per_class, random_state=42)
    no_df = df[df["answer"] == False].sample(n_per_class, random_state=42)
    balanced_df = pd.concat([yes_df, no_df]).sample(frac=1, random_state=42)  # shuffle
    return Dataset.from_pandas(balanced_df)

def setup_boolq(cfg, tokenizer, subset_size=None, n_per_class=500):
    boolq_ds = load_dataset("boolq")

    prompts, targets, targets_good, targets_bad = [], [], [], []

    # Sampling options
    if n_per_class is not None:  # Balanced sampling
        boolq_train = balanced_subset(boolq_ds["train"], n_per_class)
    elif subset_size is not None:  # Random subset
        boolq_train = boolq_ds["train"].shuffle(seed=42).select(range(subset_size))
    else:
        boolq_train = boolq_ds["train"]

    for row in boolq_train:
        passage = row["passage"].strip()
        question = row["question"].strip()
        label = bool(row["answer"])  # True/False

        # Make explicit yes/no style question
        prompt = f"Passage: {passage}\nQuestion: {question}? ( yes or no)\nAnswer:"
        
        good = " yes" if label else " no"
        bad = " no" if label else " yes"

        prompts.append(prompt)
        targets.append((good.strip(), bad.strip()))
        targets_good.append(good)
        targets_bad.append(bad)

    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    data_dict = {
        "prompts": prompts,
        "targets": targets,
        "input_ids": tokenized["input_ids"],
        "seq_lens": tokenized["attention_mask"].sum(-1),
    }

    first_token_idx = 0
    data_dict["target good"] = [
        token_ids[first_token_idx]
        for token_ids in tokenizer(targets_good, add_special_tokens=False)["input_ids"]
    ]
    data_dict["target bad"] = [
        token_ids[first_token_idx]
        for token_ids in tokenizer(targets_bad, add_special_tokens=False)["input_ids"]
    ]
    return data_dict

def setup_snli(cfg, tokenizer, subset_size=None, n_per_class=500):
    snli_ds = load_dataset("snli")

    # Filter out -1 labels (missing)
    def filter_valid(ex):
        return ex["label"] in [0, 1, 2]

    snli_train = snli_ds["train"].filter(filter_valid)

    # Optional sampling
    if n_per_class is not None:
        samples = []
        for label in [0, 1, 2]:  # entailment=0, neutral=1, contradiction=2
            label_subset = [ex for ex in snli_train if ex["label"] == label]
            samples.extend(random.sample(label_subset, min(n_per_class, len(label_subset))))
        snli_train = Dataset.from_list(samples)
    elif subset_size is not None:
        snli_train = snli_train.shuffle(seed=42).select(range(subset_size))

    prompts, targets, tg, tb = [], [], [], []

    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    for ex in snli_train:
        premise = ex["premise"].strip()
        hypothesis = ex["hypothesis"].strip()
        gold_label = label_map[ex["label"]]

        # Instruction-style prompt with explicit label choices
        instruction = (
            "Decide the relationship between the premise and the hypothesis. "
            "Choose only one of: entailment, neutral, contradiction."
        )
        input_text = f"Premise: {premise}\nHypothesis: {hypothesis}"
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"

        # Good target
        good = f" {gold_label}"

        # Pick a "bad" target randomly from the other two classes
        bad_label = random.choice([l for l in label_map.values() if l != gold_label])
        bad = f" {bad_label}"

        prompts.append(prompt)
        targets.append((good.strip(), bad.strip()))
        tg.append(good)
        tb.append(bad)
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True,
                                   truncation=True)

    data_dict = {
        "prompts": prompts,
        "targets": targets,
        "input_ids": tokenized["input_ids"],
        "seq_lens": tokenized["attention_mask"].sum(-1)
    }

    data_dict["target good"] = [
        ids[0] for ids in tokenizer(tg, add_special_tokens=False)["input_ids"]
    ]
    data_dict["target bad"] = [
        ids[0] for ids in tokenizer(tb, add_special_tokens=False)["input_ids"]
    ]
    return data_dict

def setup_winogrande(cfg, tokenizer):
    ds = load_dataset("winogrande", "winogrande_s")["train"]

    prompts, targets, targets_good, targets_bad = [], [], [], []

    for ex in ds:
        sentence = ex["sentence"]
        option1, option2 = ex["option1"], ex["option2"]
        correct = option1 if ex["answer"] == "1" else option2
        incorrect = option2 if correct == option1 else option1

        # Replace the blank "_" with nothing for prompt clarity
        prompt = sentence.replace("_", "_____") + "\nFill in the blank:"
        prompts.append(prompt)

        targets.append((correct, incorrect))
        targets_good.append(" " + correct)
        targets_bad.append(" " + incorrect)

    tokenized = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)

    data_dict = {
        'prompts': prompts,
        'targets': targets,
        'input_ids': tokenized['input_ids'],
        'seq_lens': tokenized['attention_mask'].sum(-1),
        'target good': [ids[0] for ids in tokenizer(targets_good, add_special_tokens=False)['input_ids']],
        'target bad': [ids[0] for ids in tokenizer(targets_bad, add_special_tokens=False)['input_ids']],
    }
    return data_dict

def setup_copa(cfg, tokenizer):
    # Load dataset
    copa_ds = load_dataset("super_glue", "copa")

    prompts, targets, targets_good, targets_bad = [], [], [], []

    for row in copa_ds['train']:
        premise = row['premise']
        choice1 = row['choice1']
        choice2 = row['choice2']
        label = row['label']  # 0 means choice1 is correct, 1 means choice2 is correct

        # Determine connector word
        connector = " because " if row['question'] == "cause" else " so "
        prompt = premise + connector

        # Assign good/bad targets
        if label == 0:
            target_good, target_bad = choice1, choice2
        else:
            target_good, target_bad = choice2, choice1

        prompts.append(prompt)
        targets.append((target_good, target_bad))
        targets_good.append(" " + target_good)
        targets_bad.append(" " + target_bad)

    # Build data_dict
    data_dict = {
        "prompts": prompts,
        "targets": targets,
    }

    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    data_dict["input_ids"] = tokenized["input_ids"]
    data_dict["seq_lens"] = tokenized["attention_mask"].sum(-1)

    first_token_idx = 0
    data_dict["target good"] = [
        token_ids[first_token_idx] for token_ids in
        tokenizer(targets_good, add_special_tokens=False)["input_ids"]
    ]
    data_dict["target bad"] = [
        token_ids[first_token_idx] for token_ids in
        tokenizer(targets_bad, add_special_tokens=False)["input_ids"]
    ]
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