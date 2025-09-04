"""
Utilities to set up different evaluation/training tasks for DiscoGP.

Supported tasks
---
- IOI (Indirect Object Identification)
- BLiMP (Minimal pair grammaticality)
- PARArel (relational probing)

This module converts raw datasets into HuggingFace `Dataset`s and then into
PyTorch `DataLoader`s for train/eval/test splits according to configuration.
It also provides helper routines for tokenization and for filtering PARArel
examples to those the base model can already answer (so intervention analyses
aren't dominated by unanswerable cases).
"""

import json
from argparse import Namespace

from .ioi_dataset import IOIGeneratorDataset

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

import random
import pandas as pd
from collections import defaultdict 

PARAREL_RELS = [
    'P103', 'P127', 'P136', 'P138', 'P140', 'P159', 'P176', 'P19', 'P20', 'P264',
    'P279', 'P30', 'P364', 'P37', 'P407', 'P413', 'P449', 'P495', 'P740', 'P1376', 'P36'
]

def split_ratios(ratios):
    """Validate and convert (train, dev, test) ratios into split args.

    Parameters
    ---
    ratios: tuple[float, float, float]
        Expected to sum to 1.0, ordered as (train, dev, test).

    Returns
    ---
    dev_test: float
        Proportion used to carve out *all* evaluation data (dev+test) from the full set.
    test_over_dev: float
        Within the eval split, the fraction allocated to test (the remainder is dev).

    Notes
    ---
    We first split into [train] vs [dev+test], then split [dev+test] into [dev] vs [test].
    This mirrors `Dataset.train_test_split` usage.
    """
    assert sum(ratios) == 1.0, "Ratios must sum to 1.0"
    assert len(ratios) == 3, "Ratios must be a tuple of three values (train, dev, test)"
    _, dev, test = ratios

    return dev + test, test / (dev + test)

def setup_task(disco_gp):
    """Dispatch to a specific task setup based on `disco_gp.cfg.task_type`.

    Expected values: {'ioi', 'blimp', 'pararel'}.
    Returns a Namespace with three DataLoaders: train, eval, test.
    """
    if disco_gp.cfg.task_type == 'ioi':
        return setup_ioi(disco_gp)
    elif disco_gp.cfg.task_type == 'blimp':
        return setup_blimp(disco_gp)
    elif disco_gp.cfg.task_type == 'pararel':
        return setup_pararel(disco_gp)
"""     elif disco_gp.cfg.task_type == 'glue':
        return setup_sst2(disco_gp.cfg, disco_gp.tokenizer)
    elif disco_gp.cfg.task_type == 'boolq':
        data_dict = setup_boolq(disco_gp.cfg, disco_gp.tokenizer)
    elif disco_gp.cfg.task_type == 'winogrande':
        data_dict = setup_winogrande(disco_gp.cfg, disco_gp.tokenizer)
    elif disco_gp.cfg.task_type == 'copa':
        data_dict = setup_copa(disco_gp.cfg, disco_gp.tokenizer)
    elif disco_gp.cfg.task_type == 'snli':
        data_dict = setup_snli(disco_gp.cfg, disco_gp.tokenizer)
    return get_dataloader_big_dataset(disco_gp.cfg, data_dict, disco_gp.tokenizer) """

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

def setup_winogrande(disco_gp):
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

    tokenized = disco_gp.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)

    data_dict = {
        'prompts': prompts,
        'targets': targets,
        'input_ids': tokenized['input_ids'],
        'seq_lens': tokenized['attention_mask'].sum(-1),
        'target good': [ids[0] for ids in disco_gp.tokenizer(targets_good, add_special_tokens=False)['input_ids']],
        'target bad': [ids[0] for ids in disco_gp.tokenizer(targets_bad, add_special_tokens=False)['input_ids']],
    }
    
    return get_dataloader(disco_gp, ds, test_over_dev)

def setup_copa(disco_gp):
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

    tokenized = disco_gp.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    data_dict["input_ids"] = tokenized["input_ids"]
    data_dict["seq_lens"] = tokenized["attention_mask"].sum(-1)

    first_token_idx = 0
    data_dict["target good"] = [
        token_ids[first_token_idx] for token_ids in
        disco_gp.tokenizer(targets_good, add_special_tokens=False)["input_ids"]
    ]
    data_dict["target bad"] = [
        token_ids[first_token_idx] for token_ids in
        disco_gp.tokenizer(targets_bad, add_special_tokens=False)["input_ids"]
    ]
    # Split into train / (dev+test) first, then split latter into dev / test
    dev_test, test_over_dev = split_ratios(disco_gp.cfg.ds_split_ratios)

    ds = Dataset.from_dict(data_dict).train_test_split(dev_test).with_format('torch')

    return get_dataloader(disco_gp, ds, test_over_dev)



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


def setup_sst2(disco_gp, n_per_class=500):
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

    tokenized = disco_gp.tokenizer(prompts, return_tensors='pt', padding=True)
    data_dict = {
        'prompts': prompts,
        'targets': targets,
        'input_ids': tokenized['input_ids'],
        'seq_lens': tokenized['attention_mask'].sum(-1),
        'target good': [
            token_ids[0] for token_ids in
            disco_gp.tokenizer(targets_good, add_special_tokens=False)['input_ids']
        ],
        'target bad': [
            token_ids[0] for token_ids in
            disco_gp.tokenizer(targets_bad, add_special_tokens=False)['input_ids']
        ]
    }
    # Split into train / (dev+test) first, then split latter into dev / test
    dev_test, test_over_dev = split_ratios(disco_gp.cfg.ds_split_ratios)

    ds = Dataset.from_dict(data_dict).train_test_split(dev_test).with_format('torch')

    return get_dataloader(disco_gp, ds, test_over_dev)

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
    """Prepare BLiMP minimal-pair data as next-token prediction prompts.

    For each BLiMP example we find the longest common prefix between the
    `sentence_good` and `sentence_bad`, then use that as the prompt and treat the
    immediately following tokens as the contrasting targets.
    """
    task = disco_gp.cfg.paradigm
    prompts, targets, targets_good, targets_bad = [], [], [], []

     # Load a specific BLiMP paradigm (e.g., 'anaphor_gender_agreement')
    blimp_ds = load_dataset('blimp', task)
    for row in blimp_ds['train']:
        # Drop trailing period to avoid creating a separate token for '.'
        sg, sb = row['sentence_good'][:-1].split(), row['sentence_bad'][:-1].split()

        combined = []  # longest common prefix accumulator
        target_good, target_bad = None, None
        has_got_full_prefix = False
        for i, (tg, tb) in enumerate(zip(sg, sb)):
            if tg == tb:
                combined.append(tg)
            else:
                # Divergence point: tokens to be predicted
                has_got_full_prefix = True
                target_good, target_bad = tg, tb

            # Until divergence, we keep extending the prefix
            if not has_got_full_prefix:
                continue

        # Construct prompt from common prefix; targets are the next tokens.
        sent = ' '.join(combined)
        prompts.append(sent)
        targets_good.append(' ' + target_good)
        targets_bad.append(' ' + target_bad)
        targets.append((target_good, target_bad))
    
    # Build a dictionary suitable for a HF Dataset
    data_dict = {}
    data_dict['prompt'] = prompts
    data_dict['targets'] = targets  # for convenience/debugging

    # Tokenize prompts; keep input_ids and per-sequence lengths
    tokenized = disco_gp.tokenizer(prompts, return_tensors='pt', padding=True)
    data_dict['input_ids'] = tokenized['input_ids']
    data_dict['seq_lens'] = tokenized['attention_mask'].sum(-1)

    # For next-token evaluation we only need the ID of the *first* token in the target strings
    # (which begin with a leading space to trigger correct tokenization boundaries).
    first_token_idx = 0

    data_dict['target good'] = [
        token_ids[first_token_idx] for token_ids in
        disco_gp.tokenizer(targets_good, add_special_tokens=False)['input_ids']
    ]
    data_dict['target bad'] = [
        token_ids[first_token_idx] for token_ids in
        disco_gp.tokenizer(targets_bad, add_special_tokens=False)['input_ids']
    ]
        # Split into train / (dev+test) first, then split latter into dev / test
    dev_test, test_over_dev = split_ratios(disco_gp.cfg.ds_split_ratios)

    ds = Dataset.from_dict(data_dict).train_test_split(dev_test).with_format('torch')
    return get_dataloader(disco_gp, ds, test_over_dev)

def get_dataloader(disco_gp, ds, test_over_dev):
    eval_test_ds = ds['test'].train_test_split(test_over_dev)

    train_dl = DataLoader(
        ds['train'],
        batch_size=disco_gp.cfg.batch_size,
    )
    eval_dl = DataLoader(
        eval_test_ds['train'],
        batch_size=disco_gp.cfg.batch_size,
        shuffle=False,
    )
    test_dl = DataLoader(
        eval_test_ds['test'],
        batch_size=disco_gp.cfg.batch_size,
        shuffle=False,
    )

    return Namespace(train=train_dl, eval=eval_dl, test=test_dl)


def setup_ioi(disco_gp):
    """Prepare IOI (ABBA) prompts and corresponding target entities.

    We truncate each full prompt before the last occurrence of the IO token so
    that the model must predict the final name. Targets are (IO, S) pairs.
    """
    ioi_prompts = IOIGeneratorDataset(
        prompt_type="ABBA",
        N=disco_gp.cfg.n_ioi_data,
        tokenizer=disco_gp.tokenizer,
    ).ioi_prompts

    # Create HF dataset and then split into train/dev/test
    dev_test, test_over_dev = split_ratios(disco_gp.cfg.ds_split_ratios)
    ds = setup_ioi_dataset(ioi_prompts, disco_gp).train_test_split(dev_test).with_format('torch')
    return get_dataloader(disco_gp, ds, test_over_dev)

def setup_ioi_dataset(ioi_prompts, disco_gp):
    """Convert raw IOI prompt dicts to a tokenized HF Dataset.

    Parameters
    ---
    ioi_prompts: list[dict]
        Items contain 'text', 'IO', and 'S'.
    disco_gp: object
        Expected to expose a `tokenizer` compatible with HF tokenizers.
    """
    prompts, targets, io_list, s_list = [], [], [], []
    for item in ioi_prompts:
        prompt_full = item['text']
        # Keep everything up to the last space before the IO token (so model predicts IO)
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

    # Single-token assumptions: IO and S are expected to be single tokens under the tokenizer.
    data_dict['target good'] = [token_ids[0] for token_ids in disco_gp.tokenizer(io_list)['input_ids']]
    data_dict['target bad'] = [token_ids[0] for token_ids in disco_gp.tokenizer(s_list)['input_ids']]

    ds = Dataset.from_dict(data_dict)
    return ds

def process_pararel_data(disco_gp, ds_dict):
    """Add class indices/vocab for PARArel answers and return a HF Dataset.

    - Tokenizes each answer (without special tokens) and takes the first token id
      as the class label. This assumes single-token answers after a leading space.
    - Stores `answer_idx_vocab` both on `disco_gp` and the returned dataset for later use.
    """
    answer_token_ids = [
        disco_gp.tokenizer(answer, add_special_tokens=False)['input_ids'][0] for answer in ds_dict['answer']
    ]
    # Deduplicate while preserving order via set+sort of numeric ids
    answer_idx_vocab = list(set(answer_token_ids))
    answer_idx_vocab.sort()
    disco_gp.answer_idx_vocab = answer_idx_vocab

    class_idx_list = [answer_idx_vocab.index(x) for x in answer_token_ids]
    ds_dict['answer_idx'] = class_idx_list

    ds = Dataset.from_dict(ds_dict)
    ds.answer_idx_vocab = answer_idx_vocab
    return ds

@torch.no_grad()
def filter_out_unanswerable(disco_gp, ds):
    """Keep only PARArel examples the *base* model answers correctly.

    Rationale
    ---
    For causal analysis/interventions, we often want to avoid examples the base
    model already fails on. This routine runs the current model in eval mode
    (with all masks off) and filters to correct items.

    Side effects
    ---
    - Temporarily turns off weight and edge masks via `disco_gp.turn_off_*` hooks.
    - Attaches `answer_idx_vocab` to the returned dataset for downstream indexing.
    """
    dl = DataLoader(
        ds,
        batch_size=disco_gp.cfg.batch_size,
    )
    # Ensure evaluations reflect the unmodified model
    disco_gp.turn_off_weight_masks()
    disco_gp.turn_off_edge_masks()

    filtered_ds_dict = {
        'prompt': [],
        'answer': [],
        'answer_idx': [],
    }

    for batch in dl:
        bs = torch.arange(len(batch['prompt']))
        batch_input = disco_gp.tokenizer(
            batch['prompt'], return_tensors='pt', padding=True
        ).to(disco_gp.cfg.device)
        lengths = batch_input.attention_mask.sum(dim=1)

        # Forward pass; assume disco_gp(...) returns (logits, *extras)
        logits = disco_gp(batch_input.input_ids)[0]
        # Select the logits at the last context position for each sequence
        # then restrict to the label vocabulary for classification.
        pred_labels = logits[bs, lengths - 1][:, ds.answer_idx_vocab].argmax(-1)
        correctness = (pred_labels.cpu() == batch['answer_idx'])

        for i, correct in enumerate(correctness):
            if correct:
                filtered_ds_dict['prompt'].append(batch['prompt'][i])
                filtered_ds_dict['answer'].append(batch['answer'][i])
                filtered_ds_dict['answer_idx'].append(batch['answer_idx'][i])

    # Cache tokenization for faster later use
    tokenized = disco_gp.tokenizer(filtered_ds_dict['prompt'], return_tensors='pt', padding=True)
    filtered_ds_dict['input_ids'] = tokenized['input_ids']
    filtered_ds_dict['seq_lens'] = tokenized['attention_mask'].sum(-1)

    new_ds = Dataset.from_dict(filtered_ds_dict)
    new_ds.answer_idx_vocab = ds.answer_idx_vocab
    return new_ds


def setup_pararel(disco_gp):
    """Prepare PARArel dataset and DataLoaders.

    Steps
    ---
    1) Load preprocessed PARArel JSON from `disco_gp.cfg.pararel_data_path`.
    2) Select relation IDs from space-separated string `pararel_rel_ids`.
    3) Build (prompt, answer) pairs; answers are prefixed with a leading space.
    4) Convert to HF Dataset with class indices via `process_pararel_data`.
    5) Filter out examples the base model cannot answer via `filter_out_unanswerable`.
    6) Split into train/dev/test and wrap in DataLoaders.
    """

    # 1) Load raw relation data (a dict: rel_id -> list of entries)
    with open(disco_gp.cfg.pararel_data_path) as open_file:
        pararel_rel_data = json.load(open_file)

    # 2) Which relations to include (space-separated in config)
    rel_ids = disco_gp.cfg.pararel_rel_ids.split(' ')

    # Collect all entries across the chosen relations
    data = []
    for rel_id in rel_ids:
        data += pararel_rel_data[rel_id]

    # 3) Build prompt/answer fields from the PARArel template entries
    ds_dict = {
        'prompt': [],
        'answer': [],
    }
    for entry in data:
        # entry[0] is usually a (template, answer) pair. Strip the trailing "[MASK] ." variations.
        prompt = entry[0][0].replace(' [MASK] .', '')
        prompt = prompt.replace(' [MASK].', '')
        assert '[MASK]' not in prompt
        target = entry[0][1]
        ds_dict['prompt'].append(prompt)
        ds_dict['answer'].append(' ' + target)  # leading space for tokenizer boundary

    # 4) Add class indices and vocab
    ds = process_pararel_data(disco_gp, ds_dict)

    # 5) Only keep examples that the base model can answer correctly
    ds = filter_out_unanswerable(disco_gp, ds)

    # 6) Split and wrap in loaders
    dev_test, test_over_dev = split_ratios(disco_gp.cfg.ds_split_ratios)
    ds = ds.train_test_split(dev_test).with_format('torch')
    return get_dataloader(disco_gp, ds, test_over_dev)
    
"""     eval_test_ds = ds['test'].train_test_split(test_over_dev)

    train_dl = DataLoader(
        ds['train'],
        batch_size=disco_gp.cfg.batch_size,
    )
    eval_dl = DataLoader(
        eval_test_ds['train'],
        batch_size=disco_gp.cfg.batch_size,
        shuffle=False,
    )
    test_dl = DataLoader(
        eval_test_ds['test'],
        batch_size=disco_gp.cfg.batch_size,
        shuffle=False,
    )

    return Namespace(train=train_dl, eval=eval_dl, test=test_dl) """

