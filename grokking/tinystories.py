"""Prepare a small offline TinyStories corpus and fixed next-byte examples."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset


def prepare(directory="data/tinystories", train_stories=128, validation_stories=128):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    if min(train_stories, validation_stories) <= 0:
        raise ValueError("story counts must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    revision = HfApi().dataset_info("roneneldan/TinyStories").sha
    manifest = {"source": "roneneldan/TinyStories", "revision": revision, "files": {}}
    seen = set()
    for split, count in (("train", train_stories), ("validation", validation_stories)):
        stories = []
        for row in load_dataset(manifest["source"], revision=revision, split=split, streaming=True):
            text = row["text"].strip()
            if not text or text in seen:
                continue
            seen.add(text)
            stories.append(text)
            if len(stories) == count:
                break
        if len(stories) != count:
            raise ValueError(f"not enough unique stories in {split}")
        payload = ("\n".join(json.dumps(text, ensure_ascii=False) for text in stories) + "\n").encode("utf-8")
        filename = f"{split}.jsonl"
        (directory / filename).write_bytes(payload)
        manifest["files"][filename] = {"sha256": hashlib.sha256(payload).hexdigest(), "stories": count}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def next_byte_dataset(stories, context_length, max_examples, seed, excluded_contexts=None):
    if min(context_length, max_examples) <= 0:
        raise ValueError("context length and example count must be positive")
    examples = set()
    excluded_contexts = excluded_contexts or set()
    for story in stories:
        tokens = story.encode("utf-8")
        for index in range(context_length, len(tokens)):
            context = tokens[index-context_length:index]
            if context not in excluded_contexts:
                examples.add((context, tokens[index]))
    if not examples:
        raise ValueError("no usable contexts; reduce --context-length or prepare more stories")
    examples = sorted(examples)
    indices = np.random.default_rng(seed).choice(len(examples), min(max_examples, len(examples)), replace=False)
    selected = [examples[index] for index in indices]
    return TensorDataset(torch.tensor([list(context) for context, _ in selected], dtype=torch.long),
                         torch.tensor([target for _, target in selected], dtype=torch.long))


def load_tinystories(directory, context_length=64, train_examples=512, validation_examples=512, seed=42):
    directory = Path(directory)
    if not (directory / "manifest.json").exists():
        raise FileNotFoundError("Prepare data first: python -m grokking.tinystories")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    stories = {}
    for split in ("train", "validation"):
        filename = f"{split}.jsonl"
        payload = (directory / filename).read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["files"][filename]["sha256"]:
            raise ValueError(f"TinyStories checksum mismatch: {filename}")
        stories[split] = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    if set(stories["train"]) & set(stories["validation"]):
        raise ValueError("TinyStories train and validation stories overlap")
    train = next_byte_dataset(stories["train"], context_length, train_examples, seed)
    excluded = {bytes(row.tolist()) for row in train.tensors[0]}
    validation = next_byte_dataset(stories["validation"], context_length, validation_examples, seed + 1, excluded)
    metadata = dict(manifest, tokenizer="UTF-8 bytes (IDs 0..255)", context_length=context_length,
                    train_examples=len(train), validation_examples=len(validation), seed=seed)
    return train, validation, metadata


def main():
    parser = argparse.ArgumentParser(description="Download a small TinyStories subset for offline training")
    parser.add_argument("--data-dir", default="data/tinystories")
    parser.add_argument("--train-stories", type=int, default=128)
    parser.add_argument("--validation-stories", type=int, default=128)
    args = parser.parse_args()
    prepare(args.data_dir, args.train_stories, args.validation_stories)


if __name__ == "__main__":
    main()
