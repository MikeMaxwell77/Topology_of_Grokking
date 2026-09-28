"""Pinned upstream Simple Agreement data, adapted to binary prefix classification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import torch
from torch.utils.data import Dataset

REVISION = "4744caf2fac1cc94beb8775b9b36d3fbf517cbae"
REPOSITORY = "https://github.com/kabirahuja2431/transformers-hg"
GRAMMAR = "agreement_hr_agreement_linear"
SPLITS = ("train", "val", "g1_test", "g2_test")
DEFAULT_DIRECTORY = Path("data/simple_agreement")


def download_agreement(directory=DEFAULT_DIRECTORY):
    """Explicit download only; training itself never accesses the network."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {split: f"data_utils/grammar_gen_data/{GRAMMAR}.{split}" for split in SPLITS}
    paths["tag_token_map.txt"] = "cfgs/tag_token_map.txt"
    paths["agreement_hr.gr"] = "cfgs/agreement_hr.gr"
    manifest = {"repository": REPOSITORY, "revision": REVISION, "grammar": GRAMMAR, "files": {}}
    for filename, source in paths.items():
        url = f"https://raw.githubusercontent.com/kabirahuja2431/transformers-hg/{REVISION}/{source}"
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        (directory / filename).write_bytes(payload)
        manifest["files"][filename] = {"url": url, "sha256": hashlib.sha256(payload).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_sentence(sentence, tags):
    """Low-diversity grammar has no relative clauses: its first verb is the main verb."""
    tokens = sentence.split()
    verb_indices = [i for i, token in enumerate(tokens)
                    if tags.get(token, set()) & {"v_s_trans", "v_p_trans", "v_s_intrans", "v_p_intrans"}]
    if len(verb_indices) != 1:
        raise ValueError(f"expected exactly one main verb: {sentence}")
    index = verb_indices[0]
    prefix = tokens[:index]
    expected = ["det", "noun"] if index == 2 else ["det", "noun", "prep", "det", "noun"]
    if len(prefix) != len(expected):
        raise ValueError(f"unsupported agreement prefix: {sentence}")
    for token, category in zip(prefix, expected):
        allowed = {"n_s", "n_p"} if category == "noun" else {category}
        if not tags.get(token, set()) & allowed:
            raise ValueError(f"unexpected {category}: {token}")
    plural = bool(tags[tokens[index]] & {"v_p_trans", "v_p_intrans"})
    return tuple(prefix), int(plural)


class SimpleAgreementDataset(Dataset):
    def __init__(self, examples, vocabulary, sequence_length=6):
        self.examples = examples
        self.vocabulary = vocabulary
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        prefix, target = self.examples[index]
        # Right padding precedes a fixed final readout token; padding is masked by the model.
        ids = [self.vocabulary[word] for word in prefix]
        ids += [self.vocabulary["<pad>"]] * (self.sequence_length - 1 - len(ids))
        ids += [self.vocabulary["<cls>"]]
        return torch.tensor(ids), torch.tensor(target)


def load_agreement(directory=DEFAULT_DIRECTORY):
    directory = Path(directory)
    if not (directory / "manifest.json").exists():
        raise FileNotFoundError(
            f"Simple Agreement data missing at {directory}. Run: "
            f'python -m grokking.agreement --output-dir "{directory}"'
        )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("revision") != REVISION or manifest.get("grammar") != GRAMMAR:
        raise ValueError("unexpected Simple Agreement dataset revision or grammar")
    for filename in (*SPLITS, "tag_token_map.txt", "agreement_hr.gr"):
        expected_hash = manifest["files"][filename]["sha256"]
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"dataset checksum mismatch: {filename}")
    tags = {}
    for line in (directory / "tag_token_map.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            tag, token = line.split()
            tags.setdefault(token, set()).add(tag)
    words = sorted(word for word, categories in tags.items() if categories & {"det", "prep", "n_s", "n_p"})
    vocabulary = {word: index for index, word in enumerate(["<pad>", "<cls>", *words])}
    examples, removed, seen = {}, {}, set()
    # Full sentences can become identical when reduced to prefixes. Deduplicate and
    # remove train/validation overlap after adaptation, preserving structural test splits.
    for split in SPLITS:
        rows = [parse_sentence(line, tags) for line in (directory / split).read_text(encoding="utf-8").splitlines() if line.strip()]
        unique = {}
        for prefix, label in rows:
            if prefix in unique and unique[prefix] != label:
                raise ValueError(f"conflicting labels within {split}: {prefix}")
            unique[prefix] = label
        if split in ("train", "val"):
            selected = [(prefix, label) for prefix, label in unique.items() if prefix not in seen]
            seen.update(prefix for prefix, _ in selected)
        else:
            selected = list(unique.items())
            if any(prefix in seen for prefix, _ in selected):
                raise ValueError(f"structural test overlaps training/validation: {split}")
        if not selected:
            raise ValueError(f"empty split after prefix adaptation: {split}")
        examples[split] = selected
        removed[split] = len(rows) - len(selected)
    metadata = {"source": manifest, "vocabulary": vocabulary, "labels": ["singular", "plural"],
                "adaptation": "low-diversity grammar; prefix -> main-verb number; unique prefixes",
                "split_sizes": {key: len(value) for key, value in examples.items()},
                "removed_duplicates_or_overlap": removed,
                "g2_test_meaning": "linear nearest-noun labels; diagnostic, not grammatical correctness"}
    return {key: SimpleAgreementDataset(value, vocabulary) for key, value in examples.items()}, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    download_agreement(args.output_dir)
    _, metadata = load_agreement(args.output_dir)
    print(json.dumps(metadata["split_sizes"], indent=2))
