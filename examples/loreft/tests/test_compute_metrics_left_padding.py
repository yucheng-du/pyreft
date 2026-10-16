"""Offline regression tests for intervention coordinates in LoReFT evaluation.

Generation tasks are evaluated with a left-padding collator, so every
intervention location must be shifted by exactly the padding the collator
inserted in front of that example (stanfordnlp/pyreft#110). The examples are
built with the real ``ReftDataset`` machinery (sink pad + ``+1`` shift), batched
by the real ``DataLoader``/``ReftDataCollator``, and passed through the real
``compute_metrics``. Only the generation boundary is replaced by a recorder
that returns a fixed continuation.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

LOREFT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = LOREFT_DIR.parents[1]
for path in (str(REPO_ROOT), str(LOREFT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import pytest
import torch
from datasets import ClassLabel, Dataset, Features, Value
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast

import compute_metrics as metrics_module
from dataset import LoReftGLUEDataset, LoReftSupervisedDataset
from pyreft import ReftDataCollator, ReftDataset

VOCAB = [
    "<pad>", "<unk>", "<s>", "</s>",
    "alpha", "beta", "gamma", "delta", "epsilon",
    "the", "correct", "answer", "is", "true", "false",
    "###", "Response:", "ok",
]


def make_tokenizer(add_bos=True, pad_token="<pad>"):
    vocab = {token: i for i, token in enumerate(VOCAB)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if add_bos:
        backend.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", vocab["<s>"])])
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>", eos_token="</s>", unk_token="<unk>",
        pad_token=pad_token,
        model_max_length=64, padding_side="right",
    )


class RecordingIntervenable:
    """Generation boundary: records what compute_metrics hands to the model."""

    model = None

    def __init__(self, tokenizer, continuation):
        self.continuation = tokenizer(continuation, add_special_tokens=False,
                                      return_tensors="pt")["input_ids"][0]
        self.calls = []

    def generate(self, base, unit_locations, **kwargs):
        self.calls.append({"base": base, "unit_locations": unit_locations, "kwargs": kwargs})
        input_ids = base["input_ids"]
        tail = self.continuation.unsqueeze(0).expand(input_ids.shape[0], -1)
        return None, torch.cat([input_ids, tail], dim=1)


class CustomLocationDataset(ReftDataset):
    """Prompts tokenized as-is with per-intervention custom positions.

    Positions are given in prompt coordinates; ReftDataset prepends the sink
    pad and shifts them by +1 as in production.
    """

    def tokenize(self, data_item):
        input_ids = self.tokenizer(data_item["instruction"], return_tensors="pt")["input_ids"][0]
        return {"input_ids": input_ids}, len(input_ids)

    def get_intervention_locations(self, last_position, **kwargs):
        # intervention 0: first and second-to-last prompt token
        # intervention 1: second and last prompt token
        return [[0, last_position - 2], [1, last_position - 1]]


class LoraOnlyDataset(CustomLocationDataset):
    def get_intervention_locations(self, last_position, **kwargs):
        return [0]  # one flat location per example -> 2D batch -> no ReFT locations


def tokens_at(tokenizer, input_ids, locations):
    return [tokenizer.convert_ids_to_tokens(input_ids[loc].item()) for loc in locations]


# Two examples, prompt lengths 4 and 5, i.e. 5 and 6 tokens including the sink.
# The batch is 6 wide, so the collator prepends 1 pad to the first example and
# none to the second. Dataset coordinates (sink at 0):
#   example 0: intervention 0 -> [1, 3], intervention 1 -> [2, 4]
#   example 1: intervention 0 -> [1, 4], intervention 1 -> [2, 5]
# After left padding example 0 moves by 1 and example 1 stays put.
ORACLE_LOCATIONS = [[[2, 4], [1, 4]], [[3, 5], [2, 5]]]

ORACLE_CASES = {
    # name: (tokenizer kwargs, prompts, expected [example][intervention] tokens)
    "single_bos": (
        {}, ["alpha beta gamma", "alpha beta gamma delta"],
        [[["<s>", "beta"], ["alpha", "gamma"]],
         [["<s>", "gamma"], ["alpha", "delta"]]],
    ),
    # same lengths and masks, extra BOS ids inside the text
    "repeated_bos": (
        {}, ["alpha <s> gamma", "<s> beta <s> delta"],
        [[["<s>", "<s>"], ["alpha", "gamma"]],
         [["<s>", "<s>"], ["<s>", "delta"]]],
    ),
    "no_bos": (
        {"add_bos": False}, ["alpha beta gamma delta", "alpha beta gamma delta epsilon"],
        [[["alpha", "gamma"], ["beta", "delta"]],
         [["alpha", "delta"], ["beta", "epsilon"]]],
    ),
    # BOS shares the pad id, so BOS positions are also masked out
    "pad_is_bos": (
        {"pad_token": "<s>"}, ["alpha beta gamma", "alpha beta gamma delta"],
        [[["<s>", "beta"], ["alpha", "gamma"]],
         [["<s>", "gamma"], ["alpha", "delta"]]],
    ),
    # pad = unk as in train.py; the unknown word "zeta" is masked mid-prompt
    "pad_is_unk": (
        {"pad_token": "<unk>"}, ["alpha zeta gamma", "alpha beta zeta delta"],
        [[["<s>", "<unk>"], ["alpha", "gamma"]],
         [["<s>", "<unk>"], ["alpha", "delta"]]],
    ),
}


@pytest.mark.parametrize("case", list(ORACLE_CASES))
def test_left_padding_offsets_follow_collator_not_bos(case):
    tokenizer_kwargs, prompts, expected_tokens = ORACLE_CASES[case]
    tokenizer = make_tokenizer(**tokenizer_kwargs)
    raw = Dataset.from_list([{"instruction": p} for p in prompts])
    eval_dataset = CustomLocationDataset(
        "alpaca", "offline", tokenizer, data_split="test", dataset=raw, position="f1+l1")
    assert [len(item["input_ids"]) for item in eval_dataset] == [5, 6]
    intervenable = RecordingIntervenable(tokenizer, "### Response: ok")

    generations, stats = metrics_module.compute_metrics(
        "alpaca", "offline_eval", intervenable, tokenizer, eval_dataset,
        eval_dataset.raw_dataset, "### Response:", "run", batch_size=2)

    assert len(intervenable.calls) == 1
    call = intervenable.calls[0]
    locations = call["unit_locations"]["sources->base"][1]
    assert call["unit_locations"]["sources->base"][0] is None
    assert locations == ORACLE_LOCATIONS
    input_ids = call["base"]["input_ids"]
    for example in range(2):
        for intervention in range(2):
            assert tokens_at(tokenizer, input_ids[example],
                             locations[intervention][example]) == expected_tokens[example][intervention]
    assert stats == {}
    assert [g["instruction"] for g in generations] == prompts
    assert [g["output"] for g in generations] == ["ok", "ok"]


def make_commonsense_eval(tokenizer):
    raw = Dataset.from_list([
        {"instruction": "alpha beta gamma", "answer": "true"},
        {"instruction": "alpha <s> beta <s> gamma delta", "answer": "false"},
        {"instruction": "delta epsilon gamma beta alpha", "answer": "true"},
    ])
    return LoReftSupervisedDataset(
        "commonsense", "offline", tokenizer, data_split="test", dataset=raw,
        num_interventions=2, position="f2+l2", share_weights=False)


@pytest.mark.parametrize("greedy_decoding,num_beams", [(True, 1), (False, 4)])
def test_production_dataset_locations_with_beams(greedy_decoding, num_beams):
    tokenizer = make_tokenizer()
    eval_dataset = make_commonsense_eval(tokenizer)
    # prompt tokens: [<s> alpha beta gamma], [<s> alpha <s> beta <s> gamma delta],
    # [<s> delta epsilon gamma beta alpha]; plus one sink each.
    assert [len(item["input_ids"]) for item in eval_dataset] == [5, 8, 7]
    intervenable = RecordingIntervenable(tokenizer, "the correct answer is true")

    generations, stats = metrics_module.compute_metrics(
        "commonsense", "offline_eval", intervenable, tokenizer, eval_dataset,
        eval_dataset.raw_dataset, eval_dataset.trigger_tokens, "run", batch_size=2,
        greedy_decoding=greedy_decoding)

    # f2+l2 -> intervention 0 on the first two prompt tokens, intervention 1
    # on the last two. Batch 1 is 8 wide: example 0 gets 3 pads, example 1
    # none; batch 2 holds example 2 alone with no padding.
    per_example = [
        [[[4, 5], [1, 2]], [[6, 7], [6, 7]]],
        [[[1, 2]], [[5, 6]]],
    ]
    expected_tokens = [
        [[["<s>", "alpha"], ["<s>", "alpha"]], [["beta", "gamma"], ["gamma", "delta"]]],
        [[["<s>", "delta"]], [["beta", "alpha"]]],
    ]
    assert len(intervenable.calls) == 2
    for call, example_locations, example_tokens in zip(
            intervenable.calls, per_example, expected_tokens):
        assert call["kwargs"].get("num_beams", 1) == num_beams
        expected = [[loc for loc in intervention for _ in range(num_beams)]
                    for intervention in example_locations]
        locations = call["unit_locations"]["sources->base"][1]
        assert locations == expected
        # every beam row r belongs to batch example r // num_beams
        input_ids = call["base"]["input_ids"]
        for intervention, rows in enumerate(locations):
            for row, row_locations in enumerate(rows):
                example = row // num_beams
                assert tokens_at(tokenizer, input_ids[example], row_locations) == \
                    example_tokens[intervention][example]

    assert [g["generation"] for g in generations] == ["true", "true", "true"]
    assert stats == {"eval/offline_eval": pytest.approx(2 / 3)}


def test_lora_only_batches_keep_dummy_locations():
    tokenizer = make_tokenizer()
    raw = Dataset.from_list([{"instruction": "alpha beta gamma"},
                             {"instruction": "alpha beta gamma delta"}])
    eval_dataset = LoraOnlyDataset(
        "alpaca", "offline", tokenizer, data_split="test", dataset=raw, position="f1+l1")
    intervenable = RecordingIntervenable(tokenizer, "### Response: ok")

    generations, _ = metrics_module.compute_metrics(
        "alpaca", "offline_eval", intervenable, tokenizer, eval_dataset,
        eval_dataset.raw_dataset, "### Response:", "run", batch_size=2)

    assert intervenable.calls[0]["unit_locations"] == {"sources->base": (None, 0)}
    assert [g["output"] for g in generations] == ["ok", "ok"]


class RecordingClassifier:
    model = None

    def __init__(self):
        self.unit_locations = []

    def __call__(self, base, unit_locations):
        self.unit_locations.append(unit_locations)
        logits = torch.zeros(base["input_ids"].shape[0], 2)
        logits[:, 1] = 1.0
        return None, type("Output", (), {"logits": logits})()


class LocalAccuracy:
    def compute(self, predictions, references):
        return {"accuracy": sum(p == r for p, r in zip(predictions, references)) / len(references)}


def test_glue_locations_are_not_shifted(monkeypatch):
    monkeypatch.setattr(metrics_module.evaluate, "load", lambda *args, **kwargs: LocalAccuracy())
    tokenizer = make_tokenizer()
    raw = Dataset.from_list(
        [{"sentence": "alpha beta", "label": 1}, {"sentence": "alpha beta gamma delta", "label": 0}],
        features=Features({"sentence": Value("string"), "label": ClassLabel(names=["neg", "pos"])}))
    eval_dataset = LoReftGLUEDataset(
        "glue", "sst2", tokenizer, data_split="validation", dataset=raw,
        num_interventions=2, position="f1+l1", share_weights=False)
    collator = ReftDataCollator(data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"))
    classifier = RecordingClassifier()

    generations, report = metrics_module.compute_metrics(
        "glue", "sst2", classifier, tokenizer, eval_dataset, raw, None, "run",
        batch_size=2, data_collator=collator, split="validation")

    # right padding, sink appended at the end: first and last prompt token
    assert classifier.unit_locations == [{"sources->base": (None, [[[0], [0]], [[2], [4]]])}]
    assert tokenizer.padding_side == "right"
    assert generations == []
    assert report == {"validation_accuracy": 0.5}
