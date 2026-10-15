import unittest

import datasets
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers

from pyreft import ReftDataCollator, ReftRawDataset, ReftSupervisedDataset
from pyreft.dataset import prompt_no_input

SUBSPACES = [0, 1, 2]
INSTRUCTION = "please say hi to me"
OUTPUT = "hello there"


def make_tokenizer():
    # tiny deterministic word-level tokenizer; unknown words map to [UNK]
    vocab = {"[PAD]": 0, "[UNK]": 1, "</s>": 2}
    for word in (INSTRUCTION + " " + OUTPUT).split():
        vocab.setdefault(word, len(vocab))
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]",
        eos_token="</s>", model_max_length=128)


def num_interventions_for(num_layers, position, share_weights):
    # mirrors examples/loreft/train.py: layers are only duplicated for
    # combined first+last positions when weights are not shared
    if "+" in position and not share_weights:
        return 2 * num_layers
    return num_layers


CASES = [
    (position, share_weights, num_layers)
    for position in ["f1", "l1", "f1+l1", "f2+l2"]
    for share_weights in [True, False]
    for num_layers in [1, 3, 12]
]


class SubspacesPerInterventionTestCase(unittest.TestCase):
    """Regression tests for stanfordnlp/pyreft#181."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = make_tokenizer()
        cls.collator = ReftDataCollator(
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer=cls.tokenizer, model=None,
                label_pad_token_id=-100, padding="longest"))
        # last prompt position (before the leading pad token is added)
        cls.last_position = len(cls.tokenizer(prompt_no_input % INSTRUCTION)["input_ids"]) - 1

    def make_supervised_dataset(self, position, share_weights, num_interventions):
        dataset = datasets.Dataset.from_list([
            {"instruction": INSTRUCTION, "input": "", "output": OUTPUT, "subspaces": SUBSPACES},
            {"instruction": INSTRUCTION, "input": "", "output": OUTPUT + " " + OUTPUT,
             "subspaces": SUBSPACES},
        ])
        return ReftSupervisedDataset(
            "test", None, self.tokenizer, dataset=dataset,
            input_field="input", output_field="output", instruction_field="instruction",
            num_interventions=num_interventions, position=position, share_weights=share_weights)

    def make_raw_dataset(self, position, share_weights, num_interventions):
        dataset = datasets.Dataset.from_list([
            {"instruction": INSTRUCTION, "output": OUTPUT, "subspaces": SUBSPACES},
            {"instruction": INSTRUCTION, "output": OUTPUT + " " + OUTPUT, "subspaces": SUBSPACES},
        ])
        return ReftRawDataset(
            "test", None, self.tokenizer, dataset=dataset,
            num_interventions=num_interventions, position=position, share_weights=share_weights)

    def assert_one_subspace_per_intervention(self, dataset, num_interventions):
        for example in dataset:
            self.assertEqual(len(example["intervention_locations"]), num_interventions)
            self.assertEqual(example["subspaces"], [SUBSPACES] * num_interventions)

        batch = self.collator([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["subspaces"].shape), (2, num_interventions, len(SUBSPACES)))
        self.assertEqual(tuple(batch["intervention_locations"].shape[:2]), (2, num_interventions))
        # the trainers pass both tensors to pyvene as one entry per intervention
        self.assertEqual(
            len(batch["subspaces"].permute(1, 0, 2).tolist()),
            len(batch["intervention_locations"].permute(1, 0, 2).tolist()))

    def test_supervised_dataset(self):
        for position, share_weights, num_layers in CASES:
            num_interventions = num_interventions_for(num_layers, position, share_weights)
            with self.subTest(position=position, share_weights=share_weights,
                              num_interventions=num_interventions):
                dataset = self.make_supervised_dataset(position, share_weights, num_interventions)
                self.assert_one_subspace_per_intervention(dataset, num_interventions)

    def test_raw_dataset(self):
        for position, share_weights, num_layers in CASES:
            num_interventions = num_interventions_for(num_layers, position, share_weights)
            with self.subTest(position=position, share_weights=share_weights,
                              num_interventions=num_interventions):
                dataset = self.make_raw_dataset(position, share_weights, num_interventions)
                self.assert_one_subspace_per_intervention(dataset, num_interventions)

    def test_combined_positions_location_groups(self):
        # locations are shifted by one for the leading pad token
        first, last = 1, self.last_position
        num_layers = 3

        shared = self.make_supervised_dataset("f1+l1", True, num_layers)[0]
        self.assertEqual(shared["intervention_locations"], [[first, last]] * num_layers)

        # non-shared keeps the original ordering: all first-token groups,
        # then all last-token groups
        unshared = self.make_supervised_dataset("f1+l1", False, 2 * num_layers)[0]
        self.assertEqual(
            unshared["intervention_locations"], [[first]] * num_layers + [[last]] * num_layers)
        self.assertEqual(unshared["subspaces"], [SUBSPACES] * (2 * num_layers))


if __name__ == "__main__":
    unittest.main()
