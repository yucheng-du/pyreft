"""Offline Trainer regressions using a randomly initialized tiny language model."""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments

from pyreft import (
    LoreftIntervention,
    ReftConfig,
    ReftTrainerForCausalLM,
    get_reft_model,
)


class BaseOnlyModel(torch.nn.Module):
    """Exercise the base-output fallback with the intervenable call interface."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, base, unit_locations=None, labels=None, subspaces=None):
        return self.model(**base, labels=labels), None


@pytest.fixture(params=["intervened", "base_only"])
def trainer_and_inputs(request, tmp_path):
    torch.manual_seed(42)
    config = GPT2Config(
        vocab_size=32, n_positions=8, n_embd=16, n_layer=1, n_head=2,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False,
    )
    config._attn_implementation = "eager"
    model = GPT2LMHeadModel(config)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    labels = input_ids.clone()
    labels[:, 0] = -100
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }
    if request.param == "intervened":
        reft_config = ReftConfig(representations={
            "layer": 0,
            "component": "block_output",
            "low_rank_dimension": 2,
            "intervention": LoreftIntervention(
                embed_dim=16, low_rank_dimension=2, dtype=torch.float32,
            ),
        })
        model = get_reft_model(model, reft_config)
        inputs["intervention_locations"] = torch.tensor([[[0]], [[0]]])
    else:
        model = BaseOnlyModel(model)

    trainer = ReftTrainerForCausalLM(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to=[],
            per_device_eval_batch_size=2, gradient_accumulation_steps=2,
            remove_unused_columns=False, label_names=["labels"],
            disable_tqdm=True,
        ),
    )
    model.eval()
    return trainer, inputs


def test_compute_loss_returns_scalar(trainer_and_inputs):
    trainer, inputs = trainer_and_inputs
    loss = trainer.compute_loss(trainer.model, inputs)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_compute_loss_returns_loss_and_outputs(trainer_and_inputs):
    trainer, inputs = trainer_and_inputs
    loss, outputs = trainer.compute_loss(trainer.model, inputs, return_outputs=True)
    assert loss is outputs.loss
    assert loss.ndim == 0
    torch.testing.assert_close(loss, trainer.compute_loss(trainer.model, inputs))


@pytest.mark.parametrize("prediction_loss_only", [False, True])
def test_prediction_step(trainer_and_inputs, prediction_loss_only):
    trainer, inputs = trainer_and_inputs
    with torch.no_grad():
        expected_loss = trainer.compute_loss(trainer.model, inputs)
    loss, logits, labels = trainer.prediction_step(
        trainer.model, inputs, prediction_loss_only=prediction_loss_only,
    )
    torch.testing.assert_close(loss, expected_loss)
    assert not loss.requires_grad
    if prediction_loss_only:
        assert logits is None
        assert labels is None
    else:
        assert logits.shape == (2, 4, 32)
        assert not logits.requires_grad
        torch.testing.assert_close(labels, inputs["labels"])


def test_evaluate_reports_loss(trainer_and_inputs):
    trainer, inputs = trainer_and_inputs
    examples = [{key: value[i] for key, value in inputs.items()} for i in range(2)]
    with torch.no_grad():
        expected_loss = trainer.compute_loss(trainer.model, inputs).item()
    metrics = trainer.evaluate(eval_dataset=examples)
    assert metrics["eval_loss"] == pytest.approx(expected_loss)


def test_training_step_preserves_accumulated_gradients(trainer_and_inputs):
    trainer, inputs = trainer_and_inputs
    trainer.model.train()
    # The loss is a per-microbatch mean; Trainer must apply accumulation scaling.
    assert not trainer.model_accepts_loss_kwargs
    trainer.compute_loss(trainer.model, inputs).backward()
    expected_grads = {
        name: param.grad.clone()
        for name, param in trainer.model.named_parameters()
        if param.grad is not None
    }
    assert expected_grads
    assert any(grad.abs().max() > 1e-5 for grad in expected_grads.values())
    for param in trainer.model.parameters():
        param.grad = None
    trainer.current_gradient_accumulation_steps = 2
    num_items = inputs["labels"].ne(-100).sum()
    for i in range(2):
        microbatch = {key: value[i:i + 1] for key, value in inputs.items()}
        trainer.training_step(trainer.model, microbatch, num_items_in_batch=num_items)
    for name, param in trainer.model.named_parameters():
        if name in expected_grads:
            torch.testing.assert_close(param.grad, expected_grads[name], atol=1e-6, rtol=1e-4)
