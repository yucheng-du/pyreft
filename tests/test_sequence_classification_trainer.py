"""Offline classification Trainer regressions with a real tiny ReftModel."""

import math

import pytest
import torch
import torch.nn.functional as F
from transformers import (
    GPT2Config,
    GPT2ForSequenceClassification,
    TrainingArguments,
    default_data_collator,
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from pyreft import (
    LoreftIntervention,
    ReftConfig,
    ReftDataCollator,
    ReftTrainerForSequenceClassification,
    get_reft_model,
)


def make_trainer(tmp_path, loss_kind="ce", gradient_accumulation_steps=1):
    torch.manual_seed(42)
    config = GPT2Config(
        vocab_size=32,
        n_positions=8,
        n_embd=16,
        n_layer=1,
        n_head=2,
        num_labels=1 if loss_kind == "mse" else 2,
        pad_token_id=0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = GPT2ForSequenceClassification(config)
    reft_config = ReftConfig(representations={
        "layer": 0,
        "component": "block_output",
        "low_rank_dimension": 2,
        "intervention": LoreftIntervention(
            embed_dim=16, low_rank_dimension=2, dtype=torch.float32,
        ),
    })
    model = get_reft_model(model, reft_config)
    assert not any(param.requires_grad for param in model.model.parameters())

    labels = {
        "ce": torch.tensor([0, 1], dtype=torch.long),
        "bce": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "mse": torch.tensor([0.1, -0.3], dtype=torch.float32),
    }[loss_kind]
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        # GPT2 pools the last non-pad token. With one layer, intervening at
        # position 0 after that layer would not affect the classification head.
        "intervention_locations": torch.tensor([[[3]], [[3]]]),
    }
    examples = [
        {key: value[index] for key, value in inputs.items()}
        for index in range(len(input_ids))
    ]
    trainer = ReftTrainerForSequenceClassification(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to=[],
            per_device_train_batch_size=2,
            gradient_accumulation_steps=gradient_accumulation_steps,
            remove_unused_columns=False,
            label_names=["labels"],
            disable_tqdm=True,
            max_steps=1,
            learning_rate=1e-3,
            optim="adamw_torch",
            max_grad_norm=0.0,
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="no",
        ),
        train_dataset=examples,
        data_collator=ReftDataCollator(data_collator=default_data_collator),
    )
    model.eval()
    return trainer, inputs


def trainable_parameters(model):
    return {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def nonzero_finite_gradients(model):
    gradients = {
        name: param.grad.detach().clone()
        for name, param in trainable_parameters(model).items()
        if param.grad is not None
    }
    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients.values())
    assert any(grad.abs().max() > 1e-5 for grad in gradients.values())
    return gradients


@pytest.mark.parametrize("with_num_items", [False, True], ids=["none", "nonempty"])
def test_training_step_accepts_loss_kwargs(tmp_path, with_num_items):
    trainer, inputs = make_trainer(tmp_path)
    trainer.current_gradient_accumulation_steps = 1
    assert trainer.model_accepts_loss_kwargs is False
    num_items = torch.tensor(2) if with_num_items else None

    loss = trainer.training_step(
        trainer.model, inputs, num_items_in_batch=num_items,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert not loss.requires_grad
    nonzero_finite_gradients(trainer.model)


@pytest.mark.parametrize("loss_kind", ["ce", "bce", "mse"])
def test_compute_loss_preserves_existing_loss_semantics(tmp_path, loss_kind):
    trainer, inputs = make_trainer(tmp_path, loss_kind=loss_kind)
    # MSE keeps the existing bfloat16 target cast. Its mixed-dtype CPU backward
    # is outside this signature regression; these are forward-value checks.
    with torch.no_grad():
        # Run the old, keyword-free API and its independent oracle first so the
        # baseline distinguishes working loss branches from the new call failure.
        loss, outputs = trainer.compute_loss(
            trainer.model, inputs, return_outputs=True,
        )
        assert isinstance(outputs, SequenceClassifierOutputWithPast)
        assert outputs.logits.shape == (2, 1 if loss_kind == "mse" else 2)
        assert outputs.logits.dtype == torch.float32
        if loss_kind == "ce":
            expected = F.cross_entropy(outputs.logits, inputs["labels"])
        elif loss_kind == "bce":
            expected = F.binary_cross_entropy_with_logits(
                outputs.logits, inputs["labels"],
            )
        else:
            rounded_labels = inputs["labels"].to(torch.bfloat16).to(outputs.logits.dtype)
            expected = (outputs.logits.squeeze() - rounded_labels.squeeze()).square().mean()
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(trainer.compute_loss(trainer.model, inputs), expected)

        for num_items in (None, torch.tensor(2)):
            keyword_loss, keyword_outputs = trainer.compute_loss(
                trainer.model, inputs, return_outputs=True,
                num_items_in_batch=num_items,
            )
            torch.testing.assert_close(keyword_loss, expected)
            torch.testing.assert_close(keyword_outputs.logits, outputs.logits)


def test_training_step_preserves_accumulated_gradients(tmp_path):
    trainer, inputs = make_trainer(tmp_path, gradient_accumulation_steps=2)
    trainer.model.train()
    assert trainer.model_accepts_loss_kwargs is False
    assert trainer.accelerator.gradient_accumulation_steps == 1

    # Independent of training_step's scaling: the full batch uses a CE mean.
    full_loss = trainer.compute_loss(trainer.model, inputs)
    full_loss.backward()
    expected_gradients = nonzero_finite_gradients(trainer.model)
    for param in trainable_parameters(trainer.model).values():
        param.grad = None

    trainer.current_gradient_accumulation_steps = 2
    losses = []
    for index in range(2):
        microbatch = {key: value[index:index + 1] for key, value in inputs.items()}
        losses.append(trainer.training_step(
            trainer.model, microbatch, num_items_in_batch=torch.tensor(2),
        ))

    actual_gradients = nonzero_finite_gradients(trainer.model)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name, expected in expected_gradients.items():
        torch.testing.assert_close(
            actual_gradients[name], expected, atol=1e-6, rtol=1e-4,
        )
    torch.testing.assert_close(torch.stack(losses).sum(), full_loss.detach())


def test_train_runs_one_step_with_real_collator(tmp_path):
    trainer, _ = make_trainer(tmp_path)
    before = {
        name: param.detach().clone()
        for name, param in trainable_parameters(trainer.model).items()
    }
    assert before

    result = trainer.train()

    assert result.global_step == 1
    assert math.isfinite(result.training_loss)
    assert any(
        not torch.equal(param.detach(), before[name])
        for name, param in trainable_parameters(trainer.model).items()
    )
