"""Offline public evaluation regressions with real GPT-2 and LoReFT modules."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2Config,
    GPT2ForSequenceClassification,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from pyreft import (
    LoreftIntervention,
    ReftConfig,
    ReftDataCollator,
    ReftTrainerForSequenceClassification,
    get_reft_model,
)


class EvaluationRecorder(TrainerCallback):
    def __init__(self):
        self.predictions = []
        self.evaluations = []
        self.logs = []

    def compute_metrics(self, prediction):
        logits = torch.as_tensor(prediction.predictions)
        labels = torch.as_tensor(prediction.label_ids)
        self.predictions.append((logits.clone(), labels.clone()))
        return {
            "accuracy": np.float64((logits.argmax(-1) == labels).float().mean()),
            "eval_count": np.int64(len(labels)),
        }

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.evaluations.append(dict(metrics))
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.logs.append(dict(logs))


def make_trainer(tmp_path, dropout):
    torch.manual_seed(42)
    config = GPT2Config(
        vocab_size=32, n_positions=8, n_embd=16, n_layer=2, n_head=2,
        num_labels=2, pad_token_id=0, use_cache=False,
        resid_pdrop=0.2, embd_pdrop=0.2, attn_pdrop=0.2,
    )
    config._attn_implementation = "eager"
    model = get_reft_model(
        GPT2ForSequenceClassification(config),
        ReftConfig(representations=[{
            "layer": layer,
            "component": "block_output",
            "low_rank_dimension": 2,
            "intervention": LoreftIntervention(
                embed_dim=16, low_rank_dimension=2,
                dtype=torch.float32, dropout=dropout,
            ),
        } for layer in range(2)]),
    )
    examples = [{
        "input_ids": ids,
        "attention_mask": [1, 1, 1, 1],
        "intervention_locations": [[3], [3]],
        "labels": label,
    } for ids, label in [
        ([1, 2, 3, 4], 0), ([5, 6, 7, 8], 1), ([9, 10, 11, 12], 0),
    ]]
    recorder = EvaluationRecorder()
    trainer = ReftTrainerForSequenceClassification(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to=[],
            per_device_eval_batch_size=2, remove_unused_columns=False,
            label_names=["labels"], disable_tqdm=True,
        ),
        eval_dataset=examples,
        data_collator=ReftDataCollator(default_data_collator),
        compute_metrics=recorder.compute_metrics,
        callbacks=[recorder],
    )
    assert len(model.interventions) == 2
    assert all(isinstance(module, LoreftIntervention)
               for module in model.interventions.values())
    assert not any(p.requires_grad for p in model.model.parameters())
    assert any(p.requires_grad for p in model.interventions.parameters())
    return trainer, recorder


def set_component_mode(model, training):
    # Explicit component control also works before PyVene propagates wrapper mode.
    model.model.train(training)
    for intervention in model.interventions.values():
        intervention.train(training)


def manual_eval_forward(trainer):
    set_component_mode(trainer.model, False)
    predictions, labels, batch_sizes = [], [], []
    loader = DataLoader(
        trainer.eval_dataset, batch_size=2, shuffle=False,
        collate_fn=trainer.data_collator,
    )
    with torch.no_grad():
        for batch in loader:
            _, output = trainer.model(
                {key: batch[key] for key in ("input_ids", "attention_mask")},
                unit_locations={"sources->base": (
                    None, batch["intervention_locations"].permute(1, 0, 2).tolist(),
                )},
            )
            predictions.append(output.logits)
            labels.append(batch["labels"])
            batch_sizes.append(len(batch["labels"]))
    assert batch_sizes == [2, 1]
    return torch.cat(predictions), torch.cat(labels).float()


def parameter_snapshot(model):
    return {
        name: (id(param), param.detach().clone(), param.requires_grad,
               None if param.grad is None else param.grad.detach().clone())
        for name, param in model.named_parameters()
    }


def assert_parameters_unchanged(model, before):
    parameters = dict(model.named_parameters())
    assert parameters.keys() == before.keys()
    for name, (identity, value, requires_grad, grad) in before.items():
        param = parameters[name]
        assert id(param) == identity
        assert param.requires_grad == requires_grad
        torch.testing.assert_close(param, value, rtol=0, atol=0)
        if grad is None:
            assert param.grad is None
        else:
            torch.testing.assert_close(param.grad, grad, rtol=0, atol=0)


@pytest.mark.parametrize("dropout", [0.0, 1.0], ids=["no-dropout", "dropout-one"])
def test_manual_eval_forward_control(tmp_path, dropout):
    trainer, _ = make_trainer(tmp_path, dropout)
    before = parameter_snapshot(trainer.model)
    logits, labels = manual_eval_forward(trainer)
    assert logits.shape == (3, 2)
    assert labels.tolist() == [0.0, 1.0, 0.0]
    assert torch.isfinite(logits).all()
    assert logits.abs().max() > 1e-5
    assert not logits.requires_grad
    assert_parameters_unchanged(trainer.model, before)


@pytest.mark.parametrize("dropout", [0.0, 1.0], ids=["no-dropout", "dropout-one"])
def test_public_evaluate_uses_intervention_modules(tmp_path, dropout):
    trainer, recorder = make_trainer(tmp_path, dropout)
    model = trainer.model
    # Pre-existing gradients must also survive evaluation without being cleared.
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.full_like(param, 0.25)
    before = parameter_snapshot(model)
    expected_logits, expected_labels = manual_eval_forward(trainer)
    expected_metrics = {
        "eval_accuracy": float(
            (expected_logits.argmax(-1) == expected_labels).float().mean()
        ),
        "eval_count": 3,
    }

    observed = []

    def observe_forward(module, args, kwargs):
        observed.append((id(module), args[0].shape[0]))
        assert not torch.is_grad_enabled()
        assert all(not child.training for child in model.model.modules())
        assert all(not child.training for child in module.modules())

    # PyVene clears hooks on the base model before forwarding. Intervention
    # hooks survive and observe the real computation, including base mode.
    components = list(model.interventions.values())
    handles = [module.register_forward_pre_hook(observe_forward, with_kwargs=True)
               for module in components]
    try:
        for round_index in range(2):
            set_component_mode(model, True)
            observed.clear()
            metrics = trainer.evaluate(ignore_keys=None)

            assert metrics == expected_metrics
            assert type(metrics["eval_accuracy"]) is float
            assert type(metrics["eval_count"]) is int
            assert len(recorder.predictions) == round_index + 1
            actual_logits, actual_labels = recorder.predictions[-1]
            torch.testing.assert_close(actual_logits, expected_logits)
            torch.testing.assert_close(actual_labels, expected_labels)
            assert recorder.evaluations == [expected_metrics] * (round_index + 1)
            assert len(recorder.logs) == round_index + 1
            for key, value in expected_metrics.items():
                assert recorder.logs[-1][key] == value
                assert trainer.state.log_history[-1][key] == value
            for module in components:
                sizes = [size for identity, size in observed if identity == id(module)]
                assert sizes == [2, 1]
            assert_parameters_unchanged(model, before)
    finally:
        for handle in handles:
            handle.remove()
