"""MSE regression loss of ReftTrainerForSequenceClassification with a real tiny ReftModel."""

import pytest
import torch
from transformers import GPT2Config, GPT2ForSequenceClassification, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from pyreft import (
    LoreftIntervention,
    ReftConfig,
    ReftTrainerForSequenceClassification,
    get_reft_model,
)


REGRESSION_CASES = {
    # num_labels == 1 is inferred as regression.
    "scalar": {"num_labels": 1, "problem_type": None, "labels": [0.1, -0.3]},
    # Float labels with num_labels > 1 would otherwise be inferred as
    # multi-label classification, so regression has to be explicit.
    "vector": {
        "num_labels": 2,
        "problem_type": "regression",
        "labels": [[0.1, -0.3], [0.2, 0.4]],
    },
}


def make_trainer(tmp_path, num_labels, problem_type):
    torch.manual_seed(42)
    config = GPT2Config(
        vocab_size=32,
        n_positions=8,
        n_embd=16,
        n_layer=1,
        n_head=2,
        num_labels=num_labels,
        problem_type=problem_type,
        pad_token_id=0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        attn_implementation="eager",
    )
    model = GPT2ForSequenceClassification(config)
    reft_config = ReftConfig(representations={
        "layer": 0,
        "component": "block_output",
        "low_rank_dimension": 2,
        "intervention": LoreftIntervention(
            embed_dim=16, low_rank_dimension=2, dtype=torch.float32, dropout=0.0,
        ),
    })
    reft_model = get_reft_model(model, reft_config)
    assert not any(param.requires_grad for param in reft_model.model.parameters())

    trainer = ReftTrainerForSequenceClassification(
        model=reft_model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to=[],
            remove_unused_columns=False,
            label_names=["labels"],
            eval_strategy="no",
            save_strategy="no",
            disable_tqdm=True,
        ),
    )
    reft_model.eval()
    return trainer


def make_inputs(labels):
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": torch.tensor(labels, dtype=torch.float32),
        # GPT2 pools the last non-pad token. With one layer, intervening at
        # position 0 after that layer would not affect the classification head.
        "intervention_locations": torch.tensor([[[3]], [[3]]], dtype=torch.long),
    }


@pytest.mark.parametrize("case", ["scalar", "vector"])
def test_regression_loss_backward_with_float32_labels(tmp_path, case):
    spec = REGRESSION_CASES[case]
    trainer = make_trainer(tmp_path, spec["num_labels"], spec["problem_type"])
    inputs = make_inputs(spec["labels"])
    labels = inputs["labels"]
    trainable = {
        name: param
        for name, param in trainer.model.named_parameters()
        if param.requires_grad
    }
    assert trainable
    assert not any(name.startswith("model.") for name in trainable)
    assert all(param.grad is None for param in trainable.values())

    loss, outputs = trainer.compute_loss(trainer.model, inputs, return_outputs=True)

    assert isinstance(outputs, SequenceClassifierOutputWithPast)
    logits = outputs.logits
    assert logits.shape == (2, spec["num_labels"])
    assert logits.device.type == "cpu" and logits.dtype == torch.float32
    assert logits.requires_grad and loss.requires_grad
    assert loss.dim() == 0 and torch.isfinite(loss)

    # Independent oracle: differentiable logits against the original float32 labels.
    if spec["num_labels"] == 1:
        reference = (logits.squeeze() - labels.squeeze()).square().mean()
    else:
        reference = (logits - labels).square().mean()
    print(
        f"[{case}] loss={loss.item():.9g} reference={reference.item():.9g} "
        f"labels={labels.tolist()} labels_dtype={labels.dtype}"
    )

    names = sorted(trainable)
    reference_grads = dict(zip(names, torch.autograd.grad(
        reference,
        [trainable[name] for name in names],
        retain_graph=True,
        allow_unused=True,
    )))
    used = {name: grad for name, grad in reference_grads.items() if grad is not None}
    print(f"[{case}] trainable={names} unused={sorted(set(names) - set(used))}")
    assert used
    assert all(torch.isfinite(grad).all() for grad in used.values())
    assert any(grad.abs().max() > 1e-5 for grad in used.values())
    assert all(param.grad is None for param in trainable.values())

    loss.backward()

    actual_grads = {name: param.grad for name, param in trainable.items()}
    assert {name: grad is not None for name, grad in actual_grads.items()} == {
        name: grad is not None for name, grad in reference_grads.items()
    }
    for name, grad in used.items():
        assert torch.isfinite(actual_grads[name]).all()
        torch.testing.assert_close(actual_grads[name], grad, atol=1e-6, rtol=1e-4)
    assert any(actual_grads[name].abs().max() > 1e-5 for name in used)
    assert all(param.grad is None for param in trainer.model.model.parameters())
    torch.testing.assert_close(loss.detach(), reference.detach(), atol=1e-7, rtol=1e-6)

    fresh_loss = trainer.compute_loss(trainer.model, inputs)
    assert torch.is_tensor(fresh_loss) and fresh_loss.dim() == 0
    assert torch.isfinite(fresh_loss)
    torch.testing.assert_close(fresh_loss.detach(), loss.detach(), atol=1e-7, rtol=1e-6)
