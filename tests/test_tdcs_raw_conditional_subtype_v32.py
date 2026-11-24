"""The v32 loss is masked, needs no within-batch pair, and keeps the M22 inputs."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tdcs_raw_conditional_subtype_v32 as conditional


def test_conditional_loss_masks_background_invalid_and_unknown_types():
    logits = torch.zeros((5, 4), requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 255])
    mask = torch.tensor([True, True, False, True, True])
    loss = conditional.conditional_losses(logits, labels, mask)
    torch.testing.assert_close(loss, torch.tensor([0., .4054651, 0., 1.0986123, 0.]))
    loss.sum().backward()
    assert not torch.count_nonzero(logits.grad[[0, 2, 4]])
    assert not torch.count_nonzero(logits.grad[:, 0])
    assert logits.grad[1, 3] > 0 and logits.grad[3, 3] < 0


@pytest.mark.parametrize("label, direction", [(1, 1), (3, -1)])
def test_single_subtype_microbatch_has_nonzero_conditional_signal(label, direction):
    logits = torch.zeros((2, 4), requires_grad=True)
    loss = conditional.conditional_losses(logits, torch.full((2,), label), torch.ones(2, dtype=torch.bool))
    loss.sum().backward()
    assert torch.all(direction * logits.grad[:, 3] > 0)


def test_five_synthetic_folds_preserve_m22_warmup_and_crops(tmp_path):
    raw, weak = conditional.raw, conditional.weak
    outputs = {}
    for method, suite in [("M22", "metadata"), ("S13", "sensor")]:
        session = {"suite": suite, "cfg": raw.base.smoke_config(), "output": tmp_path / method,
                   "context": {}, "results": [], "runtime": {}, "deadline": float("inf")}
        outcome = weak.run(session) if method == "M22" else conditional.run(session)
        assert outcome["status"] == "complete"
        outputs[method] = Path(outcome["run_directory"])
    new, control = outputs["S13"], outputs["M22"]
    for path in ["auxiliary_pretraining_trace.json", *[f"fold_{i}/training_trace.json" for i in range(5)]]:
        assert (new / path).read_bytes() == (control / path).read_bytes()
    with torch.no_grad():
        original = torch.load(control / "defog_pretrained.pt", weights_only=True, map_location="cpu")
        candidate = torch.load(new / "defog_pretrained.pt", weights_only=True, map_location="cpu")
        assert all(torch.equal(p, candidate["model"][name]) for name, p in original["model"].items())
    receipt = json.loads((new / "conditional_usage.json").read_text())
    assert len(receipt["folds"]) == 5 and receipt["source_sha256"] == conditional.effective_source_hash()
    assert all(f["micros_with_any_FOG"] > 0 for f in receipt["folds"])
    from validate_tdcs_conditional_subtype_v32 import audit_matched
    records, _, _ = weak.synthetic()
    assert audit_matched(new, control, records, raw.smoke_config()) == receipt
    assert conditional.METHOD not in raw.SPECS and raw.train_phase is weak.ORIGINAL_TRAIN
