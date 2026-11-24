"""Walking ranking semantics and five-fold synthetic wiring; no local real fitting."""
from pathlib import Path
import json
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import tdcs_raw_walking_rank_v30 as rank


def test_rank_pushes_Walking_above_other_FOG_and_ignores_background_and_padding():
    logits = torch.zeros(5,4,requires_grad=True)
    labels = torch.tensor([3,1,2,0,2]); valid = torch.tensor([True,True,True,True,False])
    loss, pairs = rank.rank_term(logits, labels, valid)
    loss.backward()
    assert pairs == 2 and torch.isfinite(loss)
    assert logits.grad[0,3] < 0 and (logits.grad[1:3,3] > 0).all()
    assert not logits.grad[3:].any()


def test_only_top_sixteen_other_FOG_negatives_contribute():
    logits = torch.zeros(24,4); logits[1:,3] = torch.arange(23.)
    logits.requires_grad_()
    labels = torch.tensor([3]+[2]*23); valid = torch.ones(24,dtype=torch.bool)
    loss, pairs = rank.rank_term(logits,labels,valid); loss.backward()
    assert pairs == 16 and not logits.grad[1:8].any()
    assert (logits.grad[8:,3] > 0).all()


@pytest.mark.parametrize('labels', [[0,0,3], [1,2,1], [0,0,0]])
def test_missing_class_gives_no_ranking_gradient(labels):
    logits = torch.randn(3,4,requires_grad=True)
    loss,pairs = rank.rank_term(logits,torch.tensor(labels),torch.ones(3,dtype=torch.bool))
    loss.backward(); assert pairs == 0 and loss.item() == 0 and not logits.grad.any()


def test_context_restores_frozen_training_on_error():
    before = rank.raw.train_phase
    with pytest.raises(RuntimeError):
        with rank.applied():
            assert rank.raw.source_hash() == rank.effective_source_hash()
            raise RuntimeError('synthetic fixture')
    assert rank.raw.train_phase is before
    assert rank.raw.source_hash() == rank.typed.FROZEN_RAW_SHA256 and 'S11' not in rank.raw.SPECS


def test_synthetic_folds_keep_control_fine_tune_crops(tmp_path):
    def session(folder):
        return {'suite':'sensor','cfg':rank.raw.base.smoke_config(),'output':tmp_path/folder,
            'context':{},'results':[],'runtime':{},'deadline':float('inf')}
    control = rank.typed.run(session('control'))
    tested = rank.run(session('rank'))
    assert control['status'] == tested['status'] == 'complete'
    c, t = Path(control['run_directory']), Path(tested['run_directory'])
    usage = json.loads((t/'walking_rank_usage.json').read_text())
    assert len(usage['folds']) == 5 and any(f['pairs'] > 0 for f in usage['folds'])
    for fold in range(5):
        path = f'fold_{fold}/training_trace.json'
        assert json.loads((c/path).read_text()) == json.loads((t/path).read_text())
    from validate_tdcs_raw_transfer_v28 import audit
    receipt = audit(t.parents[2],'S11',replay=True)
    assert receipt['saved_model_replay'] and not receipt['retained_heads_changed']
