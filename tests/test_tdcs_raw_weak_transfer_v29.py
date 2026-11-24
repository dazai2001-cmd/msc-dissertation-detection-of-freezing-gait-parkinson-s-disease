"""Partial-label semantics and synthetic-only five-fold training/replay."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import tdcs_raw_weak_transfer_v29 as weak
import validate_tdcs_raw_transfer_v28 as validator


def test_binary_Event_records_never_receive_a_FOG_subtype():
    frame=pd.DataFrame({'Time':np.arange(100),'AccV':-1.,'AccML':0.,'AccAP':0.,
        'Event':np.arange(100)%2,'Valid':True,'Task':True})
    record=weak.make_weak_record(frame,'weak','independent')
    # Even downsample positions are negative here; a contiguous positive block
    # makes the positive unknown-state check independent of aliasing parity.
    frame.loc[10:30,'Event']=1
    record=weak.make_weak_record(frame,'weak','independent')
    assert set(np.unique(record['classes']))=={0,255}
    assert not np.isin(record['classes'],[1,2,3]).any()
    inputs=weak.raw.features(record,np.arange(20)); record['classes'][:]=255
    np.testing.assert_array_equal(inputs,weak.raw.features(record,np.arange(20)))
    assert not any(c in frame.columns for c in weak.raw.LABELS)


def test_unknown_positive_treats_all_FOG_subtypes_symmetrically():
    logits=torch.zeros(1,1,4,requires_grad=True); labels=torch.tensor([[255]])
    value,mass=weak.partial_losses(logits,labels,torch.ones(1,1),torch.ones(4),1.)
    (value/mass).backward(); gradient=logits.grad[0,0]
    assert gradient[0]>0 and (gradient[1:]<0).all()
    torch.testing.assert_close(gradient[1:],gradient[1].expand(3))
    torch.testing.assert_close(value,-torch.log(torch.tensor(.75)))


def test_known_labels_match_weighted_cross_entropy_exactly():
    logits=torch.randn(2,6,4); labels=torch.arange(12).reshape(2,6)%4
    weight=torch.tensor([.5,1.,2.,3.]); mask=torch.ones(2,6)
    value,mass=weak.partial_losses(logits,labels,mask,weight,2.)
    expected=F.cross_entropy(logits.transpose(1,2),labels,weight=weight,reduction='none').sum()
    torch.testing.assert_close(value,expected)
    torch.testing.assert_close(mass,weight[labels].sum())


def test_masked_unknown_labels_have_no_gradient():
    logits=torch.randn(2,5,4,requires_grad=True)
    value,mass=weak.partial_losses(logits,torch.full((2,5),255),torch.zeros(2,5),torch.ones(4),1.)
    value.backward(); assert value.item()==mass.item()==0
    assert not logits.grad.any()


def test_microbatch_normalization_matches_full_batch_gradients():
    logits=torch.randn(4,7,4,requires_grad=True); labels=torch.randint(0,4,(4,7)); labels[0,:3]=255
    weight=torch.tensor([.5,1.,2.,3.]); mask=torch.ones(4,7)
    numerator,mass=weak.partial_losses(logits,labels,mask,weight,1.7)
    full=torch.autograd.grad(numerator/mass,logits)[0]
    divided=logits.detach().clone().requires_grad_()
    for a,b in [(0,2),(2,4)]:
        value,_=weak.partial_losses(divided[a:b],labels[a:b],mask[a:b],weight,1.7)
        (value/mass.detach()).backward()
    torch.testing.assert_close(divided.grad,full)


@pytest.mark.parametrize('state',[-1,4,254])
def test_other_unknown_training_states_rejected(state):
    with pytest.raises(ValueError):
        weak.partial_losses(torch.zeros(1,1,4),torch.tensor([[state]]),torch.ones(1,1),torch.ones(4),1.)


def test_weak_weights_do_not_count_unknowns_as_Walking_or_hesitation():
    _,aux,_=weak.synthetic(); weight,counts,unknown,fog_weight=weak.weak_weights(aux)
    expected=sum((np.bincount(r['classes'][r['valid']&(r['classes']<4)],minlength=4) for r in aux),np.zeros(4,np.int64))
    np.testing.assert_array_equal(counts,expected)
    assert unknown>0 and fog_weight==float(weight[1:].mean())


def test_frozen_raw_state_restores_after_failure():
    before=weak.raw.source_hash(); prepare=weak.raw.prepare
    with pytest.raises(RuntimeError):
        with weak.applied():
            assert weak.raw.source_hash()==weak.effective_source_hash()
            assert 'M22' in weak.raw.SPECS
            raise RuntimeError('fixture')
    assert weak.raw.source_hash()==before and weak.raw.prepare is prepare and 'M22' not in weak.raw.SPECS


def test_complete_synthetic_weak_pretraining_fine_tuning_and_replay(tmp_path):
    session={'suite':'metadata','cfg':weak.raw.base.smoke_config(),'output':tmp_path/'synthetic',
        'context':{},'results':[],'runtime':{},'deadline':float('inf')}
    result=weak.run(session); assert result['status']=='complete'
    audit=validator.audit(session['output'],'M22',replay=True)
    assert audit['auxiliary_weak_Event_pretraining'] and audit['auxiliary_recordings']==2
    assert audit['status']=='native_AP_and_model_replay_passed'
    assert not audit['retained_heads_changed']
    assert weak.raw.source_hash()==weak.typed.FROZEN_RAW_SHA256
