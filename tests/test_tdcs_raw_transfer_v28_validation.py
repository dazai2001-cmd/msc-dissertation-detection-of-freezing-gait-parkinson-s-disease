"""The independent read-only replay is exercised only on synthetic fits."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import tdcs_raw_transfer_v28 as exp
import validate_tdcs_raw_transfer_v28 as validator


@pytest.mark.parametrize('id,suite',[('M21','metadata'),('S10','sensor')])
def test_saved_model_native_AP_and_trace_validation(id,suite,tmp_path):
    session={'suite':suite,'cfg':exp.base.smoke_config(),'output':tmp_path/id,
        'context':{},'results':[],'runtime':{},'deadline':float('inf')}
    assert exp.run(session,id)['status']=='complete'
    result=validator.audit(session['output'],id,replay=True)
    assert result['status']=='native_AP_and_model_replay_passed'
    assert result['maximum_replay_difference']<=2e-5 and result['no_local_fitting']
    assert result['native_rows']==960 and not result['retained_heads_changed']


def test_unannotated_auxiliary_center_cannot_pass_trace_audit():
    _,aux,_=exp.synthetic(); cfg=exp.smoke_config(); r=aux[0]
    invalid=int(np.flatnonzero(~r['valid'])[0])
    with pytest.raises(AssertionError):
        validator.audit_trace([(r['id'],r['subject'],invalid,1.)]*cfg.batch_size,aux,cfg,1)


def test_native_interpolation_and_gate_match_original_checksums():
    records,_,y=exp.synthetic(); rng=np.random.default_rng(13)
    parts=[rng.uniform(0,1,(len(r['signal']),2)).astype(np.float32) for r in records]
    gates=np.asarray([.1,.3,.5,.7,.9],np.float32)
    for gated in (False,True):
        _,checksum=exp.evaluate(records,y,parts,gates,gated)
        p=validator.native_predictions(records,parts,gates,gated)
        assert exp.data.array_digest(p)==checksum


def test_batched_replay_agrees_with_independent_single_window_accumulation():
    import torch
    torch.set_num_threads(2); cfg=exp.smoke_config(); model=exp.Network(cfg).eval()
    records,_,_=exp.synthetic(); selected=records[:3]
    batched=validator.replay_records(model,selected,cfg,torch.device('cpu'),batch_windows=4)
    for record,actual in zip(selected,batched):
        expected=validator.replay_record(model,record,cfg,torch.device('cpu'))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)


def test_typed_roundoff_requires_original_bytes_and_exact_Kaggle_cache(tmp_path, monkeypatch):
    import copy
    _, aux, _ = exp.synthetic(); got=copy.deepcopy(aux[0]); got['id']='0123456789'
    got['signal'][0,0]=0
    export=copy.deepcopy(got); export['signal'][0,0]=np.float32(1e-20)
    expected=validator.signature([export]); directory=tmp_path/'train'/'defog'
    directory.mkdir(parents=True); source=directory/(got['id']+'.csv'); source.write_bytes(b'synthetic')
    monkeypatch.setattr(exp.data.frozen.shared.common,'locate_data',lambda: (directory.parent/'tdcsfog',None,None))
    root=tmp_path/'raw_v28'; cache=root/'raw_cache'; cache.mkdir(parents=True)
    (root/'raw_dataset_manifest.json').write_text(json.dumps({'records':[{'id':got['id'],'domain':'defog',
        'source_sha256':exp.data.digest(source)}]}))
    path=cache/(got['id']+'.npz')
    np.savez(path,signal=export['signal'],classes=export['classes'],valid=export['valid'])
    assert validator.audit_dataset_signature([got],expected,tmp_path)[0]['different_signal_values']==1
    got['signal'][0,0]=.01
    with pytest.raises(AssertionError): validator.audit_dataset_signature([got],expected,tmp_path)
    got['signal'][0,0]=0; source.write_bytes(b'wrong input')
    with pytest.raises(AssertionError,match='raw CSV bytes'): validator.audit_dataset_signature([got],expected,tmp_path)


def test_resumable_replay_never_reuses_a_wrong_fingerprint(tmp_path,monkeypatch):
    session={'suite':'metadata','cfg':exp.base.smoke_config(),'output':tmp_path/'synthetic',
        'context':{},'results':[],'runtime':{},'deadline':float('inf')}
    assert exp.run(session,'M21')['status']=='complete'
    progress=tmp_path/'replayed.json'
    first=validator.audit(session['output'],'M21',replay=True,progress=progress)
    assert len(first['replayed_fold_receipts'])==5
    def not_needed(*args,**kwargs): raise AssertionError('Already verified fold ran inference again')
    monkeypatch.setattr(validator,'replay_records',not_needed)
    assert validator.audit(session['output'],'M21',replay=True,progress=progress)==first
    payload=json.loads(progress.read_text()); payload['binding']['tolerance']=1
    progress.write_text(json.dumps(payload))
    with pytest.raises(AssertionError,match='different inputs'):
        validator.audit(session['output'],'M21',replay=True,progress=progress)
