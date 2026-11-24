"""Read-only native AP, provenance and optional saved-model replay for v28.

No fitting or registry changes. Preprocessing/model definitions are shared;
overlap-add inference and AP calculations are implemented separately. A passed
replay is not fresh-subject generalization or permission to auto-promote.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import errno
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

import tdcs_raw_transfer_v28 as exp
from validate_tdcs_experiment import independent_ap

ROOT=Path(__file__).resolve().parents[1]
AUX_SIGNAL_ABSOLUTE_ROUNDOFF_G=1e-12


def read_json(path):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except OSError as error:
        if error.errno!=errno.EINVAL: raise
        # Reopen once on the observed Windows read error. Every referenced
        # training artifact is still checked against its SHA256 below.
        return json.loads(path.read_text(encoding='utf-8'))


def signature(records):
    return [{'id':r['id'],'subject':r['subject'],'fold':r['fold'],'domain':r['domain'],
        'signal':exp.data.array_digest(r['signal']),'classes':exp.data.array_digest(r['classes']),
        'valid':exp.data.array_digest(r['valid'])} for r in records]


def audit_dataset_signature(records, expected, output):
    """All arrays match byte-for-byte except verified typed aux near-zero roundoff.

    The exception requires identical raw CSV bytes and exact saved-cache
    digests/label/mask values. TDCS and weak inputs are never relaxed. Neither
    the native scoring checks nor the model-replay tolerance changes.
    """
    actual=signature(records)
    if actual==expected: return []
    assert len(actual)==len(expected), 'Dataset recording count differs'
    root=Path(output)/'raw_v28'
    manifest=json.loads((root/'raw_dataset_manifest.json').read_text())
    provenance={r['id']:r for r in manifest['records'] if r['domain']=='defog'}
    assert len(provenance)==sum(r['domain']=='defog' for r in manifest['records'])
    directory=exp.data.frozen.shared.common.locate_data()[0].parent/'defog'
    roundoff=[]
    for record,got,want in zip(records,actual,expected):
        if got==want: continue
        id=record['id']
        different={key for key in want.keys()|got.keys() if got.get(key)!=want.get(key)}
        assert different=={'signal'} and record['domain']=='defog' and re.fullmatch(r'[a-f0-9]{10}',id), (id,different)
        assert exp.data.digest(directory/(id+'.csv'))==provenance[id]['source_sha256'], 'Auxiliary raw CSV bytes differ'
        with np.load(root/'raw_cache'/(id+'.npz'),allow_pickle=False) as saved:
            assert exp.data.array_digest(saved['signal'])==want['signal']
            np.testing.assert_array_equal(saved['classes'],record['classes'])
            np.testing.assert_array_equal(saved['valid'],record['valid'])
            assert saved['signal'].shape==record['signal'].shape and saved['signal'].dtype==record['signal'].dtype
            np.testing.assert_allclose(saved['signal'],record['signal'],atol=AUX_SIGNAL_ABSOLUTE_ROUNDOFF_G,rtol=0,equal_nan=False)
            delta=np.abs(saved['signal'].astype(np.float64)-record['signal'])
        roundoff.append({'id':id,'different_signal_values':int(np.count_nonzero(delta)),'signal_values':int(delta.size),
            'max_absolute_difference_g':float(delta.max()),'raw_source_and_saved_cache_hashes_verified':True,
            'class_and_valid_arrays_exact':True})
    return roundoff


def load_records(context,transfer,weak_event=False,weak_root=None,weak_manifest=None):
    main=[]; aux=[]
    for i,row in enumerate(context['manifest'].itertuples(index=False)):
        frame=pd.read_csv(row.path,usecols=['Time',*exp.data.AXES,*exp.LABELS])
        a,b=context['native_offsets'][i:i+2]
        np.testing.assert_array_equal(frame[['StartHesitation','Walking']].to_numpy(),context['native_y'][a:b])
        main.append(exp.make_record(frame,row.recording_id,row.subject,row.fold,128,'tdcs'))
    if transfer:
        from tdcs_raw_transfer_v28_recovery import auxiliary_inventory
        excluded=set(pd.read_csv(exp.data.csv_path('tdcsfog_metadata.csv'),dtype=str).Subject)
        meta=pd.read_csv(exp.data.csv_path('defog_metadata.csv'),dtype=str)
        raw_root=exp.data.frozen.shared.common.locate_data()[0].parent/'defog'
        selected,_=auxiliary_inventory(meta,raw_root,excluded)
        for row in selected.itertuples(index=False):
            frame=pd.read_csv(row.path)
            aux.append(exp.make_record(frame,row.Id,row.Subject,-1,100,'defog'))
        if weak_event:
            import tdcs_raw_weak_transfer_v29 as weak
            if weak_root is None:
                selected,_=auxiliary_inventory(meta,raw_root.parent/'notype',excluded,expected_files=46)
                for row in selected.itertuples(index=False):
                    aux.append(weak.make_weak_record(pd.read_csv(row.path),row.Id,row.Subject))
            else:
                from tdcs_weak_event_evidence import inventory, checked_csv_bytes
                assert weak_manifest is not None
                selected=inventory(weak_manifest,meta,{p.stem for p in raw_root.glob('*.csv')},excluded)
                for row in selected:
                    frame=pd.read_csv(BytesIO(checked_csv_bytes(weak_root,row)),usecols=['Time',*exp.data.AXES,'Event','Valid','Task'])
                    record=weak.make_weak_record(frame,row['id'],row['subject'])
                    assert int(record['valid'].sum())==row['valid_rows']
                    assert int(((record['classes']==weak.UNKNOWN_FOG)&record['valid']).sum())==row['positive_partial_Event_rows']
                    aux.append(record)
        assert set(r['subject'] for r in aux).isdisjoint(excluded)
    return main,aux,context['native_y'],context['gates']


def replay_record(model,record,cfg,device):
    """Independent window accumulation; shares only features/model definition."""
    length=len(record['signal']); accumulated=np.zeros((length,2),np.float64)
    mass=np.zeros(length,np.float64); window=cfg.infer_length
    weights=np.maximum(.1,np.hanning(window))
    for first in range(-(window//2),length,window//2):
        coordinates=np.arange(window)+first
        with torch.no_grad():
            logits=model(torch.from_numpy(exp.features(record,coordinates)[None]).to(device))
            values=torch.softmax(logits,-1).cpu().numpy()[0][:,[1,3]]
        valid=(coordinates>=0)&(coordinates<length); index=coordinates[valid]
        accumulated[index]+=values[valid]*weights[valid,None]; mass[index]+=weights[valid]
    assert (mass>0).all()
    return (accumulated/mass[:,None]).astype(np.float32)


def replay_records(model,records,cfg,device,batch_windows=8):
    """Batch independent windows; GroupNorm and GRUs do not mix batch members."""
    if batch_windows<1: raise ValueError('Positive replay batch size required')
    window=cfg.infer_length; taper=np.maximum(.1,np.hanning(window))
    totals=[np.zeros((len(r['signal']),2),np.float64) for r in records]
    masses=[np.zeros(len(r['signal']),np.float64) for r in records]
    work=[(i,start) for i,r in enumerate(records) for start in range(-(window//2),len(r['signal']),window//2)]
    for first in range(0,len(work),batch_windows):
        selected=work[first:first+batch_windows]
        inputs=np.stack([exp.features(records[i],start+np.arange(window)) for i,start in selected])
        with torch.no_grad():
            values=torch.softmax(model(torch.from_numpy(inputs).to(device)),-1).cpu().numpy()[:,:,[1,3]]
        for j,(i,start) in enumerate(selected):
            coordinates=start+np.arange(window); valid=(coordinates>=0)&(coordinates<len(masses[i]))
            index=coordinates[valid]; totals[i][index]+=values[j,valid]*taper[valid,None]
            masses[i][index]+=taper[valid]
        if first%128==0: print('Read-only replay windows:',min(first+batch_windows,len(work)),'/',len(work),flush=True)
    assert all((m>0).all() for m in masses)
    return [(a/b[:,None]).astype(np.float32) for a,b in zip(totals,masses)]


def audit_trace(crops,records,cfg,steps):
    assert len(crops)==steps*cfg.batch_size
    by_id={r['id']:r for r in records}
    for id,subject,center,stretch in crops:
        assert id in by_id; record=by_id[id]
        assert subject==record['subject'] and 0<=center<len(record['signal'])
        assert int(center)==center and record['valid'][int(center)]
        assert .8<=stretch<=1.2


def native_predictions(records,parts,gates,gated):
    predictions=[]
    for record,grid,gate in zip(records,parts,gates):
        assert grid.shape==(len(record['signal']),2)
        coordinates=np.arange(len(grid))*record['source_hz']/50.
        p=np.column_stack([np.interp(np.arange(record['native_length']),coordinates,grid[:,j]) for j in range(2)]).astype(np.float32)
        if gated: p[:,1]*=np.float32(gate)*np.float32(.95)+np.float32(.05)
        predictions.append(p)
    return np.concatenate(predictions)


def audit(output,id,replay=False,context=None,tolerance=2e-5,progress=None,weak_root=None):
    output=Path(output); status=json.loads((output/(id+'_status.json')).read_text())
    assert status['status']=='complete'
    run=output/'methods'/id/Path(status['run_directory']).name
    config=json.loads((run/'configuration.json').read_text()); cfg=exp.Config(**config['configuration'])
    if config['source_sha256']!=exp.source_hash():
        import tdcs_raw_transfer_v28_recovery as recovery
        if id=='S10' and config['source_sha256']==recovery.effective_source_hash():
            with recovery.applied(): return audit(output,id,replay,context,tolerance,progress,weak_root)
        if id=='S11':
            import tdcs_raw_walking_rank_v30 as rank
            assert config['source_sha256']==rank.effective_source_hash(), 'Unrecognized ranking revision'
            with rank.applied(): return audit(output,id,replay,context,tolerance,progress,weak_root)
        import tdcs_raw_weak_transfer_v29 as weak
        assert id=='M22' and config['source_sha256']==weak.effective_source_hash(), 'Unrecognized reader/model revision'
        with weak.applied(): return audit(output,id,replay,context,tolerance,progress,weak_root)
    assert exp.key(config)==run.name and config['source_sha256']==exp.source_hash()
    assert config['base_sources']==exp.base.source_hashes() and config['id']==id
    assert config['specification']==exp.SPECS[id] and config['targets']==list(exp.data.TARGETS)
    assert not config['outer_stopping'] and not config['automatic_promotion']
    assert cfg.smoke==status['smoke_only']
    if not cfg.smoke: assert asdict(cfg)==asdict(exp.Config())
    transfer=config['specification']['transfer']
    weak_event=config['specification'].get('weak_notype_Event_pretraining',False)
    if cfg.smoke:
        main,aux,y=exp.synthetic(); gates=np.ones(len(main),np.float32)
    else:
        context=context or exp.data.load_context()
        if weak_root is not None:
            assert weak_event and Path(weak_root).resolve().is_relative_to(ROOT)
            weak_manifest=read_json(output/'raw_v28/raw_dataset_manifest.json')
        else: weak_manifest=None
        main,aux,y,gates=load_records(context,transfer,weak_event,weak_root,weak_manifest)
        assert (len(main),len({r['subject'] for r in main}),len(y))==(815,61,6959762)
    roundoff=audit_dataset_signature([*main,*aux],config['dataset_signature'],output)
    usage=None
    if id=='S11':
        usage=json.loads((run/'walking_rank_usage.json').read_text())
        assert usage['source_sha256']==exp.source_hash() and usage['config']==exp.SPECS[id]
        assert not usage['warmup_received_ranking_term'] and len(usage['folds'])==5
        for fold, item in enumerate(usage['folds']):
            assert item['fold']==fold and item['seed']==cfg.seed+1009*fold+200000
            assert item['total_microbatches']==cfg.fine_steps*(cfg.batch_size//cfg.micro_batch)
            assert 0<=item['eligible_microbatches']<=item['total_microbatches'] and item['pairs']>=item['eligible_microbatches']
            if not cfg.smoke: assert item['eligible_microbatches']>0 and item['pairs']>0
    parts=[None]*len(main); replay_maximum=0.; device=torch.device('cpu'); torch.set_num_threads(2)
    completed={}
    if progress is not None:
        if not replay: raise ValueError('Fold progress requires saved-model replay')
        progress=Path(progress)
        binding={'id':id,'run':run.name,'configuration_sha256':exp.data.digest(run/'configuration.json'),
            'validator_sha256':exp.data.digest(__file__),'torch':str(torch.__version__), 'cpu_threads':2,
            'numpy':np.__version__,'tolerance':tolerance}
        if progress.is_file():
            previous=read_json(progress)
            assert previous['binding']==binding, 'Replay progress is for different inputs or validation code'
            completed=previous['folds']
            assert set(completed)<=set('01234')
    if transfer:
        auxiliary=json.loads((run/'auxiliary_pretraining_receipt.json').read_text())
        assert set(auxiliary['subjects']).isdisjoint({r['subject'] for r in main})
        assert set(auxiliary['subjects'])=={r['subject'] for r in aux if r['valid'].any()}
        assert auxiliary['updates']==cfg.warm_steps
        if weak_event:
            import tdcs_raw_weak_transfer_v29 as weak
            weights,counts,unknown,partial_weight=weak.weak_weights(aux)
            assert auxiliary['unknown_subtype_positive_rows']==unknown
            assert auxiliary['unknown_subtype_positive_weight']==partial_weight
        else: weights,counts=exp.class_weights(aux)
        np.testing.assert_array_equal(counts,auxiliary['class_counts'])
        np.testing.assert_array_equal(weights,auxiliary['class_weights'])
        audit_trace(json.loads((run/'auxiliary_pretraining_trace.json').read_text())['crops'],aux,cfg,cfg.warm_steps)
        checkpoint=torch.load(run/'defog_pretrained.pt',map_location='cpu',weights_only=True)
        assert checkpoint['updates']==cfg.warm_steps and checkpoint['source_sha256']==exp.source_hash()
    with np.load(run/'grid_scores.npz',allow_pickle=False) as saved:
        expected=saved['prediction']; offsets=saved['offsets']
        np.testing.assert_array_equal(saved['recording_ids'],[r['id'] for r in main])
        np.testing.assert_array_equal(saved['native_lengths'],[r['native_length'] for r in main])
        np.testing.assert_array_equal(saved['targets'],exp.data.TARGETS)
        np.testing.assert_array_equal(offsets,np.r_[0,np.cumsum([len(r['signal']) for r in main])])
    for fold in range(5):
        directory=run/f'fold_{fold}'; receipt=read_json(directory/'training_receipt.json')
        fitting=[r for r in main if r['fold']!=fold]; heldout=[i for i,r in enumerate(main) if r['fold']==fold]
        assert receipt['fold']==fold and receipt['fingerprint']==run.name
        assert set(receipt['fitting_subjects'])=={r['subject'] for r in fitting}
        assert set(receipt['heldout_subjects'])=={main[i]['subject'] for i in heldout}
        assert set(receipt['warmup_subjects']).isdisjoint(receipt['heldout_subjects'])
        assert receipt['warmup_updates']==cfg.warm_steps and receipt['fine_tune_updates']==cfg.fine_steps
        assert not receipt['outer_labels_used_for_training_or_stopping']
        assert receipt['saved_model_reloaded_for_all_heldout_predictions']
        for name,sha in receipt['artifact_sha256'].items(): assert exp.data.digest(directory/name)==sha
        weights,counts=exp.class_weights(fitting)
        np.testing.assert_array_equal(counts,receipt['fine_tune_class_counts'])
        np.testing.assert_array_equal(weights,receipt['fine_tune_class_weights'])
        trace=read_json(directory/'training_trace.json')
        audit_trace(trace['fine_tune'],fitting,cfg,cfg.fine_steps)
        if not transfer: audit_trace(trace['warmup'],fitting,cfg,cfg.warm_steps)
        else:
            assert trace['warmup']=='shared_independent_DeFOG'
            warmup=torch.load(directory/'warmup.pt',map_location='cpu',weights_only=True)
            assert all(torch.equal(warmup['model'][k],v) for k,v in checkpoint['model'].items())
        with np.load(directory/'heldout_grid.npz',allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['record_indices'],heldout)
            local_offsets=saved['offsets']; local=saved['prediction']
            np.testing.assert_array_equal(local_offsets,np.r_[0,np.cumsum([len(main[i]['signal']) for i in heldout])])
            for k,i in enumerate(heldout):
                parts[i]=local[local_offsets[k]:local_offsets[k+1]]
                np.testing.assert_array_equal(parts[i],expected[offsets[i]:offsets[i+1]])
        if replay:
            hashes={name:receipt['artifact_sha256'][name] for name in ('model.pt','heldout_grid.npz')}
            if str(fold) in completed:
                entry=completed[str(fold)]
                assert entry['artifact_sha256']==hashes and entry['heldout_recordings']==len(heldout)
                difference=entry['max_abs_prediction_difference']
                assert 0<=difference<=tolerance
            else:
                payload=torch.load(directory/'model.pt',map_location=device,weights_only=True)
                assert payload['configuration']==asdict(cfg) and payload['fold']==fold and payload['fingerprint']==run.name
                model=exp.Network(cfg).to(device); model.load_state_dict(payload['model']); model.eval()
                replayed=replay_records(model,[main[i] for i in heldout],cfg,device)
                assert len(replayed)==len(heldout)
                differences=[float(np.max(np.abs(actual-parts[i]))) for i,actual in zip(heldout,replayed)]
                assert all(0<=d<=tolerance for d in differences), (fold,tolerance)
                difference=max(differences)
                completed[str(fold)]={'max_abs_prediction_difference':difference,'artifact_sha256':hashes,
                    'heldout_recordings':len(heldout)}
                if progress is not None: exp.base.write(progress,{'binding':binding,'folds':completed})
            replay_maximum=max(replay_maximum,difference)
        print(id,'audited fold',fold,'; replay',replay,flush=True)
    decision=json.loads((run/'decision.json').read_text()); values={}; predictions={}
    for gated in (False,True):
        name='frozen_Walking_gate_once' if gated else 'ungated'
        p=native_predictions(main,parts,gates,gated)
        assert p.shape==y.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        metrics={t:independent_ap(y[:,j],p[:,j]) for j,t in enumerate(exp.data.TARGETS)}
        metrics['two_target_mAP']=float(np.mean(list(metrics.values())))
        reference=decision['fixed_gate_ablation'] if gated else status
        for target,value in metrics.items(): assert abs(value-reference[target])<1e-10
        checksum=decision['fixed_gate_native_sha256'] if gated else status['native_prediction_sha256']
        assert exp.data.array_digest(p)==checksum
        values[name]=metrics; predictions[name]=p
    folds=np.concatenate([np.full(r['native_length'],r['fold'],np.int8) for r in main])
    per_fold=[]
    for fold in range(5):
        selected=folds==fold
        per_fold.append({'fold':fold,**{t:independent_ap(y[selected,j],predictions['ungated'][selected,j]) if y[selected,j].any() else None
            for j,t in enumerate(exp.data.TARGETS)}})
    result={'id':id,'status':'native_AP_and_model_replay_passed' if replay else 'native_AP_audited_replay_pending',
        'metrics':values,'ungated_folds':per_fold,'native_rows':len(y),'all_negative_recordings_retained':True,
        'saved_model_replay':replay,'maximum_replay_difference':replay_maximum if replay else None,
        'replayed_fold_receipts':completed if replay else {},
        'preprocessing_and_model_definition_shared':True,'independent_generalization_established':False,
        'retained_heads_changed':False,'smoke_only':cfg.smoke,'run':str(run),'no_local_fitting':True,
        'auxiliary_recordings':len(aux) if transfer else 0,'auxiliary_weak_Event_pretraining':weak_event,
        'weak_source_CSV_hashes_and_annotation_counts_verified':bool(weak_root and weak_event and not cfg.smoke),
        'verified_auxiliary_signal_roundoff':roundoff,'Walking_rank_usage':usage}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-output',required=True,type=Path)
    parser.add_argument('--id',required=True,choices=['M21','S10','M22','S11'])
    parser.add_argument('--replay',action='store_true')
    parser.add_argument('--replay-progress',type=Path,help='Save fingerprinted fold replay receipts so a local interruption can be resumed')
    parser.add_argument('--weak-root',type=Path,help='Exact downloaded weak training CSV/ZIP inventory for M22 validation')
    parser.add_argument('--result',required=True,type=Path)
    args=parser.parse_args()
    if not args.result.resolve().is_relative_to(ROOT): raise ValueError('Result must stay within the workspace')
    if args.replay_progress is not None:
        assert args.replay_progress.resolve().is_relative_to(ROOT/'results')
        assert args.replay_progress.resolve()!=args.result.resolve()
    result=audit(args.suite_output,args.id,args.replay,progress=args.replay_progress,weak_root=args.weak_root)
    exp.base.write(args.result,result); print(json.dumps(result,indent=2))
