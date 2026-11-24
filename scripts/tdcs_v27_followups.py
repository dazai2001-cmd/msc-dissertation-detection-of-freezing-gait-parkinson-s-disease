"""Single-head v27 followups; frozen v26 data/recipes remain unchanged.

Walking: true one-output GRUs, all versus local sensors plus recording fields.
Hesitation: three pseudo-row weights with the exact same cached teacher/rows.
Real fitting is Kaggle CUDA only. Scores never promote a retained predictor.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

import tdcs_suite_data as data
import tdcs_suite_models as trees
import tdcs_suite_neural as neural
import tdcs_suite_runner as base

SPECS={
    'M19':{'id':'M19','suite':'metadata','target':'Walking','kind':'walking_gru','encoding':'record_metadata'},
    'M20':{'id':'M20','suite':'metadata','target':'Walking','kind':'walking_gru','encoding':'local_metadata'},
    'S07':{'id':'S07','suite':'sensor','target':'StartHesitation','kind':'pseudo','encoding':'sensor','pseudo_weight':.03},
    'S08':{'id':'S08','suite':'sensor','target':'StartHesitation','kind':'pseudo','encoding':'sensor','pseudo_weight':.10},
    'S09':{'id':'S09','suite':'sensor','target':'StartHesitation','kind':'pseudo','encoding':'sensor','pseudo_weight':.30},
}


def source_hash():
    return globals().get('SOURCE_SHA256') or data.digest(__file__)


def fingerprint(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]


def check_time(session):
    if time.monotonic()>=session['deadline']: raise TimeoutError('Followup suite runtime budget reached')


def assert_fit_scope(session):
    if not session['cfg'].smoke and (not Path('/kaggle/working').exists() or not torch.cuda.is_available()):
        raise RuntimeError('Real followup fitting requires Kaggle CUDA; no local or CPU fallback')


def walking_model(features,width):
    model=neural.Network(features,'bigru',width,False)
    # Preserve the initial Walking row of the original two-output model. This
    # avoids an unrelated random output initialization in the loss-scope test.
    original=model.output
    with torch.random.fork_rng(devices=[]): head=nn.Linear(original.in_features,1)
    with torch.no_grad():
        head.weight.copy_(original.weight[1:2]); head.bias.copy_(original.bias[1:2])
    model.output=head
    return model


def load_walking(path,device):
    saved=torch.load(path,map_location=device,weights_only=False)
    assert saved['targets']==['Walking']
    model=walking_model(saved['features'],saved['width']).to(device)
    model.load_state_dict(saved['model'])
    return model.eval()


def predict_walking(model,x,prep,cfg,device):
    model.eval(); values=neural.transform(x,prep)
    starts=list(range(-cfg.window//2,len(x),cfg.window//2)); taper=np.maximum(np.hanning(cfg.window),.1)
    total=np.zeros(len(x),np.float64); denominator=np.zeros(len(x),np.float64)
    with torch.no_grad():
        for offset in range(0,len(starts),cfg.batch_size):
            group=starts[offset:offset+cfg.batch_size]
            windows=np.stack([values[np.clip(s+np.arange(cfg.window),0,len(x)-1)] for s in group])
            probability=torch.sigmoid(model(torch.from_numpy(windows).to(device)))[...,0].cpu().numpy()
            for start,p in zip(group,probability):
                inside=(start+np.arange(cfg.window)>=0)&(start+np.arange(cfg.window)<len(x))
                rows=start+np.flatnonzero(inside)
                total[rows]+=p[inside]*taper[inside]; denominator[rows]+=taper[inside]
    assert (denominator>0).all()
    return (total/denominator).astype(np.float32)


def train_walking(session,spec,x,fold,directory,key):
    c,cfg=session['context'],session['cfg']; a=c['arrays']; fitting=a['fold']!=fold
    device=torch.device('cpu' if cfg.smoke else 'cuda:0'); seed=cfg.seed+1009*fold
    torch.manual_seed(seed)
    if device.type=='cuda': torch.cuda.manual_seed_all(seed)
    labels=a['y'][:,[1]].astype(np.float32)
    prep=neural.preprocessing(x,labels,fitting); np.savez_compressed(directory/'preprocessing.npz',**prep)
    values=neural.transform(x,prep); model=walking_model(x.shape[1],cfg.width).to(device)
    ema=copy.deepcopy(model).eval(); groups,probability=neural.sampling_groups(a,fitting)
    rng=np.random.default_rng(seed); optimizer=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.01)
    positive=torch.from_numpy(prep['positive_weight']).to(device); trace=[]
    model.train()
    for epoch in range(cfg.neural_epochs):
        for _ in range(cfg.steps_per_epoch):
            check_time(session)
            bx,by,mask,crops=neural.batch(groups,probability,rng,values,labels,cfg); trace.extend(crops)
            logits=model(torch.from_numpy(bx).to(device))[...,0]
            truth=torch.from_numpy(by[...,0]).to(device); valid=torch.from_numpy(mask).to(device).float()
            loss=(F.binary_cross_entropy_with_logits(logits,truth,pos_weight=positive,reduction='none')*valid).sum()/valid.sum().clamp_min(1)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite Walking loss')
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            with torch.no_grad():
                for dst,src in zip(ema.parameters(),model.parameters()): dst.lerp_(src,.005)
        torch.save({'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
            'completed_epochs':epoch+1,'fingerprint':key},directory/'epoch_checkpoint.pt')
        print(f"{spec['id']} fold {fold}: epoch {epoch+1}/{cfg.neural_epochs}",flush=True)
    np.savez_compressed(directory/'training_trace.npz',crops=np.asarray(trace,np.int64))
    torch.save({'model':ema.state_dict(),'features':x.shape[1],'width':cfg.width,'fingerprint':key,
        'fold':fold,'targets':['Walking']},directory/'model.pt')
    result=np.empty(int((~fitting).sum()),np.float32); where=np.full(len(x),-1,np.int64); where[~fitting]=np.arange((~fitting).sum())
    for record in np.unique(a['record'][~fitting]):
        rows=np.flatnonzero((a['record']==record)&~fitting)
        result[where[rows]]=predict_walking(ema,x[rows],prep,cfg,device)
    return result,{'backend':str(device),'seed':seed,'optimizer_updates':cfg.neural_epochs*cfg.steps_per_epoch,
        'fixed_epochs':cfg.neural_epochs,'final_weights':'EMA_0.995','output_heads':['Walking'],
        'SH_or_Turn_losses_used':False,'initial_Walking_head_preserved_from_two_head_initialization':True}


def shared_teacher(session,x,pool,fold):
    c,cfg=session['context'],session['cfg']; a=c['arrays']; seed=cfg.seed+1009*fold
    config={'scope':'one SH teacher shared by all pseudo-weight tests','fold':fold,'seed':seed,
        'tree_rounds':cfg.tree_rounds,'feature_sha256':data.array_digest(x),
        'pool_sha256':{k:data.array_digest(v) for k,v in pool.items()},
        'sources':base.source_hashes(),'followup_source_sha256':source_hash(),'smoke_only':cfg.smoke}
    root=session['output']/'shared_teachers'/fingerprint(config)/f'fold_{fold}'; root.mkdir(parents=True,exist_ok=True)
    receipt_path=root/'teacher_receipt.json'
    if receipt_path.is_file():
        receipt=json.loads(receipt_path.read_text()); assert receipt['configuration']==config
        for name,sha in receipt['artifact_sha256'].items(): assert data.digest(root/name)==sha
    else:
        if any(root.iterdir()): raise ValueError('Incomplete shared teacher; preserve it and use a fresh output root')
        chosen,weights,probability=data.frozen.parent.training_plan(a['y'][:,0],a['fold'],a['subject'],fold,seed,4,.5)
        model,backend=trees.fit_tree('catboost',x[chosen],a['y'][chosen,0],weights,seed,cfg,not cfg.smoke)
        model_path=trees.save_tree(model,'catboost',root/'teacher')
        heldout_subjects=set(a['subject'][a['fold']==fold]); eligible=data.external_training_indices(pool,heldout_subjects)
        p=trees.predict_tree(model,'catboost',pool['x'][eligible])
        selected=np.flatnonzero((p>=.90)|(p<=.01))
        if len(selected)>40000: selected=np.random.default_rng(seed).choice(selected,40000,replace=False)
        np.savez_compressed(root/'rows.npz',indices=chosen,weights=weights,pseudo_indices=eligible[selected],
            pseudo_labels=(p[selected]>=.90).astype(np.uint8),teacher_probabilities=p[selected])
        receipt={'configuration':config,'backend':backend,'model_file':model_path.name,
            'sample_probability':probability,'artifact_sha256':{p.name:data.digest(p) for p in root.iterdir() if p.is_file()}}
        base.write(receipt_path,receipt)
    with np.load(root/'rows.npz',allow_pickle=False) as saved: rows={k:saved[k] for k in saved.files}
    return root,receipt,rows


def train_pseudo(session,spec,x,pool,fold,directory,key):
    c,cfg=session['context'],session['cfg']; a=c['arrays']; seed=cfg.seed+1009*fold
    teacher,receipt,rows=shared_teacher(session,x,pool,fold)
    chosen,weights=rows['indices'],rows['weights']; selected=rows['pseudo_indices']; pseudo=rows['pseudo_labels']
    assert (a['fold'][chosen]!=fold).all()
    assert not set(pool['subject'][selected])&set(a['subject'][a['fold']==fold])
    if len(selected):
        w=np.r_[weights,np.full(len(selected),spec['pseudo_weight'])]; w/=w.mean()
        model,backend=trees.fit_tree('catboost',np.vstack([x[chosen],pool['x'][selected]]),
            np.r_[a['y'][chosen,0],pseudo],w,seed,cfg,not cfg.smoke)
    else:
        model=trees.load_tree('catboost',teacher/receipt['model_file']); backend={'backend':'shared_teacher_no_qualified_rows'}
    path=trees.save_tree(model,'catboost',directory/'StartHesitation')
    np.savez_compressed(directory/'training_sample.npz',**rows)
    result=trees.predict_tree(trees.load_tree('catboost',path),'catboost',x[a['fold']==fold])
    return result,{'backend':backend,'seed':seed,'model_file':path.name,'output_heads':['StartHesitation'],
        'pseudo_weight':spec['pseudo_weight'],'pseudo_rows_added':len(selected),'positive_pseudo_labels':int(pseudo.sum()),
        'shared_teacher_directory':str(teacher),'shared_teacher_receipt_sha256':data.digest(teacher/'teacher_receipt.json'),
        'shared_teacher_rows_sha256':data.digest(teacher/'rows.npz'),'teacher_changed_between_weight_tests':False}


def native_scores(c,grid,target):
    values=np.empty(len(c['native_y']),np.float32); a=c['arrays']
    for i,row in enumerate(c['manifest'].itertuples(index=False)):
        mask=a['record']==i; left,right=c['native_offsets'][i:i+2]
        values[left:right]=np.interp(np.arange(right-left),a['grid'][mask],grid[mask]).astype(np.float32)
        if target=='Walking': values[left:right]*=(np.asarray([c['gates'][i]],np.float32)*.95+.05)[0]
    ap=float(average_precision_score(c['native_y'][:,data.TARGETS.index(target)],values))
    return ap,data.array_digest(values)


def run_method(session,id):
    spec=SPECS[id]; assert spec['suite']==session['suite']; cfg=session['cfg']; c=session['context']; a=c['arrays']
    status={'id':id,'title':id+' '+spec['target']+' '+spec['kind'],'suite':session['suite'],'status':'pending','target_scope':[spec['target']]}
    try:
        assert_fit_scope(session); check_time(session)
        x=data.input_features(c,spec['encoding'])
        pool=data.external_pool(c,session['output']) if spec['kind']=='pseudo' else None
        config={'specification':spec,'configuration':asdict(cfg),'base_sources':base.source_hashes(),
            'followup_source_sha256':source_hash(),'feature_sha256':data.array_digest(x),'feature_count':x.shape[1],
            'external_pool_sha256':{k:data.array_digest(v) for k,v in pool.items()} if pool is not None else None,
            'source_inputs':c['source_inputs'],'runtime':session['runtime'],'targets':[spec['target']],
            'recordings':c['manifest'].recording_id.tolist(),'subjects':c['manifest'].subject.tolist(),
            'folds':c['manifest'].fold.tolist(),'smoke_only':cfg.smoke,'outer_stopping':False,'automatic_promotion':False}
        key=fingerprint(config); run=session['output']/'methods'/id/key; run.mkdir(parents=True,exist_ok=True)
        base.write(run/'configuration.json',config); grid=np.full(len(x),np.nan,np.float32)
        for fold in cfg.folds:
            check_time(session); directory=run/f'fold_{fold}'; directory.mkdir(exist_ok=True)
            receipt_path=directory/'training_receipt.json'; saved=directory/'heldout_grid.npz'; heldout=a['fold']==fold
            if receipt_path.is_file() and saved.is_file():
                receipt=json.loads(receipt_path.read_text()); assert receipt['fingerprint']==key
                for name,sha in receipt['artifact_sha256'].items(): assert data.digest(directory/name)==sha
                with np.load(saved,allow_pickle=False) as payload:
                    np.testing.assert_array_equal(payload['indices'],np.flatnonzero(heldout)); grid[heldout]=payload['prediction']
                continue
            if any(directory.iterdir()): raise ValueError('Incomplete followup fold; preserve it and use a fresh output root')
            parts,receipt=(train_pseudo(session,spec,x,pool,fold,directory,key) if spec['kind']=='pseudo' else
                train_walking(session,spec,x,fold,directory,key))
            assert parts.shape==(int(heldout.sum()),) and np.isfinite(parts).all() and ((parts>=0)&(parts<=1)).all()
            np.savez_compressed(saved,indices=np.flatnonzero(heldout),prediction=parts); grid[heldout]=parts
            receipt.update(fingerprint=key,fold=fold,fitting_subjects=sorted(map(str,set(a['subject'][~heldout]))),
                heldout_subjects=sorted(map(str,set(a['subject'][heldout]))),outer_labels_used_for_training_or_stopping=False,
                artifact_sha256={p.name:data.digest(p) for p in directory.iterdir() if p.is_file() and p.name!='training_receipt.json'})
            assert set(receipt['fitting_subjects']).isdisjoint(receipt['heldout_subjects'])
            base.write(receipt_path,receipt); print(id,'fold',fold,'complete',flush=True)
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        np.savez_compressed(run/'grid_scores.npz',prediction=grid[:,None],targets=np.asarray([spec['target']]))
        ap,native_hash=native_scores(c,grid,spec['target'])
        status.update(status='complete',run_directory=str(run),smoke_only=cfg.smoke,
            native_prediction_sha256=native_hash,independent_validation_pending=True,**{spec['target']:ap})
        base.write(run/'decision.json',{**status,'all_negative_rows_retained':True,'retained_heads_changed':False,
            'standalone_two_target_mAP_defined':False,'confidence_calibrated':False,'independent_generalization_established':False})
    except Exception as error:
        status.update(status='not_run_budget' if isinstance(error,TimeoutError) else 'failed',
            error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        print(id,status['status'],str(error),flush=True)
    session['results'].append(status); base.write(session['output']/(id+'_status.json'),status)
    pd.DataFrame([{k:v for k,v in r.items() if k!='traceback'} for r in session['results']]).to_csv(session['output']/'suite_summary.csv',index=False)
    return {k:v for k,v in status.items() if k!='traceback'}


def reuse_result(session,id,result):
    assert result['id']==id and result['status'] in ('complete','complete_reused_v1') and not result['smoke_only']
    row={**result,'status':'complete_reused_v1','not_retrained_in_v27':True,'artifacts_origin':'preserved completed v26 outputs'}
    session['results'].append(row); base.write(session['output']/(id+'_status.json'),row); return row


def finish(session):
    expected={s['id'] for s in base.CATALOG if s['suite']==session['suite']}|{id for id,s in SPECS.items() if s['suite']==session['suite']}
    done=[r['id'] for r in session['results'] if r['status'] in ('complete','complete_reused_v1')]
    result={'status':'all_methods_completed_validation_pending' if set(done)==expected else 'partial_coverage',
        'expected_tests':sorted(expected),'attempted_tests':[r['id'] for r in session['results']],'completed_tests':done,
        'newly_fitted_tests':[r['id'] for r in session['results'] if r['status']=='complete'],
        'reused_tests':[r['id'] for r in session['results'] if r['status']=='complete_reused_v1'],
        'missing_tests':sorted(expected-{r['id'] for r in session['results']}),'smoke_only':session['cfg'].smoke,
        'retained_heads_changed':False,'competition_submission_created':False,'automatic_promotion':False,
        'confidence_calibrated':False,'independent_generalization_established':False}
    base.write(session['output']/'suite_completion.json',result); return result
