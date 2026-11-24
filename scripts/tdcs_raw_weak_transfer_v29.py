"""M22: typed DeFOG + weak binary Event pretraining, then unchanged TDCS fits.

No subtype is invented for train/notype. Event=1 contributes -log P(any FOG),
Event=0 contributes the known NoFOG loss. Valid AND Task masks both. Compare
typed-only S10 at the same seeds/architecture/512 warmup/1536 fine-tune updates.
No real-data local fitting, test-file use, metric changes or automatic promotion.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import tdcs_raw_transfer_v28 as raw
import tdcs_raw_transfer_v28_recovery as typed

UNKNOWN_FOG=255
ORIGINAL_TRAIN=raw.train_phase
ORIGINAL_SYNTHETIC=raw.synthetic
SPEC={'suite':'metadata','transfer':True,'weak_notype_Event_pretraining':True,
    'unknown_subtype_positive_loss':'negative_log_sum_probability_of_SH_Turn_Walking',
    'unknown_subtype_positive_weight':'mean of three fitting FOG-class weights',
    'class_weights':'fitting known-state counts only; weak positives never assigned subtypes',
    'fine_tune_changed':False,'warmup_updates':512,'fine_tune_updates':1536}


def source_hash(): return globals().get('SOURCE_SHA256') or raw.data.digest(__file__)


def effective_source_hash():
    return hashlib.sha256(json.dumps({'frozen_raw_source':typed.FROZEN_RAW_SHA256,
        'typed_file_reader':typed.source_hash(),'weak_Event_training':source_hash()},sort_keys=True).encode()).hexdigest()


def make_weak_record(frame,id,subject):
    if not {'Event','Valid','Task'}.issubset(frame.columns): raise ValueError('Weak Event annotation schema missing')
    events=raw.boolean(frame.Event.to_numpy())
    # Zero columns are only a temporary adapter for shared sensor preprocessing.
    # They are replaced by the binary annotation, never used as subtype labels.
    adapted=frame.copy()
    for target in raw.LABELS: adapted[target]=0
    record=raw.make_record(adapted,id,subject,-1,100,'defog')
    nearest=np.clip(np.rint(np.arange(len(record['signal']))*2).astype(int),0,len(frame)-1)
    record['classes']=np.where(events[nearest],UNKNOWN_FOG,0).astype(np.uint8)
    record['label_kind']='weak_binary_Event'
    return record


def weak_weights(records):
    counts=np.zeros(4,np.int64); unknown=0
    for record in records:
        values=record['classes'][record['valid']]
        if not np.isin(values,[0,1,2,3,UNKNOWN_FOG]).all(): raise ValueError('Unsupported partial label')
        counts+=np.bincount(values[values<4],minlength=4)
        unknown+=int((values==UNKNOWN_FOG).sum())
    if counts.sum()+unknown==0: raise ValueError('No annotated auxiliary rows')
    weights=np.clip(np.sqrt(max(int(counts[0]),1)/np.maximum(counts,1)),.5,6.)
    weights=(weights/weights.mean()).astype(np.float32)
    return weights,counts,unknown,float(weights[1:].mean())


def partial_losses(logits,labels,mask,weight,fog_weight):
    """Unnormalized losses and weighted mass; differentiable without pseudo-types."""
    if not torch.all(((labels>=0)&(labels<=3))|(labels==UNKNOWN_FOG)): raise ValueError('Unknown partial training state')
    logp=F.log_softmax(logits,dim=-1); weak=labels==UNKNOWN_FOG
    fitting_weight=torch.where(weak,torch.as_tensor(fog_weight,device=logits.device),weight[labels.clamp_max(3)])
    typed_loss=-logp.gather(-1,labels.clamp_max(3).unsqueeze(-1)).squeeze(-1)
    weak_loss=-torch.logsumexp(logp[...,1:4],dim=-1)
    loss=torch.where(weak,weak_loss,typed_loss)*fitting_weight*mask
    return loss.sum(),(fitting_weight*mask).sum()


def train_phase(model,records,cfg,steps,lr,seed,deadline,ema=None):
    if not any(r.get('label_kind')=='weak_binary_Event' or np.any(r['classes']==UNKNOWN_FOG) for r in records):
        return ORIGINAL_TRAIN(model,records,cfg,steps,lr,seed,deadline,ema)
    if ema is not None: raise ValueError('Weak Event loss is warmup-only; TDCS fine-tune must remain unchanged')
    torch.manual_seed(seed)
    if next(model.parameters()).is_cuda: torch.cuda.manual_seed_all(seed)
    device=next(model.parameters()).device; rng=np.random.default_rng(seed)
    records,p=raw.sampling_plan(records); weights,counts,unknown,fog_weight=weak_weights(records)
    weight=torch.from_numpy(weights).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg.weight_decay)
    trace=[]; model.train(); started=time.monotonic()
    for step in range(steps):
        if time.monotonic()>=deadline: raise TimeoutError('Weak pretraining runtime budget reached')
        bx,by,mask,crops=raw.batch(records,p,rng,cfg); trace.extend(crops)
        row_weights=np.where(by==UNKNOWN_FOG,fog_weight,weights[np.minimum(by,3)])
        mass=float((row_weights*mask).sum())
        if mass<=0: raise ValueError('No annotated sampled auxiliary positions')
        optimizer.zero_grad(set_to_none=True)
        for start in range(0,cfg.batch_size,cfg.micro_batch):
            stop=start+cfg.micro_batch; x=torch.from_numpy(bx[start:stop]).to(device)
            labels=torch.from_numpy(by[start:stop]).to(device); valid=torch.from_numpy(mask[start:stop]).to(device).float()
            numerator,_=partial_losses(model(x),labels,valid,weight,fog_weight)
            loss=numerator/mass
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite weak pretraining loss')
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        if (step+1)%64==0: print('Weak Event warmup:',step+1,'/',steps,'; elapsed',round(time.monotonic()-started,1),flush=True)
    return {'updates':steps,'seed':seed,'class_counts':counts.tolist(),'class_weights':weights.tolist(),
        'unknown_subtype_positive_rows':unknown,'unknown_subtype_positive_weight':fog_weight,
        'subjects':sorted({r['subject'] for r in records}),'elapsed_seconds':time.monotonic()-started,
        'trace':trace,'loss':'typed CE plus binary Event partial-state likelihood; no invented subtype labels'}


def prepare(context,root,transfer):
    if not transfer: raise ValueError('M22 requires the declared auxiliary pretraining stage')
    main,aux,y=typed.prepare(context,root,True)
    excluded=set(pd.read_csv(raw.data.csv_path('tdcsfog_metadata.csv'),dtype=str).Subject)
    metadata=pd.read_csv(raw.data.csv_path('defog_metadata.csv'),dtype=str)
    notype_root=raw.data.frozen.shared.common.locate_data()[0].parent/'notype'
    selected,inventory=typed.auxiliary_inventory(metadata,notype_root,excluded,expected_files=46)
    inventory['selection']='actual train/notype CSVs joined to metadata; binary Event supervision only; no test files'
    typed_ids={r['id'] for r in aux}; weak_ids=set(selected.Id)
    if typed_ids&weak_ids: raise ValueError('Typed and weak training IDs overlap')
    available_typed={p.stem for p in notype_root.parent.joinpath('defog').glob('*.csv')}
    available_weak={p.stem for p in notype_root.glob('*.csv')}
    if available_typed|available_weak!=set(metadata.Id): raise ValueError('Training file roles do not cover the metadata inventory')
    root=Path(root); manifest=json.loads((root/'raw_dataset_manifest.json').read_text())
    for row in selected.itertuples(index=False):
        frame=pd.read_csv(row.path,usecols=['Time',*raw.data.AXES,'Event','Valid','Task'])
        record=make_weak_record(frame,row.Id,row.Subject); aux.append(record)
        manifest['records'].append({'id':row.Id,'subject':row.Subject,'domain':'defog_notype',
            'source_sha256':raw.data.digest(row.path),'signal_sha256':raw.data.array_digest(record['signal']),
            'class_sha256':raw.data.array_digest(record['classes']),'valid_sha256':raw.data.array_digest(record['valid']),
            'valid_rows':int(record['valid'].sum()),'positive_partial_Event_rows':int(((record['classes']==UNKNOWN_FOG)&record['valid']).sum())})
        np.savez_compressed(root/'raw_cache'/(row.Id+'.npz'),signal=record['signal'],classes=record['classes'],valid=record['valid'])
    manifest['notype_inventory']=inventory
    manifest['effective_raw_source_sha256']=effective_source_hash()
    manifest['weak_Event_source_sha256']=source_hash()
    manifest['unknown_subtype_positive_encoding']=UNKNOWN_FOG
    manifest['unknown_subtype_positive_is_a_new_FOG_class']=False
    raw.base.write(root/'raw_dataset_manifest.json',manifest)
    print('Auxiliary pool:',len(aux),'independent typed + weak records; no subtype pseudo-labels',flush=True)
    return main,aux,y


def synthetic():
    main,aux,y=ORIGINAL_SYNTHETIC()
    aux[-1]['classes']=np.where(aux[-1]['classes']>0,UNKNOWN_FOG,0).astype(np.uint8)
    aux[-1]['label_kind']='weak_binary_Event'
    return main,aux,y


@contextmanager
def applied():
    assert raw.source_hash()==typed.FROZEN_RAW_SHA256
    previous={name:raw.__dict__.get(name) for name in ['prepare','train_phase','synthetic','SOURCE_SHA256']}
    had_override='SOURCE_SHA256' in raw.__dict__; previous_spec=raw.SPECS.get('M22')
    raw.prepare=prepare; raw.train_phase=train_phase; raw.synthetic=synthetic
    raw.SOURCE_SHA256=effective_source_hash(); raw.SPECS['M22']=SPEC
    try: yield
    finally:
        for name in ['prepare','train_phase','synthetic']: setattr(raw,name,previous[name])
        if had_override: raw.SOURCE_SHA256=previous['SOURCE_SHA256']
        else: raw.__dict__.pop('SOURCE_SHA256',None)
        if previous_spec is None: raw.SPECS.pop('M22',None)
        else: raw.SPECS['M22']=previous_spec


def run(session,id='M22'):
    if id!='M22' or session['suite']!='metadata': raise ValueError('Only the declared M22 weak-transfer test is supported')
    with applied(): return raw.run(session,id)


def reuse_result(session,id,result):
    assert result['id']==id and result['status'] in ('complete','complete_reused_v1','complete_reused') and not result['smoke_only']
    row={**result,'status':'complete_reused','not_retrained_in_v29':True,'artifacts_origin':result['run_directory']}
    session['results'].append(row); raw.base.write(session['output']/(id+'_status.json'),row); return row


def finish(session):
    with applied(): return raw.finish(session)
