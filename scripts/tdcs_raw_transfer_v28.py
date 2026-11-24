"""Raw-signal v28: matched TDCS warmup versus independent DeFOG pretraining.

Both models use 50-Hz raw accelerometry, CNN/residual BiGRU context and a
four-state training objective. Only SH/Walking are evaluated. No validation
labels select epochs; no local real-data fitting or automatic promotion.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import resample_poly
from sklearn.metrics import average_precision_score

import tdcs_suite_data as data
import tdcs_suite_runner as base

LABELS=('StartHesitation','Turn','Walking')
SPECS={'M21':{'suite':'metadata','transfer':False},'S10':{'suite':'sensor','transfer':True}}


@dataclass(frozen=True)
class Config:
    sample_hz:int=50
    train_length:int=1000
    infer_length:int=5000
    batch_size:int=8
    micro_batch:int=2
    conv_width:int=64
    hidden:int=96
    layers:int=3
    dropout:float=.15
    warm_steps:int=512
    fine_steps:int=1536
    warm_lr:float=.001
    fine_lr:float=.0005
    weight_decay:float=.001
    ema_decay:float=.995
    seed:int=2851
    max_seconds:int=21600
    smoke:bool=False


def smoke_config():
    return replace(Config(),train_length=16,infer_length=32,batch_size=2,micro_batch=1,
        conv_width=8,hidden=8,layers=1,dropout=0.,warm_steps=1,fine_steps=1,max_seconds=600,smoke=True)


def source_hash(): return globals().get('SOURCE_SHA256') or data.digest(__file__)
def key(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:20]


def check_config(cfg):
    if cfg.micro_batch<1 or cfg.batch_size<1 or cfg.sample_hz!=50 or cfg.batch_size%cfg.micro_batch or cfg.conv_width%8:
        raise ValueError('Fixed 50-Hz rate, valid microbatch division and GroupNorm width required')
    if min(cfg.train_length,cfg.infer_length,cfg.hidden,cfg.layers,cfg.warm_steps,cfg.fine_steps)<1:
        raise ValueError('Positive training sizes and budgets required')
    if not 0<=cfg.dropout<1 or not 0<cfg.ema_decay<1: raise ValueError('Invalid regularization')


def boolean(values):
    values=np.asarray(values)
    if values.dtype.kind=='b': return values.astype(bool)
    if values.dtype.kind in 'iuf' and np.isin(values,[0,1]).all(): return values.astype(bool)
    words=np.char.lower(np.char.strip(values.astype(str)))
    if not np.isin(words,['true','false','0','1']).all(): raise ValueError('Invalid annotation mask values')
    return np.isin(words,['true','1'])


def make_record(frame,id,subject,fold,source_hz,domain):
    if (domain,source_hz) not in (('tdcs',128),('defog',100)):
        raise ValueError('Only the documented TDCS/DeFOG unit-rate contracts are supported')
    times=frame.Time.to_numpy(float)
    if not np.isfinite(times).all() or not np.equal(np.diff(times),1).all():
        raise ValueError('Consecutive sample-index Time is required')
    raw=frame[list(data.AXES)].to_numpy(np.float64)
    if not np.isfinite(raw).all(): raise ValueError('Nonfinite accelerometer input')
    labels=frame[list(LABELS)].to_numpy()
    if not np.isin(labels,[0,1]).all() or (labels.sum(1)>1).any(): raise ValueError('Exclusive binary class labels required')
    classes=(labels@np.asarray([1,2,3])).astype(np.uint8)
    valid=np.ones(len(frame),bool) if domain=='tdcs' else boolean(frame.Valid)&boolean(frame.Task)
    raw_g=raw/9.80665 if domain=='tdcs' else raw
    signal=resample_poly(raw_g,25,64,axis=0,padtype='line') if source_hz==128 else resample_poly(raw_g,1,2,axis=0,padtype='line')
    if len(signal)<2: raise ValueError('Recording is too short after resampling')
    positions=np.arange(len(signal),dtype=np.float64)*source_hz/50.
    nearest=np.clip(np.rint(positions).astype(np.int64),0,len(frame)-1)
    signal=signal.astype(np.float32)
    record={'id':str(id),'subject':str(subject),'fold':int(fold),'domain':domain,'signal':signal,
        'classes':classes[nearest],'valid':valid[nearest],'native_length':len(frame),'source_hz':source_hz,
        'mean':signal.mean(0,dtype=np.float64),'cov':np.cov(signal.astype(np.float64),rowvar=False,bias=True)}
    record['valid_indices']=np.flatnonzero(record['valid']).astype(np.int32)
    return record


def features(record,positions,angle=0.,gain=1.,noise=None):
    positions=np.asarray(positions,np.float64); n=len(record['signal'])
    clipped=np.clip(positions,0,n-1); left=np.floor(clipped).astype(np.int64); right=np.minimum(left+1,n-1)
    fraction=(clipped-left)[:,None]
    raw=(1-fraction)*record['signal'][left]+fraction*record['signal'][right]
    cosine,sine=np.cos(angle),np.sin(angle)
    rotation=np.asarray([[1,0,0],[0,cosine,-sine],[0,sine,cosine]],np.float64)
    raw=raw@rotation.T; mean=record['mean']@rotation.T
    std=np.sqrt(np.maximum(np.diag(rotation@record['cov']@rotation.T),.005**2))
    raw=mean+gain*(raw-mean)
    if noise is not None: raw=raw+noise
    z=np.clip((raw-mean)/std,-8,8)
    return np.column_stack([np.clip(raw,-4,4),z,np.minimum(np.linalg.norm(z,axis=1)/np.sqrt(3),8),
        np.clip(positions/max(n-1,1),0,1)]).astype(np.float32)


def class_weights(records):
    counts=sum((np.bincount(r['classes'][r['valid']],minlength=4) for r in records),np.zeros(4,np.int64))
    if counts.sum()==0: raise ValueError('No annotated fitting samples')
    weights=np.clip(np.sqrt(max(int(counts[0]),1)/np.maximum(counts,1)),.5,6.)
    return (weights/weights.mean()).astype(np.float32),counts


def sampling_plan(records):
    records=[r for r in records if len(r['valid_indices'])]
    if not records: raise ValueError('No annotated fitting recordings')
    totals={s:sum(len(r['valid_indices']) for r in records if r['subject']==s) for s in {r['subject'] for r in records}}
    p=np.asarray([len(r['valid_indices'])/np.sqrt(totals[r['subject']]) for r in records]); p/=p.sum()
    return records,p


def batch(records,probability,rng,cfg):
    xs=[]; ys=[]; masks=[]; trace=[]
    for _ in range(cfg.batch_size):
        record=records[int(rng.choice(len(records),p=probability))]
        center=int(rng.choice(record['valid_indices'])); stretch=float(rng.uniform(.8,1.2))
        positions=center+np.linspace(-cfg.train_length*stretch/2,cfg.train_length*stretch/2,cfg.train_length)
        nearest=np.clip(np.rint(positions).astype(int),0,len(record['signal'])-1)
        inside=(positions>=0)&(positions<=len(record['signal'])-1)
        xs.append(features(record,positions,float(rng.uniform(-.2,.2)),float(rng.uniform(.9,1.1)),rng.normal(0,.001,(cfg.train_length,3))))
        ys.append(record['classes'][nearest]); masks.append(inside&record['valid'][nearest])
        trace.append((record['id'],record['subject'],center,stretch))
    return np.stack(xs),np.stack(ys).astype(np.int64),np.stack(masks),trace


class Network(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; width=cfg.conv_width; size=2*cfg.hidden
        self.stem=nn.Sequential(nn.Conv1d(8,width,7,padding=3),nn.GroupNorm(8,width),nn.GELU(),
            nn.Conv1d(width,width,7,padding=3),nn.GroupNorm(8,width),nn.GELU())
        self.project=nn.Sequential(nn.Linear(width,size),nn.LayerNorm(size),nn.GELU())
        self.blocks=nn.ModuleList([nn.GRU(size,cfg.hidden,batch_first=True,bidirectional=True) for _ in range(cfg.layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(size) for _ in range(cfg.layers)])
        self.head=nn.Sequential(nn.Linear(size,64),nn.GELU(),nn.Linear(64,4))

    def forward(self,x):
        h=self.project(self.stem(x.transpose(1,2)).transpose(1,2))
        for block,norm in zip(self.blocks,self.norms):
            h=norm((h+F.dropout(block(h)[0],self.cfg.dropout,self.training))/np.sqrt(2))
        return self.head(h)


def train_phase(model,records,cfg,steps,lr,seed,deadline,ema=None):
    torch.manual_seed(seed)
    if next(model.parameters()).is_cuda: torch.cuda.manual_seed_all(seed)
    device=next(model.parameters()).device; rng=np.random.default_rng(seed); records,p=sampling_plan(records)
    weights,counts=class_weights(records); weight=torch.from_numpy(weights).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg.weight_decay)
    trace=[]; model.train(); started=time.monotonic()
    for step in range(steps):
        if time.monotonic()>=deadline: raise TimeoutError('Raw sequence runtime budget reached')
        bx,by,mask,crops=batch(records,p,rng,cfg); mass=float((weights[by]*mask).sum())
        if mass<=0: raise ValueError('Sampled batch contains no annotated positions')
        trace.extend(crops); optimizer.zero_grad(set_to_none=True)
        for start in range(0,cfg.batch_size,cfg.micro_batch):
            stop=start+cfg.micro_batch; x=torch.from_numpy(bx[start:stop]).to(device)
            y=torch.from_numpy(by[start:stop]).to(device); valid=torch.from_numpy(mask[start:stop]).to(device).float()
            losses=F.cross_entropy(model(x).transpose(1,2),y,weight=weight,reduction='none')
            loss=(losses*valid).sum()/mass
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite raw sequence loss')
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for dst,src in zip(ema.parameters(),model.parameters()): dst.lerp_(src,1-cfg.ema_decay)
        if (step+1)%64==0: print(f'Raw phase: {step+1}/{steps} updates; elapsed {time.monotonic()-started:.1f}s',flush=True)
    return {'updates':steps,'seed':seed,'class_counts':counts.tolist(),'class_weights':weights.tolist(),
        'subjects':sorted({r['subject'] for r in records}),'elapsed_seconds':time.monotonic()-started,'trace':trace}


def predict(model,record,cfg,device):
    n=len(record['signal']); total=np.zeros((n,2),np.float64); denominator=np.zeros(n,np.float64)
    taper=np.maximum(np.hanning(cfg.infer_length),.1); model.eval()
    with torch.no_grad():
        for start in range(-cfg.infer_length//2,n,cfg.infer_length//2):
            positions=start+np.arange(cfg.infer_length); x=features(record,positions)
            p=torch.softmax(model(torch.from_numpy(x[None]).to(device)),dim=-1)[0][:,[1,3]].cpu().numpy()
            inside=(positions>=0)&(positions<n); rows=positions[inside]
            total[rows]+=p[inside]*taper[inside,None]; denominator[rows]+=taper[inside]
    assert (denominator>0).all()
    return (total/denominator[:,None]).astype(np.float32)


def synthetic():
    main=[]; aux=[]; native=[]
    for i in range(7):
        hz=128 if i<5 else 100; n=128+i*32; t=np.arange(n)/hz
        frame=pd.DataFrame({'Time':np.arange(n),'AccV':-1+.1*np.sin(2*np.pi*t),
            'AccML':.2*np.cos(6*np.pi*t),'AccAP':.2*np.sin(4*np.pi*t)})
        if i<5: frame[list(data.AXES)]*=9.80665
        labels=(np.arange(n)//12)%4
        for j,name in enumerate(LABELS,1): frame[name]=(labels==j).astype(np.uint8)
        frame['Valid']=np.arange(n)>4; frame['Task']=np.arange(n)<n-4
        record=make_record(frame,f'toy_{i}',f'subject_{i}',i if i<5 else -1,hz,'tdcs' if i<5 else 'defog')
        (main if i<5 else aux).append(record)
        if i<5: native.append(frame[['StartHesitation','Walking']].to_numpy(np.uint8))
    return main,aux,np.concatenate(native)


def prepare(context,root,transfer):
    cache=Path(root)/'raw_cache'; cache.mkdir(parents=True,exist_ok=True); main=[]; aux=[]; provenance=[]
    for i,row in enumerate(context['manifest'].itertuples(index=False)):
        frame=pd.read_csv(row.path,usecols=['Time',*data.AXES,*LABELS])
        left,right=context['native_offsets'][i:i+2]
        np.testing.assert_array_equal(frame[['StartHesitation','Walking']].to_numpy(),context['native_y'][left:right])
        assert len(frame)==right-left
        record=make_record(frame,row.recording_id,row.subject,row.fold,128,'tdcs'); main.append(record)
        provenance.append({'id':row.recording_id,'subject':row.subject,'fold':row.fold,'domain':'tdcs',
            'source_sha256':data.digest(row.path),'signal_sha256':data.array_digest(record['signal']),
            'class_sha256':data.array_digest(record['classes']),'valid_sha256':data.array_digest(record['valid'])})
        np.savez_compressed(cache/(row.recording_id+'.npz'),signal=record['signal'],classes=record['classes'],valid=record['valid'])
    tdcs_metadata=pd.read_csv(data.csv_path('tdcsfog_metadata.csv'),dtype=str)
    if tdcs_metadata.Subject.isna().any(): raise ValueError('Missing TDCS subject for auxiliary exclusion')
    excluded=set(tdcs_metadata.Subject)
    if transfer:
        metadata_path=data.csv_path('defog_metadata.csv'); meta=pd.read_csv(metadata_path,dtype=str)
        if meta.Id.duplicated().any() or meta.Id.isna().any() or meta.Subject.isna().any(): raise ValueError('Invalid auxiliary metadata')
        raw_root=data.frozen.shared.common.locate_data()[0].parent/'defog'
        for row in meta.itertuples(index=False):
            if row.Subject in excluded: continue
            path=raw_root/(row.Id+'.csv')
            frame=pd.read_csv(path,usecols=['Time',*data.AXES,*LABELS,'Valid','Task'])
            record=make_record(frame,row.Id,row.Subject,-1,100,'defog'); aux.append(record)
            provenance.append({'id':row.Id,'subject':row.Subject,'domain':'defog','source_sha256':data.digest(path),
                'signal_sha256':data.array_digest(record['signal']),'class_sha256':data.array_digest(record['classes']),
                'valid_sha256':data.array_digest(record['valid']),'valid_rows':int(record['valid'].sum())})
            np.savez_compressed(cache/(row.Id+'.npz'),signal=record['signal'],classes=record['classes'],valid=record['valid'])
        if not aux or set(r['subject'] for r in aux)&excluded: raise ValueError('No independent auxiliary records')
    base.write(Path(root)/'raw_dataset_manifest.json',{'records':provenance,'sample_hz':50,
        'auxiliary_TDCS_subject_overlap':0,'auxiliary_loss_mask':'Valid AND Task','clinical_or_ID_predictors':False})
    print('Prepared raw data:',len(main),'TDCS records;',len(aux),'auxiliary records',flush=True)
    return main,aux,context['native_y']


def evaluate(main,native_y,parts,gates,apply_gate):
    if not len(main)==len(parts)==len(gates): raise ValueError('Recording prediction alignment mismatch')
    predictions=[]
    for record,p,q in zip(main,parts,gates):
        if p.shape!=(len(record['signal']),2): raise ValueError('Unexpected raw prediction length or heads')
        grid=np.arange(len(p))*record['source_hz']/50.
        native=np.column_stack([np.interp(np.arange(record['native_length']),grid,p[:,j]) for j in range(2)]).astype(np.float32)
        if apply_gate: native[:,1]*=(np.asarray([q],np.float32)*.95+.05)[0]
        predictions.append(native)
    p=np.concatenate(predictions)
    assert p.shape==native_y.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    values={t:float(average_precision_score(native_y[:,j],p[:,j])) for j,t in enumerate(data.TARGETS)}
    values['two_target_mAP']=float(np.mean(list(values.values())))
    return values,data.array_digest(p)


def run(session,id):
    spec=SPECS[id]; assert spec['suite']==session['suite']; cfg=smoke_config() if session['cfg'].smoke else Config()
    status={'id':id,'title':'Raw 50-Hz CNN/residual GRU '+('DeFOG transfer' if spec['transfer'] else 'TDCS matched control'),
        'suite':session['suite'],'status':'pending'}
    try:
        check_config(cfg)
        if not cfg.smoke and (not Path('/kaggle/working').exists() or not torch.cuda.is_available()):
            raise RuntimeError('Real raw fitting requires Kaggle CUDA')
        root=session['output']/'raw_v28'; root.mkdir(parents=True,exist_ok=True)
        main,aux,y=synthetic() if cfg.smoke else prepare(session['context'],root,spec['transfer'])
        signature=[{'id':r['id'],'subject':r['subject'],'fold':r['fold'],'domain':r['domain'],
            'signal':data.array_digest(r['signal']),'classes':data.array_digest(r['classes']),'valid':data.array_digest(r['valid'])} for r in [*main,*aux]]
        config={'specification':spec,'id':id,'configuration':asdict(cfg),'source_sha256':source_hash(),
            'base_sources':base.source_hashes(),'dataset_signature':signature,'targets':list(data.TARGETS),
            'training_classes':['NoFOG',*LABELS],'outer_stopping':False,'automatic_promotion':False,
            'primary_postprocessing':'ungated','secondary_postprocessing':'frozen_Walking_gate_once',
            'runtime':session['runtime'],'smoke_only':cfg.smoke,'warm_model_initialization_seed':cfg.seed}
        run_dir=session['output']/'methods'/id/key(config); run_dir.mkdir(parents=True,exist_ok=True); base.write(run_dir/'configuration.json',config)
        device=torch.device('cpu' if cfg.smoke else 'cuda:0'); torch.set_num_threads(2)
        deadline=min(session['deadline'],time.monotonic()+cfg.max_seconds)
        initial=None; warm_receipt=None
        if spec['transfer']:
            assert set(r['subject'] for r in aux).isdisjoint({r['subject'] for r in main})
            if (run_dir/'defog_pretrained.pt').is_file():
                stored=torch.load(run_dir/'defog_pretrained.pt',map_location='cpu',weights_only=False)
                assert stored['source_sha256']==source_hash() and stored['configuration']==asdict(cfg)
                assert stored['updates']==cfg.warm_steps
                assert set(stored['training_subjects'])=={r['subject'] for r in aux if r['valid'].any()}
                initial=stored['model']; warm_receipt=json.loads((run_dir/'auxiliary_pretraining_receipt.json').read_text())
            else:
                torch.manual_seed(cfg.seed); model=Network(cfg).to(device)
                warm_receipt=train_phase(model,aux,cfg,cfg.warm_steps,cfg.warm_lr,cfg.seed+999,deadline)
                torch.save({'model':model.state_dict(),'configuration':asdict(cfg),'source_sha256':source_hash(),
                    'training_subjects':warm_receipt['subjects'],'updates':cfg.warm_steps},run_dir/'defog_pretrained.pt')
                initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; del model
                base.write(run_dir/'auxiliary_pretraining_receipt.json',{k:v for k,v in warm_receipt.items() if k!='trace'})
                base.write(run_dir/'auxiliary_pretraining_trace.json',{'crops':warm_receipt['trace']})
        parts=[None]*len(main); completed=[]
        for fold in range(5):
            if time.monotonic()>=deadline: raise TimeoutError('Raw suite budget reached between folds')
            directory=run_dir/f'fold_{fold}'; directory.mkdir(exist_ok=True)
            fitting=[r for r in main if r['fold']!=fold]; heldout=[i for i,r in enumerate(main) if r['fold']==fold]
            assert {r['subject'] for r in fitting}.isdisjoint({main[i]['subject'] for i in heldout})
            receipt_path=directory/'training_receipt.json'
            if receipt_path.is_file():
                receipt=json.loads(receipt_path.read_text()); assert receipt['fingerprint']==run_dir.name
                for name,sha in receipt['artifact_sha256'].items(): assert data.digest(directory/name)==sha
                with np.load(directory/'heldout_grid.npz',allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['record_indices'],heldout)
                    np.testing.assert_array_equal(saved['offsets'],np.r_[0,np.cumsum([len(main[i]['signal']) for i in heldout])])
                    for k,i in enumerate(saved['record_indices']): parts[int(i)]=saved['prediction'][saved['offsets'][k]:saved['offsets'][k+1]]
                completed.append(fold); continue
            if any(directory.iterdir()): raise ValueError('Incomplete raw fold; preserve it and use a fresh output root')
            torch.manual_seed(cfg.seed); model=Network(cfg).to(device)
            if initial is not None: model.load_state_dict(initial)
            else: warm_receipt=train_phase(model,fitting,cfg,cfg.warm_steps,cfg.warm_lr,cfg.seed+1009*fold+999,deadline)
            torch.save({'model':model.state_dict(),'warm_steps':cfg.warm_steps,'fingerprint':run_dir.name},directory/'warmup.pt')
            ema=copy.deepcopy(model).eval()
            fine=train_phase(model,fitting,cfg,cfg.fine_steps,cfg.fine_lr,cfg.seed+1009*fold+200000,deadline,ema)
            torch.save({'model':ema.state_dict(),'configuration':asdict(cfg),'fingerprint':run_dir.name,
                'fold':fold,'classes':['NoFOG',*LABELS]},directory/'model.pt')
            payload=torch.load(directory/'model.pt',map_location=device,weights_only=False)
            replay_model=Network(cfg).to(device); replay_model.load_state_dict(payload['model']); replay_model.eval()
            assert all(torch.equal(value,payload['model'][name]) for name,value in ema.state_dict().items())
            base.write(directory/'training_trace.json',{'warmup':warm_receipt['trace'] if not spec['transfer'] else 'shared_independent_DeFOG',
                'fine_tune':fine['trace']})
            for i in heldout: parts[i]=predict(replay_model,main[i],cfg,device)
            offsets=np.r_[0,np.cumsum([len(parts[i]) for i in heldout])]
            np.savez_compressed(directory/'heldout_grid.npz',record_indices=np.asarray(heldout),offsets=offsets,
                prediction=np.concatenate([parts[i] for i in heldout]))
            receipt={'fingerprint':run_dir.name,'fold':fold,'fitting_subjects':sorted({r['subject'] for r in fitting}),
                'heldout_subjects':sorted({main[i]['subject'] for i in heldout}),'warmup_subjects':warm_receipt['subjects'],
                'warmup_updates':cfg.warm_steps,'fine_tune_updates':cfg.fine_steps,'fine_tune_class_counts':fine['class_counts'],
                'fine_tune_class_weights':fine['class_weights'],'fixed_final_EMA':cfg.ema_decay,'backend':str(device),
                'warmup_elapsed_seconds':warm_receipt['elapsed_seconds'],'fine_tune_elapsed_seconds':fine['elapsed_seconds'],
                'outer_labels_used_for_training_or_stopping':False,'auxiliary_valid_mask_applied':spec['transfer'],
                'saved_model_reloaded_for_all_heldout_predictions':True,
                'artifact_sha256':{p.name:data.digest(p) for p in directory.iterdir() if p.is_file() and p.name!='training_receipt.json'}}
            assert not set(receipt['warmup_subjects'])&set(receipt['heldout_subjects'])
            base.write(receipt_path,receipt); completed.append(fold); print(id,'fold',fold,'complete',flush=True)
            del model,ema,replay_model,payload
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        offsets=np.r_[0,np.cumsum([len(p) for p in parts])]
        np.savez_compressed(run_dir/'grid_scores.npz',prediction=np.concatenate(parts),offsets=offsets,
            recording_ids=np.asarray([r['id'] for r in main]),native_lengths=np.asarray([r['native_length'] for r in main]),
            targets=np.asarray(data.TARGETS))
        gates=np.ones(len(main),np.float32) if cfg.smoke else session['context']['gates']
        values,native_hash=evaluate(main,y,parts,gates,False); gated,gated_hash=evaluate(main,y,parts,gates,True)
        status.update(status='complete',**values,run_directory=str(run_dir),smoke_only=cfg.smoke,
            native_prediction_sha256=native_hash,independent_validation_pending=True)
        base.write(run_dir/'decision.json',{**status,'fixed_gate_ablation':gated,'fixed_gate_native_sha256':gated_hash,
            'all_negative_native_rows_retained':True,'native_rows':len(y),'retained_heads_changed':False,
            'independent_generalization_established':False,'confidence_calibrated':False})
    except Exception as error:
        status.update(status='not_run_budget' if isinstance(error,TimeoutError) else 'failed',
            error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        print(id,status['status'],str(error),flush=True)
    session['results'].append(status); base.write(session['output']/(id+'_status.json'),status)
    pd.DataFrame([{k:v for k,v in r.items() if k!='traceback'} for r in session['results']]).to_csv(session['output']/'suite_summary.csv',index=False)
    return {k:v for k,v in status.items() if k!='traceback'}


def reuse_result(session,id,result):
    assert result['id']==id and result['status'] in ('complete','complete_reused_v1') and not result['smoke_only']
    row={**result,'status':'complete_reused','not_retrained_in_v28':True,
        'artifacts_origin':result['run_directory'],'reused_scores_are_not_a_new_fit':True}
    session['results'].append(row); base.write(session['output']/(id+'_status.json'),row)
    return row


def finish(session):
    import tdcs_v27_followups as earlier
    expected={s['id'] for s in base.CATALOG if s['suite']==session['suite']}
    expected|={id for id,s in earlier.SPECS.items() if s['suite']==session['suite']}
    expected|={id for id,s in SPECS.items() if s['suite']==session['suite']}
    done=[r['id'] for r in session['results'] if r['status'] in ('complete','complete_reused')]
    assert len({r['id'] for r in session['results']})==len(session['results'])
    result={'status':'all_methods_completed_validation_pending' if set(done)==expected else 'partial_coverage',
        'expected_tests':sorted(expected),'attempted_tests':[r['id'] for r in session['results']],
        'completed_tests':done,'newly_fitted_tests':[r['id'] for r in session['results'] if r['status']=='complete'],
        'reused_tests':[r['id'] for r in session['results'] if r['status']=='complete_reused'],
        'missing_tests':sorted(expected-set(done)),'smoke_only':session['cfg'].smoke,
        'retained_heads_changed':False,'competition_submission_created':False,'automatic_promotion':False,
        'confidence_calibrated':False,'independent_generalization_established':False}
    base.write(session['output']/'suite_completion.json',result)
    pd.DataFrame([{k:v for k,v in r.items() if k!='traceback'} for r in session['results']]).to_csv(session['output']/'suite_summary.csv',index=False)
    return result
