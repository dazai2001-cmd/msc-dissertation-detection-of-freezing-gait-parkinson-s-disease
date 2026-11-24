"""Matched native-rate gated-GRU candidates: 15 features versus 27 features.

Restore the stronger v1 temporal encoder/timing/context recipe, but train all
three targets. The spectral variant adds only label-free rotation-invariant
band-energy features. Reuse immutable v3 recovery utilities, not its network.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

import tdcs_resource_safe_experiment as safe

common,legacy=safe.common,safe.legacy
ROOT=Path(__file__).resolve().parents[1]
TARGETS=legacy.TARGETS
ARCHITECTURES=('native_clock','spectral_clock')
BANDS=((.5,3.),(3.,8.),(8.,16.))
ENERGY_WINDOWS=(128,512)


@dataclass(frozen=True)
class Config:
    name: str='tdcs_native_context_v4'
    targets: tuple=TARGETS
    folds: tuple=(0,1,2,3,4)
    epochs: int=24
    steps_per_epoch: int=144
    batch_size: int=32
    micro_batch: int=4
    train_length: int=1000
    inference_length: int=5000
    inference_batch: int=1
    width: int=64
    hidden: int=128
    gru_layers: int=3
    dropout: float=.15
    learning_rate: float=.001
    weight_decay: float=.0001
    event_fraction: float=.625
    negative_fraction: float=.25
    target_sampling: tuple=(.4,.2,.4)
    rank_weight: float=.1
    ema_decay: float=.995
    checkpoint_steps: int=12
    max_seconds: int=21600
    seed: int=2107


CFG=Config()


def check_config(cfg):
    if tuple(cfg.targets)!=TARGETS or tuple(cfg.folds)!=(0,1,2,3,4):
        raise ValueError('All three targets and all five frozen folds are required')
    if cfg.micro_batch<1 or cfg.batch_size%cfg.micro_batch:
        raise ValueError('Invalid microbatch accumulation')
    if cfg.width%8 or min(cfg.epochs,cfg.steps_per_epoch,cfg.train_length,cfg.inference_length,cfg.checkpoint_steps,cfg.max_seconds)<1:
        raise ValueError('Invalid network or training budget')
    if len(cfg.target_sampling)!=3 or min(cfg.target_sampling)<0 or not np.isclose(sum(cfg.target_sampling),1):
        raise ValueError('Three target-sampling probabilities must sum to one')
    if min(cfg.event_fraction,cfg.negative_fraction)<0 or cfg.event_fraction+cfg.negative_fraction>1:
        raise ValueError('Invalid event/negative/natural mixture')


def spectral_features(raw):
    """Twelve offline energy features; no labels, subject ID or fitted parameters."""
    raw=np.asarray(raw,np.float64)
    if raw.ndim!=2 or raw.shape[1]!=3 or len(raw)<2 or not np.isfinite(raw).all():
        raise ValueError('Finite native accelerometer axes required')
    centered=raw-np.median(raw,axis=0,keepdims=True)
    energies=[]
    for low,high in BANDS:
        sos=butter(2,(low,high),btype='bandpass',fs=128.,output='sos')
        filtered=sosfiltfilt(sos,centered,axis=0,padlen=min(15,len(raw)-1))
        energies.append(np.square(filtered).sum(axis=1))
    energies=np.column_stack(energies)
    parts=[]
    for window in ENERGY_WINDOWS:
        powers=np.maximum(uniform_filter1d(energies,size=window,axis=0,mode='nearest'),0)
        total=powers.sum(axis=1)
        ratio=np.clip(np.log(powers[:,1]+1e-6)-np.log(powers[:,0]+1e-6),-12,12)
        fraction=powers[:,1]/np.maximum(total,1e-6)
        parts.extend([np.log1p(powers),ratio[:,None],fraction[:,None],np.log1p(total)[:,None]])
    result=np.column_stack(parts).astype(np.float32)
    assert result.shape==(len(raw),12) and np.isfinite(result).all()
    return result


def features(raw,architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError('Unknown feature comparison')
    original=common.sensor_features(raw)
    if architecture=='native_clock':
        return original
    return np.column_stack([original,spectral_features(raw)]).astype(np.float32)


def prepare_records(manifest,architecture):
    records=[]
    for number,row in enumerate(manifest.itertuples(index=False),1):
        frame=pd.read_csv(row.path,usecols=[*common.AXES,*TARGETS])
        y=frame.loc[:,list(TARGETS)].to_numpy(np.float32)
        assert np.isin(y,[0,1]).all()
        x=features(frame.loc[:,list(common.AXES)].to_numpy(np.float32),architecture)
        negative=np.flatnonzero(y.sum(axis=1)==0)
        negative=negative[::max(1,len(negative)//2000)]
        records.append({'recording_id':row.recording_id,'subject':row.subject,'fold':row.fold,
            'x':x,'y':y,'negative':negative,'centers':[common.event_centers(y[:,i]) for i in range(3)]})
        if number%100==0:
            print(f'Prepared {number}/{len(manifest)} native records',flush=True)
    assert sum(len(record['y']) for record in records)==6959762
    return records


class Sampler(legacy.Sampler):
    def plan(self,rng):
        draw=rng.random()
        if draw<self.cfg.event_fraction:
            target=int(rng.choice(3,p=self.cfg.target_sampling))
            record=common.EventSampler.choose_record(self.positive[target],rng)
            center=int(rng.choice(record['centers'][target]))
            center+=int(rng.integers(-self.cfg.train_length//5,self.cfg.train_length//5+1))
        elif draw<self.cfg.event_fraction+self.cfg.negative_fraction and self.negative:
            record=common.EventSampler.choose_record(self.negative,rng)
            center=int(rng.choice(record['negative']))
        else:
            record=common.EventSampler.choose_record(self.subjects,rng)
            center=int(rng.integers(len(record['x'])))
        center=int(np.clip(center,0,len(record['x'])-1))
        return record,center-self.cfg.train_length//2

    def batch(self,rng):
        xs,ys,masks=[],[],[]
        for _ in range(self.cfg.batch_size):
            record,start=self.plan(rng)
            x,y,mask=legacy.window(record,start,self.cfg.train_length)
            angle=float(rng.uniform(-.1,.1))
            rotation=np.asarray([[1,0,0],[0,np.cos(angle),-np.sin(angle)],
                                 [0,np.sin(angle),np.cos(angle)]],np.float32)
            for begin in range(0,12,3):
                x[:,begin:begin+3]=x[:,begin:begin+3]@rotation.T
            # Timing and rotation-invariant band powers are not perturbed.
            # No anisotropic gain/cutout is used in either matched candidate.
            xs.append(x)
            ys.append(y)
            masks.append(mask)
        return tuple(torch.from_numpy(np.stack(parts)) for parts in (xs,ys,masks))


class Network(common.TwoTargetGRU):
    def __init__(self,architecture,cfg=CFG):
        if architecture not in ARCHITECTURES:
            raise ValueError('Unknown feature comparison')
        super().__init__(cfg,variant='gated_bigru_rank')
        self.head[-1]=nn.Linear(128,3)
        self.architecture=architecture
        self.input_channels=15 if architecture=='native_clock' else 27
        if self.input_channels==27:
            original=self.input[0]
            # Shared weights/RNG start identically; new feature weights learn
            # from zero rather than perturbing the entire initial network.
            with torch.random.fork_rng(devices=[]):
                replacement=nn.Linear(27,cfg.width)
            with torch.no_grad():
                replacement.weight[:,:15].copy_(original.weight)
                replacement.weight[:,15:].zero_()
                replacement.bias.copy_(original.bias)
            self.input[0]=replacement


def sequence_loss(logits,y,mask,rank_weight):
    value=legacy.masked_loss(logits,y,mask)
    terms=[]
    if rank_weight:
        for target in (0,2):
            valid=mask[...,target]>0
            positive=logits[...,target][valid&(y[...,target]>.5)].float()
            negative=logits[...,target][valid&(y[...,target]<.5)].float()
            if positive.numel() and negative.numel():
                positive=positive[torch.randperm(len(positive),device=positive.device)[:192]]
                negative=torch.topk(negative,min(192,len(negative))).values
                terms.append(F.softplus(negative[None,:]-positive[:,None]).mean())
    if terms:
        value=value+rank_weight*torch.stack(terms).mean()
    return value


def backward_batch(model,batch,cfg,device,scaler):
    total_valid=batch[2].sum().clamp_min(1)
    value=0.
    for start in range(0,len(batch[0]),cfg.micro_batch):
        x,y,mask=[part[start:start+cfg.micro_batch].to(device) for part in batch]
        weight=float(mask.sum().detach().cpu()/total_valid)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            loss=sequence_loss(model(x),y,mask,cfg.rank_weight)*weight
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite native-context loss')
        scaler.scale(loss).backward()
        value+=float(loss.detach().cpu())
    return value


def train_model(records,fold,architecture,cfg,directory,device,fingerprint,health,deadline):
    directory.mkdir(parents=True,exist_ok=True)
    seed=cfg.seed+1009*fold
    torch.manual_seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
    model=Network(architecture,cfg).to(device)
    final=directory/'final_ema.pt'
    if final.is_file():
        state=torch.load(final,map_location='cpu',weights_only=False)
        assert state['fingerprint']==fingerprint and tuple(state['targets'])==TARGETS
        model.load_state_dict(state['model'])
        return model
    sampler=Sampler(records,fold,cfg)
    training=sorted(sampler.subjects)
    heldout=sorted({record['subject'] for record in records if record['fold']==fold})
    assert set(training).isdisjoint(heldout)
    common.write_json(directory/'training_receipt.json',{'targets':TARGETS,'architecture':architecture,
        'training_subjects':training,'held_out_subjects':heldout,'fixed_epochs':cfg.epochs,
        'outer_labels_used_for_checkpoint_selection':False,'effective_batch':cfg.batch_size,
        'micro_batch':cfg.micro_batch,'gradient_accumulation':cfg.batch_size//cfg.micro_batch,
        'data_parallel':False,'training_window':cfg.train_length,'inference_window':cfg.inference_length,
        'input_features':model.input_channels,'target_sampling':cfg.target_sampling,
        'previous_models_overwritten':False,'device':str(device),'seed':seed})
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step:common.learning_rate_multiplier(step,cfg.epochs*cfg.steps_per_epoch))
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    rng=np.random.default_rng(seed)
    resume,history,first_epoch,first_step,partial=directory/'resume.pt',[],0,0,0.
    if resume.is_file():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        assert state['fingerprint']==fingerprint and tuple(state['targets'])==TARGETS
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        rng.bit_generator.state=state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda':
            torch.cuda.set_rng_state(state['cuda_rng'],device)
        first_epoch,first_step,partial,history=state['next_epoch'],state['next_step'],state['partial_loss_sum'],state['history']
    for epoch in range(first_epoch,cfg.epochs):
        begin=first_step if epoch==first_epoch else 0
        loss_sum=partial if epoch==first_epoch else 0.
        started=time.monotonic()
        model.train()
        print(f'{architecture} fold {fold} epoch {epoch+1}/{cfg.epochs}',flush=True)
        for step in range(begin,cfg.steps_per_epoch):
            if time.monotonic()>=deadline:
                raise TimeoutError('Six-hour safety budget reached; latest checkpoint retained')
            health.emit('training',fold=fold,epoch=epoch+1,step=step+1)
            optimizer.zero_grad(set_to_none=True)
            loss_sum+=backward_batch(model,sampler.batch(rng),cfg,device,scaler)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),2)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            with torch.no_grad():
                for destination,source in zip(ema.parameters(),model.parameters()):
                    destination.lerp_(source.detach(),1-cfg.ema_decay)
            last=step+1==cfg.steps_per_epoch
            if last:
                history.append({'epoch':epoch+1,'loss':loss_sum/cfg.steps_per_epoch,
                    'seconds':time.monotonic()-started,'learning_rate':scheduler.get_last_lr()[0]})
                pd.DataFrame(history).to_csv(directory/'training_log.csv',index=False)
            if last or (step+1)%cfg.checkpoint_steps==0:
                health.emit('checkpoint',fold=fold,epoch=epoch+1,step=step+1)
                safe.atomic_checkpoint({'fingerprint':fingerprint,'targets':TARGETS,
                    'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
                    'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                    'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                    'next_epoch':epoch+1 if last else epoch,'next_step':0 if last else step+1,
                    'partial_loss_sum':0. if last else loss_sum,'history':history},resume)
                common.write_json(directory.parent/'progress.json',{'stage':'training','architecture':architecture,
                    'fold':fold,'epoch':epoch+1,'step':step+1,'fixed_epochs':cfg.epochs})
    safe.atomic_checkpoint({'fingerprint':fingerprint,'targets':TARGETS,'model':ema.state_dict()},final)
    return ema


def run_experiment(architecture,cfg=CFG,output_root=None,synthetic_records=None):
    check_config(cfg)
    if architecture not in ARCHITECTURES:
        raise ValueError('Unknown feature comparison')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if synthetic_records is None and device.type!='cuda':
        raise RuntimeError('Full training requires CUDA; no silent synthetic fallback')
    manifest=None
    if synthetic_records is None:
        raw,metadata=common.locate_data()
        manifest=common.build_manifest(raw,metadata)
        population=manifest[['recording_id','subject','fold','source_bytes']].to_csv(index=False,lineterminator='\n')
    else:
        population='synthetic wiring only; not TDCS performance'
    def digest(module,override):
        return globals().get(override) or hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    configuration={'configuration':asdict(cfg),'architecture':architecture,'targets':TARGETS,
        'input_features':15 if architecture=='native_clock' else 27,
        'source_sha256':globals().get('SOURCE_SHA256') or hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'common_source_sha256':digest(common,'COMMON_SOURCE_SHA256'),
        'legacy_source_sha256':digest(legacy,'LEGACY_SOURCE_SHA256'),
        'recovery_source_sha256':digest(safe,'RECOVERY_SOURCE_SHA256'),
        'population_sha256':hashlib.sha256(population.encode()).hexdigest(),
        'smoke_only':synthetic_records is not None,
        'runtime':{'torch':str(torch.__version__),'numpy':np.__version__,
            'visible_gpu_count':torch.cuda.device_count(),'used_gpu_indices':[0] if device.type=='cuda' else []},
        'augmentation':'matched small rotations only; timing and invariant band energies unchanged',
        'ranking':'train-only hard-negative ranking for SH/Walking, computed within microbatches',
        'checkpoint_selection':'fixed final EMA; no outer-label epoch selection',
        'inference_mode':'offline; native 128-Hz GRU, 1000-row training and 5000-row inference as the stronger v1 recipe',
        'matched_comparison':'only the 12 additional band-energy features and their zero-initialized input weights differ',
        'retained_heads_overwritten':False,'competition_submission_created':False}
    fingerprint=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()[:20]
    output_root=Path(output_root or ROOT/'results/tdcs_native_context')
    run=output_root/fingerprint
    run.mkdir(parents=True,exist_ok=True)
    common.write_json(run/'configuration.json',configuration)
    if manifest is not None:
        manifest.to_csv(run/'frozen_tdcs_manifest.csv',index=False)
    (run/'oof').mkdir(exist_ok=True)
    deadline=time.monotonic()+cfg.max_seconds
    with safe.ResourceLog(run,device) as health:
        health.emit('preparing_records')
        records=prepare_records(manifest,architecture) if synthetic_records is None else synthetic_records
        rows,labels,predictions=[],[],[]
        for fold in cfg.folds:
            model=train_model(records,fold,architecture,cfg,run/f'fold_{fold}',device,fingerprint,health,deadline)
            ys,ps=[],[]
            for record in records:
                if record['fold']!=fold:
                    continue
                if time.monotonic()>=deadline:
                    raise TimeoutError('Six-hour safety budget reached during inference')
                health.emit('inference',fold=fold,recording_id=record['recording_id'])
                inference_cfg=replace(safe.CFG,window=cfg.inference_length,inference_batch=cfg.inference_batch)
                pred=legacy.predict_record(model,record['x'],inference_cfg,device)
                assert pred.shape==record['y'].shape
                np.savez_compressed(run/'oof'/f'{record["recording_id"]}.npz',prediction=pred,targets=np.asarray(TARGETS))
                ys.append(record['y'])
                ps.append(pred)
            rows.append({'fold':fold,**legacy.score(np.concatenate(ys),np.concatenate(ps))})
            pd.DataFrame(rows).to_csv(run/'fold_metrics.csv',index=False)
            labels.extend(ys)
            predictions.extend(ps)
            del model
            if device.type=='cuda':
                torch.cuda.empty_cache()
        metrics=legacy.score(np.concatenate(labels),np.concatenate(predictions))
        pd.DataFrame([metrics]).to_csv(run/'native_oof_metrics.csv',index=False)
        result={'run_dir':str(run),'status':'complete','architecture':architecture,**metrics,
            'smoke_only':synthetic_records is not None,'native_rows':sum(len(y) for y in labels),
            'goal_two_target_mAP':.6,'goal_reached':metrics['two_target_mAP']>=.6 and synthetic_records is None,
            'retained_heads_unchanged':True,'independent_generalization_established':False,
            'competition_submission_created':False}
        common.write_json(run/'decision.json',result)
        common.write_json(run/'progress.json',result)
        common.write_json(output_root/'latest_run.json',result)
        health.emit('complete',native_rows=result['native_rows'])
    print(json.dumps(result,indent=2),flush=True)
    return result


def smoke_experiment(architecture,output_root):
    cfg=replace(CFG,name='synthetic_native_context_wiring',epochs=1,steps_per_epoch=2,
        batch_size=4,micro_batch=2,train_length=64,inference_length=128,
        width=16,hidden=8,gru_layers=1,checkpoint_steps=1)
    rng,records=np.random.default_rng(1),[]
    for fold in range(5):
        x=features(rng.normal(size=(96+fold,3)).astype(np.float32),architecture)
        y=np.zeros((len(x),3),np.float32)
        for target,start in enumerate((12,35,62)):
            y[start:start+12,target]=1
        records.append({'recording_id':f'smoke_{fold}','subject':f'synthetic_{fold}','fold':fold,
            'x':x,'y':y,'negative':np.flatnonzero(y.sum(1)==0),
            'centers':[common.event_centers(y[:,i]) for i in range(3)]})
    return run_experiment(architecture,cfg,output_root,records)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture',choices=ARCHITECTURES,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.smoke:
        smoke_experiment(args.architecture,args.output_root)
    else:
        run_experiment(args.architecture,output_root=args.output_root)
