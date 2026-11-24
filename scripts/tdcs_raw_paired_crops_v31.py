"""Matched M22 follow-ups: deliberately co-sample rare and confusable subtypes.

M23 and S12 share weak Event pretraining and TDCS crop streams. One of four
fine-tune microbatches pairs Walking with a known other-FOG event. M23 changes
only crop sampling versus M22; S12 adds the fixed S11 rank term versus M23.
No evaluation labels, prediction artifacts, inputs or score changes.
"""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

import tdcs_raw_weak_transfer_v29 as weak
import tdcs_raw_walking_rank_v30 as rank

raw=weak.raw
ORIGINAL_BATCH=raw.batch
ORIGINAL_SMOKE=raw.smoke_config
PAIR_SEED_OFFSET=81731
COMMON={**weak.SPEC,'fine_tune_changed':True,'matched_warmup_control':'M22',
    'paired_crops_per_batch':2,'remaining_crops':'the unchanged final 6 M22 natural crops in each batch',
    'pairing':'position 0 Walking; position 1 alternates Turn and StartHesitation',
    'episode_sampling':'uniform subject then uniform contiguous event then uniform interior center, >=3 valid 50Hz positions',
    'pair_rng_seed_offset':PAIR_SEED_OFFSET}
SPECS={'M23':{**COMMON,'suite':'metadata','rank_term_weight':0.0},
       'S12':{**COMMON,'suite':'sensor','matched_rank_control':'M23','rank_term_weight':rank.TERM_WEIGHT,
              'rank_margin_log_odds':rank.MARGIN,'hard_negatives_per_microbatch':rank.HARD_NEGATIVES}}


def source_hash(): return globals().get('SOURCE_SHA256') or raw.data.digest(__file__)


def effective_source_hash():
    pins={'M22_weak_source':weak.effective_source_hash(),'S11_rank_source':rank.source_hash(),'paired_source':source_hash()}
    return hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()


def episode_pools(records):
    pools={target:{} for target in (1,2,3)}
    for record in records:
        if record['domain']!='tdcs': raise ValueError('Only outer fitting TDCS records can be paired')
        for target in pools:
            membership=record['valid']&(record['classes']==target)
            changes=np.flatnonzero(np.diff(membership.astype(np.int8),prepend=0,append=0))
            for left,right in zip(changes[::2],changes[1::2]):
                if right-left<3: continue
                pools[target].setdefault(record['subject'],[]).append((record,int(left+1),int(right-1)))
    if not all(pools.values()): raise ValueError('A fitting fold lacks an eligible subtype event')
    return pools


class PairedBatch:
    def __init__(self,records,cfg,seed):
        if cfg.batch_size!=8 or cfg.micro_batch!=2: raise ValueError('Pairing requires 8 crops / 2 per microbatch')
        self.pools=episode_pools(records)
        self.subjects={target:sorted(values) for target,values in self.pools.items()}
        self.rng=np.random.default_rng(seed+PAIR_SEED_OFFSET)
        self.steps=self.eligible=self.total=self.pairs=0
        self.targeted={1:0,2:0,3:0}

    def crop(self,target,cfg):
        rng=self.rng; subjects=self.subjects[target]
        subject=subjects[int(rng.integers(len(subjects)))]; events=self.pools[target][subject]
        record,low,high=events[int(rng.integers(len(events)))]; center=int(rng.integers(low,high))
        stretch=float(rng.uniform(.8,1.2)); width=cfg.train_length
        coords=center+np.linspace(-width*stretch/2,width*stretch/2,width)
        n=len(record['signal']); idx=np.clip(np.rint(coords).astype(int),0,n-1)
        mask=(coords>=0)&(coords<=n-1)&record['valid'][idx]
        y=record['classes'][idx]; assert np.any(mask&(y==target))
        x=raw.features(record,coords,float(rng.uniform(-.2,.2)),float(rng.uniform(.9,1.1)),rng.normal(0,.001,(width,3)))
        return x,y,mask,(record['id'],record['subject'],center,stretch)

    def __call__(self,records,probability,rng,cfg):
        # Consume the identical natural RNG stream before replacing two crops.
        x,y,mask,trace=ORIGINAL_BATCH(records,probability,rng,cfg)
        other=2 if self.steps%2==0 else 1
        for slot,target in enumerate((3,other)):
            x[slot],y[slot],mask[slot],trace[slot]=self.crop(target,cfg)
            self.targeted[target]+=1
        n=y.reshape(-1,cfg.micro_batch*cfg.train_length); m=mask.reshape(n.shape)
        pos=np.count_nonzero(m&(n==3),axis=1); neg=np.count_nonzero(m&((n==1)|(n==2)),axis=1)
        pairs=pos*np.minimum(rank.HARD_NEGATIVES,neg); assert pairs[0]>0
        self.steps+=1; self.total+=len(pairs); self.eligible+=int(np.count_nonzero(pairs)); self.pairs+=int(pairs.sum())
        return x,y,mask,trace

    def receipt(self,seed):
        return {'seed':seed,'pair_rng_seed':seed+PAIR_SEED_OFFSET,'paired_batches':self.steps,
            'eligible_microbatches':self.eligible,'total_microbatches':self.total,'pairs':self.pairs,
            'targeted_centers':{raw.LABELS[k-1]:v for k,v in self.targeted.items()},
            'natural_crops':6*self.steps,'pair_subject_counts':{raw.LABELS[k-1]:len(v) for k,v in self.pools.items()}}


def smoke_config(): return replace(ORIGINAL_SMOKE(),batch_size=8,micro_batch=2)


@contextmanager
def applied(id,usage=None):
    if id not in SPECS: raise ValueError('Unexpected paired-crop test ID')
    with weak.applied():
        prior_source=raw.SOURCE_SHA256; prior_train=raw.train_phase; prior_smoke=raw.smoke_config
        assert rank.METHOD not in raw.SPECS and not set(SPECS)&set(raw.SPECS)

        def phase(model,records,cfg,steps,lr,seed,deadline,ema=None):
            if ema is None: return prior_train(model,records,cfg,steps,lr,seed,deadline)
            sampler=PairedBatch(records,cfg,seed)
            prior_batch=raw.batch; raw.batch=sampler
            try:
                fine=rank.train_fine if id=='S12' else prior_train
                out=fine(model,records,cfg,steps,lr,seed,deadline,ema)
            finally: raw.batch=prior_batch
            row=sampler.receipt(seed); assert row['paired_batches']==steps
            if id=='S12': assert all(row[k]==v for k,v in out['rank_usage'].items())
            if usage is not None: usage.append(row)
            return out

        raw.train_phase=phase; raw.SOURCE_SHA256=effective_source_hash(); raw.smoke_config=smoke_config
        raw.SPECS[id]=SPECS[id]; raw.SPECS[rank.METHOD]=rank.SPEC
        try: yield
        finally:
            raw.train_phase=prior_train; raw.SOURCE_SHA256=prior_source; raw.smoke_config=prior_smoke
            raw.SPECS.pop(id); raw.SPECS.pop(rank.METHOD)


def run(session,id):
    usage=[]
    with applied(id,usage): result=raw.run(session,id)
    if result['status']=='complete':
        destination=Path(result['run_directory'])/'paired_crop_usage.json'
        if len(usage)==5:
            cfg=smoke_config() if session['cfg'].smoke else raw.Config()
            for fold,row in enumerate(usage):
                row['fold']=fold; assert row['seed']==cfg.seed+1009*fold+200000
            raw.base.write(destination,{'id':id,'source_sha256':effective_source_hash(),'folds':usage,
                'specification':SPECS[id],'smoke_only':cfg.smoke,'ranking_used':id=='S12',
                'auxiliary_pretraining_crops_changed':False,'outer_labels_used_for_pairing':False})
        elif not destination.is_file(): raise ValueError('Paired-crop usage missing for reused folds')
    return result


def finish(session,id):
    with applied(id): return raw.finish(session)
