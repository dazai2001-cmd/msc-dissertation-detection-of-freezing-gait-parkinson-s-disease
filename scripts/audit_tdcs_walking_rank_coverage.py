"""Reconstruct eligible S11 training pairs from the matched S10 crop trace.

This reads fitting annotations only for the per-fold grouping. No logits are
computed and no ranking term, training update, or threshold is fit locally.
The top-k identities still depend on S11 training predictions; the count is
fixed by the available fitting labels in each unchanged microbatch.
"""
from pathlib import Path
import json

import numpy as np

import tdcs_raw_walking_rank_v30 as rank
from validate_tdcs_raw_transfer_v28 import audit_trace, load_records, signature

ROOT=Path(__file__).resolve().parents[1]


def count_pairs(crops, records, cfg):
    all_records={r['id']:r for r in records}
    assert len(crops)%(cfg.micro_batch)==0
    walk=np.zeros(len(crops),np.int64); other=np.zeros_like(walk)
    for i,(id,subject,center,stretch) in enumerate(crops):
        record=all_records[id]; n=len(record['signal'])
        assert subject==record['subject']
        coords=center+np.linspace(-cfg.train_length*stretch/2,cfg.train_length*stretch/2,cfg.train_length)
        nearest=np.clip(np.rint(coords).astype(int),0,n-1)
        classes=record['classes'][nearest]
        valid=(coords>=0)&(coords<=n-1)&record['valid'][nearest]
        walk[i]=np.count_nonzero(valid&(classes==3))
        other[i]=np.count_nonzero(valid&((classes==1)|(classes==2)))
    walk=walk.reshape(-1,cfg.micro_batch).sum(1)
    other=other.reshape(-1,cfg.micro_batch).sum(1)
    pairs=walk*np.minimum(rank.HARD_NEGATIVES,other)
    return {'eligible_microbatches':int(np.count_nonzero(pairs)), 'total_microbatches':len(pairs),
        'pairs':int(pairs.sum()),'walking_labeled_crop_positions':int(walk.sum()),
        'other_FOG_labeled_crop_positions':int(other.sum())}


def main():
    raw=rank.raw; dates=ROOT/'results'
    audit=json.loads((dates/'tdcs_v28_S10_native_audit.json').read_text())
    assert audit['id']=='S10' and not audit['smoke_only'] and audit['status']=='native_AP_audited_replay_pending'
    run=(ROOT/audit['run']).resolve(); assert run.is_relative_to(ROOT/'kaggle-output')
    configuration=json.loads((run/'configuration.json').read_text()); cfg=raw.Config(**configuration['configuration'])
    context=raw.data.load_context(); main,_,_,_=load_records(context,False)
    assert signature(main)==configuration['dataset_signature'][:len(main)]
    folds=[]
    for fold in range(5):
        directory=run/f'fold_{fold}'; path=directory/'training_trace.json'
        receipt=json.loads((directory/'training_receipt.json').read_text())
        assert raw.data.digest(path)==receipt['artifact_sha256'][path.name]
        crops=json.loads(path.read_text())['fine_tune']; fitting=[r for r in main if r['fold']!=fold]
        audit_trace(crops,fitting,cfg,cfg.fine_steps)
        counts=count_pairs(crops,fitting,cfg)
        assert counts['total_microbatches']==cfg.fine_steps*cfg.batch_size//cfg.micro_batch
        folds.append({'fold':fold,'matched_S10_trace_sha256':raw.data.digest(path),**counts})
        print(folds[-1],flush=True)
    total=sum(f['total_microbatches'] for f in folds)
    result={'status':'matched_S10_crop_coverage_read_only', 'folds':folds,
        'fraction_of_eligible_microbatches':sum(f['eligible_microbatches'] for f in folds)/total,
        'actual_S11_usage_not_yet_measured':True,'holdout_labels_used_for_fitting_or_mining':False,
        'real_data_model_fitting_performed':False}
    raw.base.write(dates/'tdcs_v30_rank_matched_crop_coverage.json',result)


if __name__=='__main__': main()
