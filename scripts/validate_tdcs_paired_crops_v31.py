"""Audit v31 predictions, recorded pair exposure, and fitting-only crop centers."""
import argparse
import json
from pathlib import Path

import numpy as np

import tdcs_raw_paired_crops_v31 as paired
import validate_tdcs_raw_transfer_v28 as validator
from audit_tdcs_walking_rank_coverage import count_pairs

ROOT=Path(__file__).resolve().parents[1]


def audit_usage(run,id,records,cfg):
    usage=validator.read_json(run/'paired_crop_usage.json')
    assert usage['id']==id and usage['source_sha256']==paired.effective_source_hash()
    assert usage['specification']==paired.SPECS[id] and usage['ranking_used']==(id=='S12')
    assert usage['smoke_only']==cfg.smoke and not usage['outer_labels_used_for_pairing']
    assert not usage['auxiliary_pretraining_crops_changed'] and len(usage['folds'])==5
    for fold,row in enumerate(usage['folds']):
        assert row['fold']==fold and row['seed']==cfg.seed+1009*fold+200000
        assert row['pair_rng_seed']==row['seed']+paired.PAIR_SEED_OFFSET
        assert row['paired_batches']==cfg.fine_steps and row['natural_crops']==6*cfg.fine_steps
        fitting=[r for r in records if r['fold']!=fold]; by_id={r['id']:r for r in fitting}
        trace=validator.read_json(run/f'fold_{fold}'/'training_trace.json')['fine_tune']
        validator.audit_trace(trace,fitting,cfg,cfg.fine_steps)
        counts=count_pairs(trace,fitting,cfg)
        for name in ('eligible_microbatches','total_microbatches','pairs'): assert row[name]==counts[name]
        measured={1:0,2:0,3:0}
        for step in range(cfg.fine_steps):
            for slot,target in enumerate((3,2 if step%2==0 else 1)):
                rid,subject,center,_=trace[step*cfg.batch_size+slot]; r=by_id[rid]
                assert subject==r['subject'] and 0<center<len(r['classes'])-1
                idx=np.arange(center-1,center+2)
                assert r['valid'][idx].all() and np.all(r['classes'][idx]==target)
                measured[target]+=1
        assert row['targeted_centers']=={paired.raw.LABELS[k-1]:v for k,v in measured.items()}
    return usage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-output',type=Path,required=True); parser.add_argument('--id',choices=paired.SPECS,required=True)
    parser.add_argument('--weak-root',type=Path,required=True); parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--replay',action='store_true'); parser.add_argument('--replay-progress',type=Path)
    args=parser.parse_args()
    assert args.result.resolve().is_relative_to(ROOT/'results')
    if args.replay_progress is not None:
        assert args.replay_progress.resolve().is_relative_to(ROOT/'results')
        assert args.result.resolve()!=args.replay_progress.resolve()
    context=paired.raw.data.load_context()
    with paired.applied(args.id):
        result=validator.audit(args.suite_output,args.id,args.replay,context=context,
            progress=args.replay_progress,weak_root=args.weak_root)
    run=Path(result['run']); cfg=paired.raw.Config(**validator.read_json(run/'configuration.json')['configuration'])
    records,_,_,_=validator.load_records(context,False)
    result['paired_crop_usage']=audit_usage(run,args.id,records,cfg)
    result['paired_training_centers_and_counts_independently_reconstructed']=True
    paired.raw.base.write(args.result,result)
    print(json.dumps({k:v for k,v in result.items() if k!='paired_crop_usage'},indent=2))


if __name__=='__main__': main()
