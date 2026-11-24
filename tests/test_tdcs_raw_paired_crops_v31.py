"""Synthetic-only checks for matched subtype-pair sampling and fitting."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import tdcs_raw_paired_crops_v31 as paired


def test_pair_and_natural_stream_match_declared_sampling():
    records,_,_=paired.weak.synthetic(); cfg=paired.smoke_config()
    records,p=paired.raw.sampling_plan(records)
    paired_rng=np.random.default_rng(44); natural_rng=np.random.default_rng(44)
    sampler=paired.PairedBatch(records,cfg,99)
    for other in (2,1,2):
        x,y,m,trace=sampler(records,p,paired_rng,cfg)
        natural=paired.ORIGINAL_BATCH(records,p,natural_rng,cfg)
        for a,b in zip((x,y,m),natural[:3]): np.testing.assert_array_equal(a[2:],b[2:])
        assert trace[2:]==natural[3][2:]
        assert (m[0]&(y[0]==3)).any() and (m[1]&(y[1]==other)).any()
    use=sampler.receipt(99)
    assert use['paired_batches']==3 and use['eligible_microbatches']>=3
    assert use['natural_crops']==18 and use['targeted_centers']=={'Walking':3,'Turn':2,'StartHesitation':1}


def test_training_episode_pool_refuses_auxiliary_and_missing_class():
    main,aux,_=paired.weak.synthetic()
    with pytest.raises(ValueError,match='Only outer fitting TDCS'): paired.episode_pools(main+aux)
    with pytest.raises(ValueError,match='lacks'): paired.episode_pools([])


def test_matched_pair_fits_have_exact_same_crops_and_restore_original_code(tmp_path):
    raw=paired.raw
    results={}
    for id,suite in [('M23','metadata'),('S12','sensor')]:
        session={'suite':suite,'cfg':raw.base.smoke_config(),'output':tmp_path/id,
            'context':{},'results':[],'runtime':{},'deadline':float('inf')}
        result=paired.run(session,id); assert result['status']=='complete'
        results[id]=Path(result['run_directory'])
        info=json.loads((results[id]/'paired_crop_usage.json').read_text())
        assert info['ranking_used']==(id=='S12') and not info['outer_labels_used_for_pairing']
        assert len(info['folds'])==5 and all(r['paired_batches']==1 for r in info['folds'])
        from validate_tdcs_paired_crops_v31 import audit_usage
        records,_,_=paired.weak.synthetic()
        assert audit_usage(results[id],id,records,paired.smoke_config())==info
        for fold in range(5):
            trace=json.loads((results[id]/f'fold_{fold}/training_trace.json').read_text())
            assert {row[1] for row in trace['fine_tune']}.isdisjoint({r['subject'] for r in records if r['fold']==fold})
    for fold in range(5):
        trace=f'fold_{fold}/training_trace.json'
        assert (results['M23']/trace).read_bytes()==(results['S12']/trace).read_bytes()
    assert raw.source_hash()==paired.weak.typed.FROZEN_RAW_SHA256
    assert raw.batch is paired.ORIGINAL_BATCH and not set(paired.SPECS)&set(raw.SPECS)
