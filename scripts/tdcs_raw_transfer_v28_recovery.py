"""S10 reader-only recovery: join subtype-labelled train/defog files to metadata.

The 137 metadata IDs cover 91 typed DeFOG and 46 train/notype recordings.
notype has no subtype supervision for the fixed four-state objective. Do not
invent those labels or read test/defog, even when an ID also occurs there.
The frozen v28 model, optimizer, budgets, masks, units and TDCS folds do not
change. A composite source hash explicitly binds this reader replacement.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import tdcs_raw_transfer_v28 as raw

FROZEN_RAW_SHA256='17268a0911be41d6064edd7f696dd406dbbf17759561b96c36275207a0781231'
ORIGINAL_PREPARE=raw.prepare


def source_hash():
    return globals().get('SOURCE_SHA256') or raw.data.digest(__file__)


def effective_source_hash():
    return hashlib.sha256(json.dumps({'frozen_raw_source':FROZEN_RAW_SHA256,
        'labelled_file_manifest_reader':source_hash()},sort_keys=True).encode()).hexdigest()


def auxiliary_inventory(metadata,raw_root,excluded_subjects,expected_files=91):
    """Select actual typed training files, not all IDs in a wider dimension table."""
    raw_root=Path(raw_root)
    if not {'Id','Subject'}.issubset(metadata.columns): raise ValueError('Required auxiliary metadata columns missing')
    if metadata.Id.isna().any() or metadata.Subject.isna().any() or metadata.Id.duplicated().any():
        raise ValueError('Auxiliary metadata IDs/subjects must be nonmissing and IDs unique')
    files=sorted(raw_root.glob('*.csv')); ids=[p.stem for p in files]
    if not files or (expected_files is not None and len(files)!=expected_files):
        raise ValueError(f'Expected {expected_files} subtype-labelled train/defog CSVs, found {len(files)}')
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate auxiliary training file ID')
    unknown=set(ids)-set(metadata.Id)
    if unknown: raise ValueError('Auxiliary training files without metadata: '+str(sorted(unknown)))
    matched=metadata.set_index('Id').loc[ids].reset_index()
    selected=matched[~matched.Subject.isin(excluded_subjects)].copy()
    selected['path']=[str(raw_root/(id+'.csv')) for id in selected.Id]
    if selected.empty: raise ValueError('No subject-independent subtype-labelled auxiliary training files')
    audit={'metadata_rows':len(metadata),'metadata_subjects':int(metadata.Subject.nunique()),
        'available_subtype_labelled_training_files':len(files),'available_subtype_labelled_subjects':int(matched.Subject.nunique()),
        'metadata_IDs_outside_typed_training_folder':sorted(set(metadata.Id)-set(ids)),
        'selected_training_files':len(selected),'selected_subjects':int(selected.Subject.nunique()),
        'excluded_TDCS_subject_file_IDs':matched.loc[matched.Subject.isin(excluded_subjects),'Id'].tolist(),
        'selected_IDs':selected.Id.tolist(),'subject_ID_overlap_after_exclusion':0,
        'selection':'actual train/defog CSVs inner-joined to unique metadata; train/notype and all test files excluded',
        'missing_subtype_labels_invented':False,'auxiliary_loss_mask':'Valid AND Task'}
    assert set(selected.Subject).isdisjoint(excluded_subjects)
    return selected,audit


def prepare(context,root,transfer):
    if not transfer: return ORIGINAL_PREPARE(context,root,False)
    main,_,y=ORIGINAL_PREPARE(context,root,False)
    excluded=set(pd.read_csv(raw.data.csv_path('tdcsfog_metadata.csv'),dtype=str).Subject)
    metadata=pd.read_csv(raw.data.csv_path('defog_metadata.csv'),dtype=str)
    raw_root=raw.data.frozen.shared.common.locate_data()[0].parent/'defog'
    selected,audit=auxiliary_inventory(metadata,raw_root,excluded)
    aux=[]; root=Path(root); manifest=json.loads((root/'raw_dataset_manifest.json').read_text())
    for row in selected.itertuples(index=False):
        frame=pd.read_csv(row.path,usecols=['Time',*raw.data.AXES,*raw.LABELS,'Valid','Task'])
        record=raw.make_record(frame,row.Id,row.Subject,-1,100,'defog'); aux.append(record)
        manifest['records'].append({'id':row.Id,'subject':row.Subject,'domain':'defog',
            'source_sha256':raw.data.digest(row.path),'signal_sha256':raw.data.array_digest(record['signal']),
            'class_sha256':raw.data.array_digest(record['classes']),'valid_sha256':raw.data.array_digest(record['valid']),
            'valid_rows':int(record['valid'].sum())})
        np.savez_compressed(root/'raw_cache'/(row.Id+'.npz'),signal=record['signal'],classes=record['classes'],valid=record['valid'])
    manifest['auxiliary_inventory']=audit
    manifest['reader_recovery_source_sha256']=source_hash()
    manifest['effective_raw_source_sha256']=effective_source_hash()
    raw.base.write(root/'raw_dataset_manifest.json',manifest)
    raw.base.write(root/'reader_recovery_provenance.json',{
        'frozen_raw_source_sha256':FROZEN_RAW_SHA256,'reader_source_sha256':source_hash(),
        'effective_source_sha256':effective_source_hash(),'inventory':audit,
        'model_optimizer_masks_units_budgets_and_TDCS_folds_changed':False})
    print('Recovered auxiliary inventory:',audit,flush=True)
    return main,aux,y


@contextmanager
def applied():
    """Replace the reader only in this process; always restore frozen module state."""
    assert raw.source_hash()==FROZEN_RAW_SHA256, 'Unexpected base raw revision or nested recovery'
    previous_prepare=raw.prepare; had_override='SOURCE_SHA256' in raw.__dict__
    previous_override=raw.__dict__.get('SOURCE_SHA256')
    raw.prepare=prepare; raw.SOURCE_SHA256=effective_source_hash()
    try: yield
    finally:
        raw.prepare=previous_prepare
        if had_override: raw.SOURCE_SHA256=previous_override
        else: raw.__dict__.pop('SOURCE_SHA256',None)


def run(session,id='S10'):
    if id!='S10' or session['suite']!='sensor': raise ValueError('Only the failed S10 reader is recovered')
    with applied(): return raw.run(session,id)
