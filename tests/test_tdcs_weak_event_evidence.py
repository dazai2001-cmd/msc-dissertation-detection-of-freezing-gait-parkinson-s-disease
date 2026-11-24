"""Exact M22 weak training source checks, using synthetic CSVs only."""
import hashlib
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tdcs_weak_event_evidence import checked_csv_bytes,inventory


def test_zip_csv_integrity_and_no_ambiguous_source(tmp_path):
    payload=b'Time,Event,Valid,Task\n0,1,1,1\n'
    item={'id':'abc0123456','source_sha256':hashlib.sha256(payload).hexdigest()}
    source=tmp_path/'abc0123456.csv.zip'
    with ZipFile(source,'w') as z: z.writestr('abc0123456.csv',payload)
    assert checked_csv_bytes(tmp_path,item)==payload
    with ZipFile(source,'w') as z: z.writestr('../abc0123456.csv',payload)
    with pytest.raises(ValueError,match='Unexpected archive'): checked_csv_bytes(tmp_path,item)
    with ZipFile(source,'w') as z: z.writestr('abc0123456.csv',b'wrong data')
    with pytest.raises(ValueError,match='bytes differ'): checked_csv_bytes(tmp_path,item)
    (tmp_path/'abc0123456.csv').write_bytes(payload)
    with pytest.raises(ValueError,match='exactly one'): checked_csv_bytes(tmp_path,item)


def test_weak_training_IDs_require_all_sources_and_exclude_TDCS_subject():
    ids=[f'{i:010x}' for i in range(48)]
    metadata=pd.DataFrame({'Id':ids,'Subject':['weak']*46+['typed','tdcs']})
    rows=[{'id':i,'subject':'weak','source_sha256':'a'*64,'domain':'defog_notype'} for i in ids[:46]]
    args=(metadata,{ids[46]},{'tdcs'})
    assert inventory({'records':rows},*args)==rows
    with pytest.raises(ValueError,match='differ'): inventory({'records':rows[:-1]},*args)
    with pytest.raises(ValueError,match='differ'):
        inventory({'records':[*rows[:-1],{**rows[-1],'id':ids[-1],'subject':'tdcs'}]},*args)
