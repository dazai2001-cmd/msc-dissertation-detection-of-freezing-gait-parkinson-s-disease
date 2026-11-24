"""Verify an exact set of weak DeFOG training CSVs or official ZIP archives."""
import hashlib
import re
from pathlib import Path
from zipfile import ZipFile


def inventory(manifest, metadata, typed_ids, excluded_subjects):
    if metadata.Id.duplicated().any(): raise ValueError('Duplicate auxiliary metadata ID')
    expected={i:s for i,s in zip(metadata.Id,metadata.Subject) if i not in typed_ids and s not in excluded_subjects}
    selected=[r for r in manifest['records'] if r['domain']=='defog_notype']
    actual={r['id']:r['subject'] for r in selected}
    if len(selected)!=46 or len(actual)!=len(selected) or actual!=expected:
        raise ValueError('M22 weak file IDs/subjects differ from the frozen training inventory')
    for r in selected:
        if not re.fullmatch('[a-f0-9]{10}',r['id']) or not re.fullmatch('[a-f0-9]{64}',r['source_sha256']):
            raise ValueError('Malformed training file ID or digest')
    return sorted(selected,key=lambda r:r['id'])


def checked_csv_bytes(directory, item):
    directory=Path(directory).resolve()
    id=item['id']; assert re.fullmatch('[a-f0-9]{10}',id)
    name=id+'.csv'; plain=directory/name; archive=directory/(name+'.zip')
    if plain.is_file()==archive.is_file():
        raise ValueError(f'Expected exactly one official CSV or ZIP for {id}')
    source=plain if plain.is_file() else archive
    if not source.resolve().is_relative_to(directory): raise ValueError('Weak source escaped its directory')
    if source==plain: payload=plain.read_bytes()
    else:
        with ZipFile(archive) as zip_file:
            # Do not extract paths or accept archives containing other files.
            if zip_file.namelist()!=[name]: raise ValueError(f'Unexpected archive members for {id}')
            if zip_file.getinfo(name).file_size>200_000_000: raise ValueError(f'Unbounded archive for {id}')
            payload=zip_file.read(name)
    if hashlib.sha256(payload).hexdigest()!=item['source_sha256']:
        raise ValueError(f'Weak CSV bytes differ from training manifest: {id}')
    return payload
