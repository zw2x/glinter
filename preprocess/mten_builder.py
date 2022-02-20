import argparse
import pickle
from pathlib import Path
import random
import re
import torch

from glinter.protein import (
    encode_atoms, encode_ss8, three_to_one,
    cigar_to_index, show_aln, read_ents
)

def load_pssm(path,):
    with open(path, 'rb') as h:
        hhr = pickle.load(h)
        pssm = torch.FloatTensor(hhr['PSSM'])
    return pssm

def to_tensor_(feat, *keys, dtype=torch.float16):
    for k in keys:
        feat[k] = torch.tensor(feat[k], dtype=dtype)
    
def tensorize_feat(args, feat, ignore_h=True):
    _feat = dict(
        SEQ=[],
        COORD=[],
        ATOM=[],
        SAS=[],
        GROUP=[], # number of atoms in a residue
    )
    if args.use_dssp:
        _feat.update(dict(
            SS8=[],
            rASA=[],
            Phi=[],
            Psi=[],
        ))

    for s in feat:
        _feat['SEQ'].append(three_to_one(s['name']))

        if args.use_dssp:
            if 'dssp' not in s or s['dssp'] is None:
                return
    
            for di, dk in enumerate(('SS8', 'rASA', 'Phi', 'Psi')):
                _feat[dk].append(s['dssp'][di])

        n = 0
        for k, v in sorted(s['atoms'].items()):
            if ignore_h and k.startswith('H'):
                continue
            _feat['ATOM'].append(k)
            _feat['COORD'].append(v['coord'])
            _feat['SAS'].append(v['sas'])
            n += 1
        assert n < 250
        _feat['GROUP'].append(n)

    _feat['SEQ'] = ''.join(_feat['SEQ'])
    _feat['GROUP'] = torch.tensor(_feat['GROUP'], dtype=torch.uint8)
    _feat['ATOM'] = encode_atoms(_feat['ATOM']).to(dtype=torch.uint8)
    _tensor_keys = ['COORD', 'SAS',]

    if args.use_dssp:
        _feat['SS8'] =  encode_ss8(_feat['SS8']).to(dtype=torch.uint8)
        _tensor_keys += ['rASA', 'Phi', 'Psi']

    to_tensor_(_feat, *_tensor_keys)

    return _feat

def _get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hhr', type=Path, required=True, help='hhr directory')
    parser.add_argument('--msms', type=Path, required=True, help='MSMS feature directory')
    parser.add_argument('--tgtdir', type=Path, required=True,)
    parser.add_argument('--ents', type=Path, help='dimer file')
    parser.add_argument('--filter', action='store_true',)
    parser.add_argument('--use-dssp', action='store_true',)
    parser.add_argument('--use-hydrogen', action='store_true',)
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    """
    Tensorize features for **monomers**
    """
    args = _get_options()

    if args.ents:
        ents = read_ents(args.ents)
    else:
        ents = None 


    if ents is None:
        sources = sorted(args.msms.glob('**/*.feat'))
    else:
        sources = []
        for ent in ents:
            src = args.msms.joinpath(f'{ent}/{ent}.feat') 
            if src.exists():
                sources.append(src)
        sources = sorted(sources)

    tgtdir = Path(args.tgtdir)
    if not tgtdir.exists():
        tgtdir.mkdir(parents=True)

    n = 0
    for src_path in sources:
        with open(src_path, 'rb') as h:
            feat = pickle.load(h)

        seqmap = feat['seqmap']
        # filtering
        if args.filter:
            _len = idx.size(0)
            if _len > 1200 or _len < 20:
                continue

        # load hhr
        if seqmap is not None:
            ref = seqmap['ref']
        else:
            ref = src_path.stem
        hhrpath = args.hhr.joinpath(f'{ref}/{ref}.hhm.pkl')
        if not hhrpath.exists():
            continue
        pssm = load_pssm(hhrpath)
        if pssm is None:
            continue

        # tensorize features
        mtensor = tensorize_feat(
            args, feat['feat'], ignore_h=not args.use_hydrogen,
        )
        if mtensor is None:
            continue

        if mtensor['SEQ'] != feat['seq']:
            continue

        mtensor['name'] = feat['name']    
        mtensor['pssm'] = pssm
        mtensor['seqmap'] = feat['seqmap']
        mtensor['vertex'] = dict(
            coord=torch.HalfTensor(feat['vertex']['coords'],),
            normal=torch.HalfTensor(feat['vertex']['normals']),
        )
        ch = src_path.stem
        tgt_path = tgtdir.joinpath(ch + '.mten')
        if args.debug:
            print(seqmap)
            print(mtensor)
            print(pssm.size())
            print(feat['seq'])
            print(seqmap['refseq'])
            print("********************")
            # pdbseq to a3mseq index
            alnidx = cigar_to_index(
                seqmap['cigar'], qbeg=seqmap['qbeg']-1, tbeg=seqmap['tbeg']-1
            ) # qbeg and tbeg starts from 1 instead of 0
            show_aln(alnidx, feat['seq'], seqmap['refseq'],)
            break
        else:
            with open(tgt_path, 'wb') as h:
                pickle.dump(mtensor, h)
            print(f'dump mtensor at {tgt_path}')
        n += 1

    # print(n)
