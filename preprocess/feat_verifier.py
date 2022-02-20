import re
import sys
import pickle
import random
import argparse
from pathlib import Path

import torch
import numpy as np

from glinter.protein import (
    encode_atoms, encode_ss8, three_to_one,
    cigar_to_index, show_aln, get_alniden,
    read_models,
)

def load_dtensor(path):
    """Load dimer tensors (i.e. .dten)
    """
    with open(path, 'rb') as h:
        dten = pickle.load(h)
    return dten

def load_mtensor(path):
    """Load monomer tensors (i.e. .mten)
    """
    with open(path, 'rb') as h:
        mten = pickle.load(h)
    return mten

def load_target(path):
    """Load target tensors
    """
    with open(path, 'rb') as h:
        tgt = pickle.load(h)
    return tgt

def check_consistency(
    tpaths, mpaths, dpaths, rec, lig, dname, msa_repo=None, repo=None, augment=True, ignore_seqid=True,
):
    """Return a list of models that have consistent features 
    Args:
        tpaths (dict)
        mpaths (dict)
        dpaths (dict)
        rec (str)
        lig (str)
        dname (str)
    """
    mname = f'{rec}:{lig}'
    if tpaths:
        tgt = load_target(tpaths[mname])
    else:
        tgt = None
    dten = load_dtensor(dpaths[dname])
    dseq = dten['query']
    if augment:
        augment = dten['concated']

    if not dten['concated']:
        dname = dname + ':' + dname
        dseq += dseq

    recten = load_mtensor(mpaths[rec])
    ligten = load_mtensor(mpaths[lig])
    # check seqmap
    rec_seqmap = recten['seqmap']
    lig_seqmap = ligten['seqmap']
    if rec_seqmap is not None and lig_seqmap is not None:
        # check refname
        refrec = rec_seqmap['ref']
        reflig = lig_seqmap['ref']
        if dname != refrec + ':' + reflig:
            print(dname, reflig, refrec)
            return False
        # check refseq
        refrecseq = rec_seqmap['refseq']
        refligseq = lig_seqmap['refseq']
        assert refrecseq + refligseq == dseq
        # filter by sequence identity of the alignment
        recseq, ligseq = recten['SEQ'], ligten['SEQ']
        rec_alnidx = cigar_to_index(
            rec_seqmap['cigar'],rec_seqmap['qbeg']-1, rec_seqmap['tbeg']-1
        )
        if not ignore_seqid and get_alniden(rec_alnidx, recseq, refrecseq) < 0.9:
            print(dname, reflig, refrec)
            return False
        lig_alnidx = cigar_to_index(
            lig_seqmap['cigar'],lig_seqmap['qbeg']-1, lig_seqmap['tbeg']-1
        )
        if not ignore_seqid and get_alniden(lig_alnidx, ligseq, refligseq) < 0.9:
            print(dname, reflig, refrec)
            return False

    # check target size
    assert tgt is None or tuple(tgt.size()) == (len(recseq), len(ligseq))

    if repo is not None:
        repo[mname] = dict(
            rec=recten,
            lig=ligten,
            tgt=tgt,
        )
        if msa_repo is not None:
            msa_repo[mname] = dten
        else:
            repo[mname]['dimer'] = dten

        if augment:
            mname = f'{lig}:{rec}'
            assert mname not in repo
            repo[mname] = dict(
                lig=recten,
                rec=ligten,
                tgt=tgt.T if tgt is not None else None,
            )
            _msa = dten['msa']
            dten_swapped = dict(
                reclen=dten['liglen'],
                liglen=dten['reclen'],
                hw=dten['hw'],
                idx=dten['idx'],
                query=dseq[int(dten['reclen']):] + dseq[:int(dten['reclen'])],
                msa=np.concatenate(
                    (_msa[:,int(dten['reclen']):], _msa[:,:int(dten['reclen'])]),
                    axis=-1,
                ),
                concated=True,
            )

            if msa_repo is not None:
                msa_repo[mname] = dten_swapped
            else:
                repo[mname]['dimer'] = dten_swapped

    return True

def _get_options():
    parser = argparse.ArgumentParser(
        prog='check feature consistency and output models that passed the check'
    )
    parser.add_argument(
        '--msa-dir', type=Path, required=True, help='msa tensor dir',
    )
    parser.add_argument(
        '--mten-dir', type=Path, required=True, help='monomer tensor dir',
    )
    parser.add_argument(
        '--dist-dir', type=Path, help='distance tensor dir',
    )
    parser.add_argument('--model', type=Path)
    parser.add_argument(
        '--export', type=str, choices=['ALL', 'HT', 'HM'], default='ALL',
    )
    parser.add_argument('--repo', type=Path, help='repo path',)
    parser.add_argument('--msa-repo', type=Path, help='repo MSA path',)
    parser.add_argument('--check-seqid', action='store_true')
    parser.add_argument('--from-paths', action='store_true')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    """
    Verify and collect features
    """
    args = _get_options()
    dpaths = dict(
        (_.stem.split('.')[0], _) for _ in args.msa_dir.glob('**/*.msa')
    )
    mpaths = dict((_.stem, _) for _ in args.mten_dir.glob('**/*.mten'))
    if args.dist_dir is not None:
        tpaths = dict((_.stem, _) for _ in args.dist_dir.glob('**/*.dist'))
    else:
        tpaths = None

    if args.from_paths:
        models = {'A:B':'A:B'}
    else:
        if args.model:
            models = read_models(args.model)
        else:
            models = {args.mten_dir.stem: args.msa_dir.stem}

    # find avail models
    hts, hms = dict(), dict()
    for mname in list(models.keys()):
        if tpaths is not None and mname not in tpaths:
            del models[mname]
            continue

        rec, lig = mname.split(':')
        if not (rec in mpaths and lig in mpaths):
            del models[mname]
            continue

        # check seq entries
        dname = models[mname]
        if ':' in dname:
            rec, lig = dname.split(':')
        else:
            rec, lig = dname, dname
        if rec == lig:
            if rec not in dpaths:
                del models[mname]
                continue
            hms[mname] = (rec, lig)
        else:
            if models[mname] not in dpaths:
                del models[mname]
                continue
            hts[mname] = (rec, lig)

#    print(len(hts), len(hms))
    repo = {} if args.repo is not None else None
    msa_repo = {} if args.msa_repo is not None else None
    n = 0
    for name, (srec, slig) in hts.items():
        dname = srec + ':' + slig
        rec, lig = name.split(':')
        if check_consistency(
            tpaths, mpaths, dpaths, rec, lig, dname,
            repo=repo, msa_repo=msa_repo, ignore_seqid=not args.check_seqid,
        ):
            n += 1

    for name, (srec, slig) in hms.items():
        dname = srec
        rec, lig = name.split(':')
        if check_consistency(
            tpaths, mpaths, dpaths, rec, lig, dname,
            repo=repo, msa_repo=msa_repo, ignore_seqid=not args.check_seqid,
        ):
            n += 1

    if repo is not None and len(repo) > 0:
        with open(args.repo, 'wb') as h:
            pickle.dump(repo, h)

    if msa_repo is not None and len(msa_repo) > 0:
        with open(args.msa_repo, 'wb') as h:
            pickle.dump(msa_repo, h)
