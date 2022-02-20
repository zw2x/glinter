import re
import sys
import string
from pathlib import Path
import pickle

import numpy as np

from glinter.protein import read_seqs, read_ents

WORDS = string.ascii_uppercase + '-'
GAP = len(WORDS) - 1
AA = bytes.maketrans(WORDS.encode('latin1'), bytearray(range(len(WORDS))))

def get_len(desc):
    desc = re.search('.+ (\d+) / (\d+) ->.*', desc)
    return int(desc.group(2))

def num_words(msa, include=None, exclude=None):
    if include is not None:
        _imask = msa == include
    else:
        _imask = 1

    if exclude is not None:
        _emask = msa != exclude
    else:
        _emask = 1

    _mask = _imask * _emask

    return np.sum(_mask, axis=-1)

def read_a3mcc(path, fetch_length=True):
    seqs = read_seqs(path, ignore_header=True)
    msa = []
    rec_len, lig_len = None, None
    query = None
    for k, s in seqs.items():
        if rec_len is None and fetch_length:
            rec, lig = k.split('&')
            rec_len = get_len(rec)
            lig_len = get_len(lig)
        if query is None:
            query = s
        s = re.sub('[a-z]', '', s)
        s = s.encode('latin1').translate(AA)
        s = np.frombuffer(s, dtype=np.uint8)
        msa.append(s)

    try:
        msa = np.vstack(msa)
    except Exception as e:
        # sequence length doesn't match
        l = len(msa[0])
        for i, s in enumerate(msa):
            if len(s) != l:
                print(i)
        raise e

    if rec_len is not None:
        assert rec_len + lig_len == len(query)
        assert rec_len + lig_len == msa.shape[-1]
        return msa, query, (rec_len, lig_len)
    else:
        return msa, query, len(query)

def read_seqfile(path):
    seq = read_seqs(path, topk=1)
    assert len(seq) == 1
    return list(seq.values())[0]

def heniw(msa, discount_gaps=True):
    """Compute henikoff weights
    """
    ncol = msa.shape[-1]

    cnt = np.zeros((len(WORDS), ncol), dtype=np.float32)
    for i in range(msa.shape[-1]):
        cnt[:,i] = np.bincount(msa[:, i], minlength=len(WORDS))

    _w = 1 / np.sum(cnt > 0, axis=0)
    msaw = _w / cnt[msa, np.arange(ncol, dtype=np.int64)]
    hw = np.sum(msaw, axis=-1)
    hw = hw / np.sum(hw)
    if discount_gaps:
        # re-weight by the density of non-gaps
        # _pow = 1
        # hw = hw * ((num_words(msa, exclude=GAP) / ncol) ** _pow)
        hw = hw * num_words(msa, exclude=GAP) / ncol
    return hw

def build_msa(
    tgtdir, msa_paths, seq_paths, use_a3mcc, use_tqdm, dump=True, maxk=128,
    no_check=False,
):
    if use_tqdm:
        msa_paths = tqdm(msa_paths)
    n = 0
    for msa_path in msa_paths:
        msa_name = msa_path.stem.split('.')[0]
        if use_a3mcc:
            msa, query, (rec_len, lig_len) = read_a3mcc(
                msa_path, fetch_length=True
            )
            try:
                rec, lig = msa_name.split(':')
            except ValueError:
                rec, lig = 'A', 'B'
            concated = True
            if not no_check:
                assert len(read_seqfile(seq_paths[rec])) == rec_len
                assert len(read_seqfile(seq_paths[lig])) == lig_len
        else:
            msa, query, _len = read_a3mcc(
                msa_path, fetch_length=False
            )
            rec_len, lig_len = _len, _len
            rec, lig = msa_name, msa_name
            concated = False

        hw = heniw(msa,)
        if len(hw) > maxk and maxk > 0:
            idx = np.argsort(hw)[::-1][:maxk]
            msa = msa[idx]
            hw = hw[idx]
        else:
            idx = None

        sample = dict(
            rec=rec,
            lig=lig,
            query=query,
            msa=msa,
            hw=hw,
            reclen=rec_len,
            liglen=lig_len,
            idx=idx,
            concated=concated,
        )

        if not dump:
            nrow, ncol = sample['msa'].shape
            print(sample)
            if idx is not None:
                print(nrow, ncol)
                print(sample)
                import matplotlib.pyplot as plt
                gap_den = np.sort(num_words(sample['msa'], include=GAP) / ncol)
                plt.plot(gap_den)
                plt.show()
                break

        tgt_path = tgtdir.joinpath(msa_path.stem + '.msa')
        with open(tgt_path, 'wb') as h:
            pickle.dump(sample, h)

        print(f'dump msa feature at {tgt_path}')

        n += 1
    return n

def _get_options():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--a3mdir', type=Path, required=True)
    parser.add_argument(
        '--model', type=Path, help='heterodimer / homodimer list'
    )
    parser.add_argument(
        '--seqdir', type=Path, help='refseq dir'
    )
    parser.add_argument('--tgtdir', type=Path,)
    parser.add_argument('--use-concat', action='store_true')
    parser.add_argument(
        '--use-hhfilter', action='store_true',
        help='the a3mdir is already filtered by hhfilter'
    )
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-check', action='store_true')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    """Build MSA features for monomers and dimers
    """
    args = _get_options()
    models, ents = None, None
    if args.model:
        models, ents = read_ents(args.model, return_both=True, key=1,)
    else:
        models = args.a3mdir.stem
        ents = models.split(':')
        models = [ models ]
    _ref = models if args.use_concat else ents
    a3mpaths = []
    for name in _ref:
        if args.use_hhfilter:
            a3mpt = args.a3mdir.joinpath(f'{name}.hh.a3m')
        else:
            if args.use_concat:
                a3mpt = args.a3mdir.joinpath(f'{name}/{name}.a3m_cc')
            else:
                a3mpt = args.a3mdir.joinpath(f'{name}/{name}.a3m')
        if a3mpt.exists():
            a3mpaths.append(a3mpt)

    seqpaths = dict() 
    for name in ents:
        if args.seqdir is None:
            seqpt = args.a3mdir.joinpath(f'{name}/{name}.seq')
        else:
            seqpt = args.seqdir.joinpath(f'{name}/{name}.seq')
        if seqpt.exists():
            seqpaths[seqpt.stem] = seqpt

    if len(a3mpaths) == 0:
        raise RuntimeError('cannot find any a3ms')
    if not args.no_check and len(seqpaths) == 0:
        raise RuntimeError('cannot find any seqs')

    if args.debug:
        build_msa(
            args.tgtdir, a3mpaths[:2], seqpaths, args.use_concat, True,
            dump=False, maxk=1024,
        )
        exit()

    assert args.tgtdir is not None
    if not args.tgtdir.exists():
        args.tgtdir.mkdir(parents=True,)

    build_msa(args.tgtdir, a3mpaths, seqpaths, args.use_concat, False, 
        no_check=args.no_check)
