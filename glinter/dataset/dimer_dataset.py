from pathlib import Path
import pickle
import random
import copy

from collections import OrderedDict

import numpy as np
import torch

from glinter.esm_embed import ESMROOT
from glinter.protein import read_models, seq_encode, cigar_to_index

from glinter.dataset.msa_utils import load_tt, load_msa
from glinter.dataset._geometric_graph import (
    build_ca_graph, build_atom_graph, build_surface_graph,
)
from glinter.points.utils import get_random_rotmat, add_gaussian_noise

from glinter.dataset._feature import DimerFeature

class DimerDataset:
    def __init__(
        self, args, split='dev', max_row=-1, max_len=-1, esm_alphabet=None,
        training=False,
    ):
        self.args = args
        self.training = training
        self._feature = args.feature

        self.dataset = dict()
        self.mtens = dict()
        self.dtens = dict()
        self._dummy = None

        # don't use esm if esm_alphabet is None
        self.esm_alphabet = esm_alphabet
        # TODO: load dynamicly
        self.esm_tt = load_tt(args.esm_tt)

        self.max_row = max_row
        self.max_len = max_len

        # load data
        self.dimers = [] # (mname, rec, lig, dname)
        if self._feature.use('pickled-esm') :
            self.esm_dir = args.esm_root

        path = args.dimer_root
        self.dimers = self._load_from_pickle(path)

        self._esm_data = dict()

    def _load_from_pickle(self, path):
        esms = None
        if self._feature.use('pickled-esm'):
            esms = dict()
            for esmpt in self.esm_dir.glob('**/*.esm.npz'):
                esms[esmpt.stem.split('.')[0]] = esmpt

        with open(path, 'rb') as h:
            data = pickle.load(h)

        _dimers = []
        for mname in sorted(data):
            if esms is not None and mname not in esms:
                continue
            dten = data[mname]['dimer']
            recten = data[mname]['rec']
            ligten = data[mname]['lig']
            
            rec, lig = mname.split(':')
            if rec not in self.mtens:
                self.mtens[rec] = self._load_mten(mten=recten)
            if lig not in self.mtens:
                self.mtens[lig] = self._load_mten(mten=ligten)
            rec_alnidx = self.mtens[rec]['alnidx']
            lig_alnidx = self.mtens[lig]['alnidx']

            # avoid cutting sequences for ESM-MSA-1
            if ( 
                self.max_len > 0 and 
                rec_alnidx.size(-1) + lig_alnidx.size(-1) > self.max_len
            ):
                continue
            # the naming for hm is "recref:ligref", 
            # not compatible with the default naming used in _read_dimers
            dname = recten['seqmap']['ref'] + ':' + ligten['seqmap']['ref']
            if dname not in self.dtens:
                self.dtens[dname] = self._load_dten(dten=dten)


            _dimers.append((mname, rec, lig, dname))

        if len(_dimers) == 0:
            raise ValueError(f'emtpy dataset from {path}')

        return _dimers

    @classmethod 
    def add_args(cls, parser):
        parser.add_argument(
            '--dimer-root', type=Path, required=True, help='dimer data root',
        )
        parser.add_argument(
            '--esm-root', type=Path, help='esm root',
        )
        parser.add_argument(
            '--esm-tt', type=Path,
            default=ESMROOT.joinpath('esm_msa1_t12_100M_UR50S.tt'),
        )
        parser.add_argument(
            '--feature', type=DimerFeature, required=True,
            help='the names of features to use, separated by commas',
        )
        parser.add_argument(
            '--add-gaussian-noise', action='store_true'
        )
        parser.add_argument(
            '--cag-radius', type=float, default=8, help='ca-graph radius',
        )
        parser.add_argument(
            '--atg-radius', type=float, default=6, help='atom-graph radius',
        )
        parser.add_argument(
            '--sug-radius', type=float, default=6, help='surface-graph radius',
        )

    def _add_noise(self, sample):
        assert isinstance(sample, dict)
        for k in list(sample.keys()):
            if isinstance(sample[k], dict):
                sample[k] = self._add_noise(sample[k])
            elif hasattr(sample[k], 'pos'):
                sample[k].pos = add_gaussian_noise(sample[k].pos, std=0.5)
            # TODO : randomly rotate lrf and nor
            #elif hasattr(sample[k], 'nor'):
            #    sample[k].nor = add_gaussian_noise(sample[k].nor, std=0.5)
        return sample

    def _copy_graph(self, d):
        assert isinstance(d, dict)
        _d = dict()
        for k in d:
            if isinstance(d[k], dict):
                _d[k] = self._copy_graph(d[k])
            elif hasattr(d[k], 'pos'):
                _d[k] = copy.deepcopy(d[k])
            else:
                _d[k] = d[k]
        return _d

    def get_msa(self, i):
        return self.getitem(i, return_msa=True)

    def __getitem__(self, i):
        return self.getitem(i)

    def getitem(self, i, return_msa=False):
        if i in self.dataset:
            sample = dict(data=dict(),)
            sample['data'].update(self.dataset[i]['feat'])
            if self.args.add_gaussian_noise and self.training:
                sample = self._copy_graph(sample)
                sample['data'] = self._add_noise(sample['data'])
            if self._feature.use('pickled-esm'):
                sample['data']['esm'] = self._esm_data[i].to(dtype=torch.float32)
            return sample

        data = dict()

        # load monomer tensors
        recent, ligent = self.dimers[i][1:3]
        if recent not in self.mtens:
            self.mtens[recent] = self._load_mten(recent)
        if ligent not in self.mtens:
            self.mtens[ligent] = self._load_mten(ligent)

        rec = self.mtens[recent]
        lig = self.mtens[ligent]
        rec_alnidx = rec['alnidx']
        lig_alnidx = lig['alnidx']

        data = dict(
            recidx=rec_alnidx[0], # only store srcidx, since msa features are pre-cut
            ligidx=lig_alnidx[0],
        )

        if self._feature.use('ca-embed', 'coordinate-ca-graph', 'distance-ca-graph'):
            _init_kwargs = dict(
                use_distance_graph=self._feature.use('distance-ca-graph'),
                only_embed=self._feature.use('ca-embed'),
                r=self.args.cag_radius,
            )
            rec_rotmat = get_random_rotmat(3)
            rec_cag = build_ca_graph(
                rec, rec_alnidx, rotmat=rec_rotmat, **_init_kwargs,
            )
            lig_rotmat = get_random_rotmat(3)
            lig_cag = build_ca_graph(
                lig, lig_alnidx, rotmat=lig_rotmat, **_init_kwargs
            )
            if not self._feature.use('ca-embed'):
                data['rec_cag'] = rec_cag
                data['lig_cag'] = lig_cag
            else:
                data['rec_embed'] = rec_cag.transpose(0,1)
                data['lig_embed'] = lig_cag.transpose(0,1)

            if self._feature.use('atom-graph', 'heavy-atom-graph'):
                rec_atg = build_atom_graph(
                    rec, r=self.args.atg_radius, rotmat=rec_rotmat,
                    remove_hydrogen=self._feature.use('heavy-atom-graph'),
                )
                lig_atg = build_atom_graph(
                    lig, r=self.args.atg_radius, rotmat=lig_rotmat,
                    remove_hydrogen=self._feature.use('heavy-atom-graph'),
                )
                data['rec_atg'] = rec_atg
                data['lig_atg'] = lig_atg

            if self._feature.use('surface-graph'):
                rec_sug = build_surface_graph(
                    rec, r=self.args.sug_radius, rotmat=rec_rotmat,
                )
                lig_sug = build_surface_graph(
                    lig, r=self.args.sug_radius, rotmat=lig_rotmat,
                )
                data['rec_sug'] = rec_sug
                data['lig_sug'] = lig_sug

        # load dimer tensors
        dname = self.dimers[i][-1] # dtensor name
        if dname not in self.dtens:
            self.dtens[dname] = self._load_dten(dname)

        recidx, ligidx = rec_alnidx[1], lig_alnidx[1]
        reclen, liglen = recidx.size(0), ligidx.size(0)
        dten = self.dtens[dname]

        # load msa
        # pre-cut MSA features, because some targets could be overly long, since
        # the (potentialy) low tcov used during selecting pdb-a3m mappings
        if 'msa' in dten:
            tgtlen = reclen + liglen + 1 # pre-cut with bos
            # tgtlen = -1
            data['msa'] = load_msa(
                dten, self.max_row, recidx=recidx, ligidx=ligidx,
                esm_alphabet=self.esm_alphabet,
            )
            assert data['msa'].size(-1) == tgtlen
            data['reclen'] = reclen
            data['liglen'] = liglen
            if return_msa:
                return dict(data=data)
        if 'ccm' in dten:
            _ccm = dten['ccm']
            if tuple(_ccm.size()) != (reclen, liglen):
                _ccm = _ccm[recidx,:][:,ligidx] # pre-cut
            data['ccm'] = _ccm.unsqueeze(0)

        # load esm
        mname = self.dimers[i][0] # model name
        if self._feature.use('pickled-esm'):
            _esm = self._load_esm(mname)
            assert tuple(_esm.size()[-2:]) == (reclen, liglen)
            self._esm_data[i] = _esm

        data['mname'] = mname
        # build sample
        self.dataset[i] = dict(
            feat=data
        )

        sample = dict(data=dict())
        sample['data'].update(self.dataset[i]['feat'])
        if self._feature.use('pickled-esm'):
            sample['data']['esm'] = self._esm_data[i].to(dtype=torch.float32)

        return sample

    def __len__(self):
        # number of loaded dimers
        return len(self.dimers)

    
    def _load_mten(self, ent=None, mten=None):
        _mten = dict()
        if mten is None: 
            mt_path = self.mtdir.joinpath(ent + '.mten')
            with open(mt_path, 'rb') as h:
                mten = pickle.load(h)

        _mten['SEQ'] = self.esm_tt[seq_encode(mten['SEQ'])]
        seqmap = mten['seqmap']
        alnidx = cigar_to_index(
            seqmap['cigar'],seqmap['qbeg']-1, seqmap['tbeg']-1,
        )
        _mten['alnidx'] = alnidx

        for k in ('COORD', 'GROUP', 'ATOM', 'SAS', 'pssm'):
            _mten[k] = mten[k]

        if self._feature.use('surface-graph'):
            _mten['vcoord'] = mten['vertex']['coord']
            _mten['vnormal'] = mten['vertex']['normal']

        return _mten

    def _load_dten(self, dname=None, dten=None):
        if dten is None:
            dt_path = self.dtdir.joinpath(dname + '.dten')
            with open(dt_path, 'rb') as h:
                dten = pickle.load(h)

        _dten = dict()
        _dten['concated'] = dten['concated']
        _dten['reclen'] = dten['reclen']
        _dten['liglen'] = dten['liglen']
        if self._feature.use('esm'):
            _msa = torch.LongTensor(dten['msa'])
            # translate to esm embedding
            try:
                _dten['msa'] = self.esm_tt[_msa]
            except IndexError as e:
                print(dten['msa'].dtype, dten['msa'].shape)
                raise e

        if self._feature.use('ccm'):
            _ccm = torch.FloatTensor(dten['ccm'])
            _dten['ccm'] = _ccm

        return _dten
    
    def _load_esm(self, mname):
        with np.load(self.esm_dir.joinpath(f'{mname}.esm.npz')) as fh:
            esm = fh['esm']
        return torch.HalfTensor(esm)
