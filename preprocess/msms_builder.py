import pickle
from pathlib import Path

import torch

from Bio.PDB.DSSP import DSSP
from Bio.PDB.PDBParser import PDBParser
import Bio.PDB.PDBExceptions as PDBExceptions

from glinter.protein import (
    get_residues, get_atoms, one_to_three, get_chainid, read_ents, read_seqmap,
    read_seqs, get_pdbseq
)
from glinter.points.mesh import (sample_points, read_msms,)

"""
assuming each atom can be uniquely named as "chainid_resid_resname_atomname"
(i.e. DisorderedResidue is "packed")
"""
def read_areas(path):
    areas = {}
    header = None
    with open(path, 'rt') as h:
        for l in h:
            l = l.strip()
            if l == '':
                continue
            if header is None:
                header = l
                continue
            _, ses, sas, name = l.split()
            assert name not in areas
            # ses = float(ses)
            sas = float(sas)
            areas[name] = sas #(ses, sas)
    return areas

def read_coords(path, ignore_h=True):
    coords = []
    parser = PDBParser(QUIET=True)
    model = parser.get_structure('', path).get_list()[0]
    chains = model.get_list()
    i = 0
    seq = ''
    # treats multiple chains as a single chain, (useful for parsing DBv5)
    for chain in chains:
        residues = get_residues(chain)
        chainid = get_chainid(chain) # replace '' with '*', consistent with BIPSPI
        seq += get_pdbseq(residues, thr=-1)
        for residue in residues:
            resid = residue.get_id()[1]
            resname = residue.resname
            name = f'{chainid}_{resid}_{resname}'
            group = []
            for atom in get_atoms(residue, ignore_h=ignore_h,):
                group.append([atom.name, atom.coord])
            coords.append((i, name, group))
            i += 1
    return coords, model, seq

""" (copied from Biopython)
The DSSP codes for secondary structure used here are:
    =====     ====
    Code      Structure
    =====     ====
     H        Alpha helix (4-12)
     B        Isolated beta-bridge residue
     E        Strand
     G        3-10 helix
     I        Pi helix
     T        Turn
     S        Bend
     \-       None
    =====     ====

The dssp data returned for a single residue is a tuple in the form:
    ============ ===
    Tuple Index  Value
    ============ ===
    0            DSSP index
    1            Amino acid
    2            Secondary structure
    3            Relative ASA
    4            Phi
    5            Psi
    6            NH-->O_1_relidx
    7            NH-->O_1_energy
    8            O-->NH_1_relidx
    9            O-->NH_1_energy
    10           NH-->O_2_relidx
    11           NH-->O_2_energy
    12           O-->NH_2_relidx
    13           O-->NH_2_energy
    ============ ===
"""
def read_dssp(coords, model, path):
    if not path.exists():
        return

    try:
        dssp = DSSP(model, path, file_type='DSSP')

    except PDBExceptions.PDBException as e:
        if 'Structure/DSSP mismatch' in str(e):
            return
        else:
            raise e
 
    _dssp = {}
    for k in dssp.keys():
        chainid = k[0]
        if chainid.strip() == '':
            chainid = '*'
        resname = one_to_three(dssp[k][1])
        resid = k[1][1]
        name = f'{chainid}_{resid}_{resname}'
        if name in _dssp:
            if dssp[k][2:6] != _dssp[name]:
                print(path, k, name,)
            continue
        _dssp[name] = dssp[k][2:6]

    dssp = {}
    for res in coords:
        name = res[1]
        if name in _dssp:
            dssp[name] = _dssp[name]

    if len(dssp) / len(coords) < 0.95:
        return
    elif len(dssp) != len(coords):
        for res in coords:
            name = res[1]
            if name not in dssp:
                dssp[name] = ('-', 0.0, 360, 360)
    return dssp

def collect_features(coords, atom_feats={}, residue_feats={}):
    """
    [ 
        dict(
            name=residue_name,
            atoms=dict(
                CA=dict(coord, sas, ...),
                ...
                N=dict(coord, sas, ...),
            ),
        ),
    ]
    """
    feats = []
    for i, name, atoms in coords:
        _, _, resname = name.split('_')
        _feat = dict(name=resname)

        if len(residue_feats) > 0:
            _rf = dict()
            for k in residue_feats:
                if residue_feats[k] is None:
                    _rf[k] = None
                else:
                    _rf[k] = residue_feats[k][name]
            _feat.update(_rf)

        group = {}
        for atomname, coord in atoms:
            _name = name + '_' + atomname
            _s = dict(coord=coord)
            # strictly requires extra features existing for each atom in the coords
            _s.update(dict((k, atom_feats[k][_name]) for k in atom_feats))
            assert atomname not in group
            group[atomname] = _s
        _feat['atoms'] = group
        feats.append(_feat)

    return feats

def dump_feature(args, chs, overwrite=False, dump=True,):
    root = args.srcdir
    seqmap = args.seqmap

    n = 0
    for ch in chs:
        ch = ch.stem
        if args.tgtdir is not None:
            featpath = args.tgtdir.joinpath(f'{ch}.feat')
        else:
            featpath = root.joinpath(f'{ch}/{ch}.feat')
        if overwrite or not featpath.exists():
            pdbpath = root.joinpath(f'{ch}/{ch}.reduced.pdb')
            if not pdbpath.exists():
                continue
            coords, model, seq = read_coords(
                pdbpath, ignore_h=not args.use_hydrogen,
            )

            afeats = dict()
            areas = read_areas(root.joinpath(f'{ch}/{ch}.area'))
            afeats['sas'] = areas

            rfeats = dict()
            if args.use_dssp:
                dssp_path = root.joinpath(f'{ch}/{ch}.dssp')
                if not dssp_path.exists():
                    dssp_path = root.joinpath(f'{ch}/{ch}.reduced.dssp')
                dssp = read_dssp(coords, model, dssp_path)
                rfeats['dssp'] = dssp

            feat = collect_features(
                coords,
                atom_feats=afeats,
                residue_feats=rfeats,
            )

            if seqmap is not None and ch in seqmap:
                _seqmap = seqmap[ch]
            else:
                _seqmap = {
                    'ref':ch, 'tbeg':1, 'qbeg':1, 'cigar':f'{len(seq)}M',
                    'refseq':seq,
                }

            sample = dict(
                name=ch,
                feat=feat,
                seqmap=_seqmap,
                seq=seq,
            )

            if not args.no_vertex:
                verts, normals = sample_points(
                    *read_msms(
                        root.joinpath(f'{ch}/{ch}.vert'),
                        root.joinpath(f'{ch}/{ch}.face'),
                    ), resolution=0.8,
                )
                sample['vertex'] = dict(
                    coords=verts,
                    normals=normals,
                )
 
            if dump:
                with open(featpath, 'wb') as h:
                    pickle.dump(sample, h)
                print(f'dump {featpath}')
            else:
                print(sample['seqmap'], feat[0], feat[-1], sample['name'], sample['seq'])
                print(sample['vertex']['coords'].shape)
                print(sample['vertex']['normals'].shape)
            n += 1
    return n


def _get_options():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcdir', type=Path, required=True, help='msms root')
    parser.add_argument('--tgtdir', type=Path, help='target dir')
    parser.add_argument('--ents', type=Path, help='dimer list')
    parser.add_argument('--seqmap', type=Path, help='the pdbseq to a3mseq map')
    parser.add_argument('--refseq', type=Path, help='reference sequence (e.g a3mseq)')
    parser.add_argument('--no-overwrite', action='store_true')
    parser.add_argument('--use-dssp', action='store_true')
    parser.add_argument('--use-hydrogen', action='store_true')
    parser.add_argument('--no-vertex', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = _get_options()

    chs = list(args.srcdir.iterdir())
    if args.ents:
        ents = read_ents(args.ents)
        chs = [ ch for ch in chs if ch.stem in ents ]

    refseqs = read_seqs(args.refseq) if args.refseq else None
    args.seqmap = read_seqmap(args.seqmap, refseqs=refseqs) if args.seqmap else None

    # for debugging
    if args.debug:
        dump_feature(-1, args, chs[:5], overwrite=True, dump=False)
        exit()
    
    if args.tgtdir and not args.tgtdir.exists():
        args.tgtdir.mkdir(parents=True)

    dump_feature(args, chs, not args.no_overwrite),
