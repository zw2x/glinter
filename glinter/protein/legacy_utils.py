import torch
from .encoding_utils import three_to_one

__all__ = [
    'get_atom_residues',
]

def get_pdbseq(chain, thr=0.9, return_resids=False):
    if return_resids:
        residues, resids = _get_residues(chain, return_resids=True)
    else:
        residues = _get_residues(chain, return_resids=False)
    if len(residues) == 0:
        if return_resids:
            return None, None
        return
    if len(residues) / int(residues[-1].get_id()[1]) > thr:
        _seq = []
        for residue in residues:
            _seq.append(three_to_one(residue.get_resname()))
        if return_resids:
            return ''.join(_seq), resids
        else:
            return ''.join(_seq)
    else:
        if return_resids:
            return None, None
        return

def _get_residues(chain, return_resids=False):
    residues = []
    resids = []
    for residue in chain.get_list(): # choose one if the residue is disordered
        hetatom, resid, icode = residue.get_id()
        if return_resids:
            resids.append(resid)
        if hetatom != ' ' or icode != ' ':
            continue
        resname = residue.get_resname()
        _names = set(atom.get_id() for atom in residue.get_list())
        if not all(_ in _names for _ in ('N', 'CA', 'C')):
            # ignore "incomplete" residues
            continue
        residues.append(residue)
    if return_resids:
        return residues, resids
    return residues

def get_atom_residues(chain, return_ca=False, ignore_h=True):
    residues = []
    for residue in chain.get_list(): # choose one if the residue is disordered
        hetatom, resid, icode = residue.get_id()
        if hetatom != ' ' or icode != ' ':
            continue
        resname = residue.get_resname()
        _atoms = residue.get_list()
        _names = set(atom.get_id() for atom in _atoms)
        if not all(_ in _names for _ in ('N', 'CA', 'C')):
            # ignore "incomplete" residues
            continue
        if return_ca:
            residues.append((residue, [residue['CA']],))
            continue
        # TODO: remove atoms, since residue include all atom information
        atoms = []
        for atom in _atoms: # choose one if the atom is disordered
            if ignore_h and atom.get_id().startswith('H'):
                continue
            atoms.append(atom)
        residues.append((residue, atoms))
    return residues

def get_coord(atoms):
    coord = torch.FloatTensor([atom.coord for atom in atoms])
    return coord

def get_coords(chain, return_ca=False, ignore_h=False):
    atoms = [ 
        _[1] for _ in get_residues(chain, return_ca=return_ca, ignore_h=ignore_h) 
    ]
    if len(atoms) == 0:
        return None, None
    sizes = torch.LongTensor([ len(_) for _ in atoms ])
    coords = torch.cat([get_coord(_) for _ in atoms], dim=0,)
    return coords, sizes
