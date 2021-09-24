import Bio
import torch

from .encoding_utils import three_to_one

__all__ = [
    'get_residues',
    'get_atoms',
    'get_pdbseq'
]

def get_pdbseq(chain, thr=0.95, return_positions=False):
    if isinstance(chain, Bio.PDB.Chain.Chain):
        residues = get_residues(chain,)
    else:
        residues = chain
    if len(residues) == 0:
        return
    if thr < 0 or len(residues) / int(residues[-1].get_id()[1]) > thr:
        _seq, _pos = [], []
        for residue in residues:
            _seq.append(three_to_one(residue.get_resname()))
            _pos.append(residue.id[1])
        if return_positions:
            return ''.join(_seq), _pos
        else:
            return ''.join(_seq)
    else:
        return
 
def get_residues(chain,):
    residues = []
    for residue in chain.get_list(): # choose one if the residue is disordered
        hetatom, resid, icode = residue.get_id()
        if hetatom != ' ' or icode != ' ':
            continue
        resname = residue.get_resname()
        _names = set(atom.get_id() for atom in residue.get_list())
        if not all(_ in _names for _ in ('N', 'CA', 'C')):
            # ignore "incomplete" residues
            continue
        residues.append(residue)
    return residues

def get_atoms(residue, ignore_h=True, return_ca=False,):
    if return_ca:
        return residue['CA']

    _atoms = residue.get_list()
    atoms = []
    for atom in _atoms: # choose one if the atom is disordered
        if ignore_h and atom.get_id().startswith('H'):
            continue
        atoms.append(atom)

    return atoms
