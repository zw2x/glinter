# modified by zw2x
# ********************
# xyzrn.py: Read a pdb file and output it is in xyzrn for use in MSMS
# Pablo Gainza - LPDI STI EPFL 2019
# This file is part of MaSIF.
# Released under an Apache License 2.0

from pathlib import Path

import numpy as np

from Bio.PDB import PDBParser
from glinter.protein import get_atom_residues
from glinter.protein.chemistry import radii

def chain_to_xyzrn(pdb_path, xyzrn_path, ignore_h=False):
    """
    from .pdb file to .xyzrn file (xyz-radius-name)
    Args:
        pdb_path (path): pdb path
        xyzrn_dir (path): xyzrn dir 
    """
    pdb_path = Path(pdb_path)
    path = Path(xyzrn_path)

    parser = PDBParser(QUIET=True)
    chains = list(parser.get_structure('', pdb_path).get_models())[0].get_list()
    with open(path, 'wt') as handle: 
        for chain in chains: 
            chainid = chain.id
            if chainid.strip() == '':
                chainid = '*'
            for residue, atoms in get_atom_residues(chain, ignore_h=ignore_h,):
                resid = residue.get_id()[1]
                resname = residue.resname
                for atom in atoms: # choose one if the atom is disordered
                    atom_type = atom.get_id()[0]
                    if atom_type not in radii:
                        continue
                    x, y, z = atom.get_coord()
                    coords = "{:.06f} {:.06f} {:.06f}".format(x, y, z)

                    full_id = "{}_{:d}_{}_{}".format(
                        chainid, resid, resname, atom.get_id(),
                    )

                    handle.write(
                        coords + " " + radii[atom_type] + " 1 " + full_id + "\n"
                    )


if __name__ == '__main__':
    import sys
    chain_to_xyzrn(sys.argv[1], sys.argv[2])
