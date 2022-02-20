import os
import sys
import numpy as np

from alphafold.common import protein

prefix = sys.argv[1]
with open(f'{prefix}.pdb') as fh:
    pdb_string = fh.read()
    p = protein.from_pdb_string(pdb_string)

for chain_id in np.unique(p.chain_index):
    chain_id = int(chain_id)
    pdb_chain_id = protein.PDB_CHAIN_IDS[chain_id]
    chain_pdb_path = f'{prefix}_{pdb_chain_id}.pdb'
    chain_mask = p.chain_index == chain_id
    chain_pdb_string = protein.to_pdb(protein.Protein(
        atom_positions=p.atom_positions[chain_mask],
        atom_mask=p.atom_mask[chain_mask],
        aatype=p.aatype[chain_mask],
        residue_index=p.residue_index[chain_mask],
        chain_index=p.chain_index[chain_mask],
        b_factors=p.b_factors[chain_mask]
    ))
    with open(chain_pdb_path, 'wt') as fh:
        fh.write(chain_pdb_string)
