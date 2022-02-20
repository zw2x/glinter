import sys

import numpy as np

from alphafold.common import protein, residue_constants

with open(sys.argv[1]) as fh:
    pdb_string = fh.read()
    p = protein.from_pdb_string(pdb_string)

ca_atom_order = residue_constants.atom_order['CA']
pos = p.atom_positions[:, ca_atom_order]
pos1 = pos[p.chain_index == 0]
pos2 = pos[p.chain_index == 1]
dist = np.sqrt(np.sum((pos1[:,None] - pos2[None,:])**2, axis=-1))
sorted_inds = np.argsort(dist.reshape(-1))
inds1 = sorted_inds // len(pos2)
inds2 = sorted_inds % len(pos2)
with open(sys.argv[2], 'wt') as fh:
    for i1, i2 in zip(inds1, inds2):
        fh.write(f'{i1} {i2} {dist[i1, i2]:.4f}\n')
