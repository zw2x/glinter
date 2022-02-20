import sys
import pickle
import numpy as np

from alphafold.common import residue_constants

def translate_sequence(aatype):
    seq = []
    for aa in aatype:
        aa = residue_constants.restypes_with_x_and_gap[aa]
        seq.append(aa)
    return ''.join(seq)

with open(sys.argv[1], 'rb') as fh:
    feature = pickle.load(fh)

msa = []
asym_id = feature['asym_id']
lengths = [sum(asym_id == i) for i in range(1, int(np.max(asym_id))+1)]

for i, aatype in enumerate(feature['msa'][:128]):
    if i:
        desc = f'seq-{i}'
    else:
        desc = '&'.join([ 
            f'{chain_id} 0 / {length} ->' for chain_id, length in 
            enumerate(lengths)
        ])
    msa.append(f'>{desc}')
    seq = translate_sequence(aatype)
    msa.append(seq)

print('\n'.join(msa))
