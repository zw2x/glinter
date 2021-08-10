import sys
from pathlib import Path

from Bio.PDB import PDBParser

from glinter.protein.pdb_utils import get_pdbseq

def pdbseq(path):
    parser = PDBParser(QUIET=True)
    chains = list(parser.get_structure('', path).get_models())[0].get_list()
    assert len(chains) == 1, f"{path} contain more than one chain"
    seq = get_pdbseq(chains[0], thr=-1)
    assert seq is not None, f"{path} does not have at least one valid residue"
    return seq

if __name__ == '__main__': 
    path = Path(sys.argv[1])
    seq = pdbseq(path)
    with open(Path(sys.argv[2]), 'wt') as h:
        name = path.stem
        h.write(f'>{name}\n{seq}')
