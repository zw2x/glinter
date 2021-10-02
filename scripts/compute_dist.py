import sys
from pathlib import Path
import pickle
from tqdm import tqdm

import numpy as np
import torch

from Bio.PDB import PDBParser
from glinter.protein.pdb_utils import get_coords

import torch
from torch_cluster import radius_graph, radius
from torch_scatter import segment_csr

def _prepare(*data):
    return [ d.cuda() if d is not None else None for d in data ]

def compute_dist(x, y, ix=None, iy=None):
    x, y, ix, iy = _prepare(x, y, ix, iy)

    d = torch.sqrt(torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=-1))

    if ix is not None:
        d = segment_csr(d, ix, reduce='min')
        
    if iy is not None:
        d = segment_csr(d, iy.unsqueeze(0), reduce='min')

    return d.cpu()

def cat_coords(coords):
    coord, size = [], []
    for c, s in coords:
        if c is None or s is None:
            continue
        coord.append(c)
        size.append(s)
    if len(coord) == 0:
        return None, None
    coord = torch.cat(coord, dim=0)
    size = torch.cat(size, dim=0)
    return coord, size

if __name__ == '__main__':
    pt1 = sys.argv[1]
    pt2 = sys.argv[2]
    parser = PDBParser(QUIET=True)
    chains1 = parser.get_structure('', pt1).get_list()[0].get_list()
    chains2 = parser.get_structure('', pt2).get_list()[0].get_list()

    coord1, size1 = cat_coords(
        [ get_coords(c, ignore_h=True) for c in chains1 ]
    )
    coord2, size2 = cat_coords(
        [ get_coords(c, ignore_h=True) for c in chains2 ]
    )

    if coord1 is not None and coord2 is not None:
        ix = torch.cat([torch.LongTensor([0]), torch.cumsum(size1, dim=0)], dim=0)
        iy = torch.cat([torch.LongTensor([0]), torch.cumsum(size2, dim=0)], dim=0)
        coord1, ix = _prepare(coord1, ix)
        coord2, iy = _prepare(coord2, iy)
        d = compute_dist(coord1, coord2, ix=ix, iy=iy)
        with open(sys.argv[3], 'wb') as dh:
            pickle.dump(d, dh)
