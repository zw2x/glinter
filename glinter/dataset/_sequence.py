import torch

from torch_scatter import segment_csr
from torch_cluster import radius

# from deepdock.points.atoms import ATOMS, ATOM_ONES
# from deepdock.protein.sequence import onehot
# from deepdock.points.geometry import compute_centered_lrf

# CA_IDX = ATOMS['CA']

def build_sequence(sample, beg=0, end=-1):
    """
    edge_index (source_to_target, i.e. target(0), source(1))
    """
    # build node pos
    coords = sample['COORD']
    atoms = sample['ATOM']
    ca_coords = coords[atoms == CA_IDX]
    
    # compute SAS
    offsets = torch.cat(
        (torch.LongTensor([0]), torch.cumsum(sample['GROUP'], dim=0),),  
        dim=0,
    )
    sas = segment_csr(sample['SAS'], offsets, reduce='sum',)
    assert sas.size(0) == ca_coords.size(0)
    
    if end <= 0:
        end = ca_coords.size(0)

    # build node embed (ca)
    pssm = sample['PSSM']
    seq = onehot(sample['SEQ'])
    node_embed = torch.cat((seq, pssm, sas.view(-1, 1)), dim=-1,)[beg:end]

    return node_embed.transpose(0,1)
