import torch

from torch_geometric.data import Data
from torch_scatter import segment_csr
from torch_cluster import radius

from glinter.protein.encoding_utils import ATOMS, ATOM_ONES, encode_aa1, SS8_ONES
from glinter.points.utils import compute_centered_lrf, get_random_rotmat

CA_IDX = ATOMS['CA']
N_IDX = ATOMS['N']
C_IDX = ATOMS['C']
H_IDX = ATOMS['HX']

def plot_points(parts):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    idx = 0
    for pos, c in parts:
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=c,)
    plt.show()

def compute_edge_dist(src, tgt, edge_index, node_dim=0):
    j, i = 0, 1
    cj = src.index_select(node_dim, edge_index[j])
    ci = tgt.index_select(node_dim, edge_index[i])

    return torch.sqrt(torch.sum((cj - ci)**2, dim=-1))

def sum_over_sas(sas, group):
    offsets = torch.cat(
        (torch.LongTensor([0]), torch.cumsum(group, dim=0),),  
        dim=0,
    )
    sas = segment_csr(sas, offsets, reduce='sum',)
    return sas

def build_ca_graph(
    sample, alnidx, r=8, k=-1, rotmat=None,
    only_embed=False, use_distance_graph=False, visualize=False,
):
    if use_distance_graph:
        only_embed = False
    elif only_embed:
        use_distance_graph = False

    srcidx, tgtidx = alnidx # src: pdbseq, tgt: a3mseq
    # build node pos
    coords = sample['COORD'].to(torch.float32)
    if rotmat is not None:
        coords = torch.matmul(
            coords.unsqueeze(1), rotmat.unsqueeze(0)
        ).squeeze(1)

    atoms = sample['ATOM'].to(torch.long)
    group = sample['GROUP'].to(torch.long)
    sas = sample['SAS'].to(torch.float32)

    ca_coords = coords[atoms == CA_IDX]
    
    # compute SAS
    sas = sum_over_sas(sas, group)
    assert sas.size(0) == ca_coords.size(0)

    # compute DSSP
    # rasa = sample['rASA']
    # ss8 = SS8_ONES[sample['SS8']]
 

    # build node embed (ca)
    # encode pdbseq
    seq = encode_aa1(sample['SEQ'], onehot=True)
    pos_encoding = torch.arange(seq.size(0), dtype=torch.float32) / seq.size(0)
    # build pssm
    pssm = torch.zeros((seq.size(0), 20), dtype=torch.float32)
    pssm[srcidx] = sample['pssm'].to(torch.float32)[tgtidx]
    
    node_embed = [sas.view(-1, 1), seq, pos_encoding.view(-1,1), pssm]

    # add DSSP
    # node_embed += [ss8, rasa.view(-1, 1),]

    node_embed = torch.cat(node_embed, dim=-1)
    # if index is not None:
    #     node_embed = node_embed[index]     
    
    if only_embed:
        return node_embed

    # build graph & edge embed (ca)
    if k < 0:
        k = ca_coords.size(0)
    col, row = radius(ca_coords, ca_coords, r, max_num_neighbors=k)
    edge_index = torch.stack((row, col,), dim=0) # (src, tgt)

    # build lrf
    n_coords = coords[atoms == N_IDX]
    c_coords = coords[atoms == C_IDX]
    lrf = compute_centered_lrf(ca_coords, c_coords, n_coords)

    if use_distance_graph:
        edge_dist = compute_edge_dist(
            ca_coords, ca_coords, edge_index
        ).view(-1,1)
        _graph = Data(
            x=node_embed,
            pos=ca_coords,
            edge_index=edge_index,
            edge_embed=edge_dist,
            lrf=lrf,
        )

    else:
        # if index is not None:
        #     lrf = lrf[index]
        _graph = Data(
            x=node_embed,
            pos=ca_coords,
            edge_index=edge_index,
            lrf=lrf,
        )
        if visualize:
            c = c_coords - ca_coords
            n = n_coords - ca_coords
            c = torch.bmm(c.unsqueeze(1), lrf).squeeze(1)
            n = torch.bmm(n.unsqueeze(1), lrf).squeeze(1)
            ca = ca_coords - ca_coords
            c = c.numpy()
            n = n.numpy()
            ca = ca.numpy()
            plot_points(
                [
                    (ca, 'r',),
                    (c, 'b',),
                    (n, 'g',),
                ],
            )

    return _graph

def build_atom_graph(sample, r=8, k=-1, rotmat=None, remove_hydrogen=False):
    """
    from atom to ca
    """
    # build node pos
    coords = sample['COORD'].to(torch.float32)
    if rotmat is not None:
        coords = torch.matmul(
            coords.unsqueeze(1), rotmat.unsqueeze(0)
        ).squeeze(1)

    group = sample['GROUP'].to(torch.long)
    atoms = sample['ATOM'].to(torch.long)
    sas = sample['SAS'].to(torch.float32)

    ca_coords = coords[atoms == CA_IDX]

    # build residue group
    residue_index = [] # residue group
    node_index = torch.arange(group.size(0), dtype=torch.long)
    for idx, sz in zip(node_index, group):
        residue_index.append(node_index.new_full((int(sz),), idx))
    residue_index = torch.cat(residue_index, dim=0)

    if remove_hydrogen:
        _mask = atoms != H_IDX
        atoms = atoms[_mask]
        residue_index = residue_index[_mask]
        coords = coords[_mask]
        sas = sas[_mask]

    # TODO: add residue types to node embeddings
    seq = encode_aa1(sample['SEQ'], onehot=True)
    res_embed = seq.index_select(0, residue_index)
    # build node embed (atoms)
    node_embed = torch.cat(
        (ATOM_ONES[atoms], sas.view(-1, 1), res_embed,), dim=-1,
    )

    assert residue_index.size(0) == coords.size(0)

    # build graph & edge embed (ca)
    if k < 0:
        k = coords.size(0)

    col, row = radius(coords, ca_coords, r, max_num_neighbors=k)
    edge_index = torch.stack((row, col,), dim=0) # (src, tgt)

    edge_embed = []
    # if atom and ca are in the same residue, residue_edge_embed is 1
    residue_edge_embed = (
        residue_index.index_select(0, edge_index[1]) == edge_index[0]
    )
    edge_embed.append(residue_edge_embed.view(-1,1))

    # edge_dist = compute_edge_dist(coords, ca_coords, edge_index,)
    # edge_embed.append(edge_dist.view(-1,1))

    if len(edge_embed) > 1:
        edge_embed = torch.cat(edge_embed, dim=-1,)
    else:
        edge_embed = edge_embed[0]

    _graph = Data(
        x=node_embed,
        pos=coords,
        edge_index=edge_index,
        edge_embed=edge_embed,
    )

    return _graph

def build_surface_graph(sample, r=4, k=-1, rotmat=None, visualize=False):
    coords = sample['COORD'].to(torch.float32)
    vcoords = sample['vcoord'].to(torch.float32)
    vnormals = sample['vnormal'].to(torch.float32)

    if rotmat is not None:
        coords = torch.matmul(
            coords.unsqueeze(1), rotmat.unsqueeze(0)
        ).squeeze(1)
        vcoords = torch.matmul(
            vcoords.unsqueeze(1), rotmat.unsqueeze(0)
        ).squeeze(1)
        vnormals = torch.matmul(
            vnormals.unsqueeze(1), rotmat.unsqueeze(0)
        ).squeeze(1)

    atoms = sample['ATOM'].to(torch.long)
    ca_coords = coords[atoms == CA_IDX]

    if k < 0:
        k = vcoords.size(0)

    col, row = radius(vcoords, ca_coords, r, max_num_neighbors=k)
    edge_index = torch.stack((row, col,), dim=0) # (src, tgt)
    _graph = Data(
        pos=vcoords,
        nor=vnormals,
        edge_index=edge_index,
    )
    if visualize:
        import matplotlib.pyplot as plt
        from cagcn.points.mesh import plot_normals
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        _x, _y, _z = ca_coords[0]
        ax.scatter(_x, _y, _z)
        index = edge_index[0][edge_index[1] == 0]
        index = sorted(index)
        plot_normals(
            vcoords.numpy()[index[:10]], vnormals.numpy()[index[:10]], magnify=10, ax=ax
        )

    return _graph

if __name__ == '__main__':
    """
    run _graph.py to check dataset and visualize built graphs
    """
    import sys
    import pickle
    from pathlib import Path
    from tqdm import tqdm

    from cagcn.protein import cigar_to_index

    with open(Path(sys.argv[1]), 'rb') as h:
        data = pickle.load(h)

    for k, sample in tqdm(list(data.items())):
        mten = sample['rec']
        print(mten['name'])
        seqmap = mten['seqmap']
        alnidx = cigar_to_index(
            seqmap['cigar'],seqmap['qbeg']-1, seqmap['tbeg']-1,
        )
        # rec_cag = build_ca_graph(mten, alnidx, visualize_transformation=True)
        # rec_sug = build_surface_graph(
        #     mten, r=4, rotmat=get_random_rotmat(3, axis=2), visualize=True,
        # )
        # rec_sug = build_surface_graph(
        #     mten, r=4, visualize=True,
        # )
        print(rec_sug)
        break
        # rec_atg = build_atom_graph(mten)
