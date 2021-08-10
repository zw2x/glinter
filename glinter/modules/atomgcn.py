import torch
from torch import nn
from torch_geometric.nn import knn_interpolate

from glinter.modules.atomconv import AtomConv, AtomConvDynamic

class MGGBlock(nn.Module):
    """
    Multi-graph grouping block. (parallel graphs)
    """
    def __init__(self, in_dim, out_dim, graph_kwargs,):
        """
        Args:
            in_dim (int): the input dimension of the query nodes
            out_dim (int): the output dimension of the query nodes
            graph_kwargs (Tuple[Dict]):
                the network dimensions of each graph convolution layers,
                each item includes:
                    (1) the node dimension of the src graph
                    (2) the edge dimension of the src graph
                    (3) the local dimension using in the graph convolution layer
                    (4) the output dimension of the graph convolution layer
        """
        super().__init__()

        self.query_dim = in_dim
        self.tgt_dim = out_dim

        self.conv_layers = nn.ModuleList()
        concated_dim = 0
        for params in graph_kwargs:
            node_dim = params['node_dim']
            del params['node_dim']
            _conv = self._build_conv_layer(
                (node_dim, in_dim), **params
            )
            self.conv_layers.append(_conv)
            concated_dim += params['tgt_dim']

        self.global_nn = None
        if out_dim != concated_dim:
            self.global_nn = nn.Sequential(
                nn.Linear(concated_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
            )

    def forward(self, x, graphs, pos=None, lrf=None):
        """
        Args:
            x (Tensor) : the input node embeddings of the target nodes
            graphs (Tuple[Batch]) : all source graphs
            pos (Tensor) : the coordinates of the target nodes
            lrf (Tensor) : the local reference frame of the target nodes
        Returns:
            g (Tensor) : the updated node embeddings of the target nodes
        """
        assert len(graphs) == len(self.conv_layers)
        g = []
        for graph, conv in zip(graphs, self.conv_layers):
            if conv.use_pos:
                _pos = (graph.pos, pos)
                _lrf = lrf
            else:
                _pos = None
                _lrf = None
            _g = conv(
                (getattr(graph, 'x', None), x),
                edge_index=getattr(graph, 'edge_index', None),
                edge_embed=getattr(graph, 'edge_embed', None),
                pos=_pos, lrf=_lrf, nor=getattr(graph, 'nor', None),
            )[0]
            g.append(_g)

        g = torch.cat(g, dim=-1)

        if self.global_nn is not None:
            g = self.global_nn(g)

        return g

    def _build_conv_layer(
        self, node_dim, edge_dim=0, local_dim=128, global_dim=128,
        use_gate_nn=False, use_dynamic=False, use_pos=True, **kwargs,
    ):
        assert isinstance(node_dim, tuple)

        src_dim, query_dim = node_dim
        embed_dim = src_dim + edge_dim

        if kwargs.get('use_concat', False):
            embed_dim += query_dim

        if use_pos:
            embed_dim += 3

        if kwargs.get('use_nor', False):
            embed_dim += 3

        local_nn = nn.Sequential(
            nn.Linear(embed_dim, local_dim),
            nn.BatchNorm1d(local_dim),
            nn.ReLU(),
        )

        if use_gate_nn:
            gate_nn = nn.Sequential(
                nn.Linear(embed_dim, local_dim),
                nn.BatchNorm1d(local_dim),
                nn.Sigmoid(),
            )
        else:
            gate_nn = None

        global_nn = nn.Sequential(
            nn.Linear(local_dim, global_dim),
            nn.BatchNorm1d(global_dim),
            nn.ReLU(),
        )


        if not use_dynamic:
            conv = AtomConv(
                local_nn=local_nn,
                global_nn=global_nn,
                gate_nn=gate_nn,
                use_pos=use_pos,
                **kwargs,
            )
        else:
            conv = AtomConvDynamic(
                local_nn=local_nn,
                global_nn=global_nn,
                gate_nn=gate_nn,
                **kwargs,
            )

        return conv


class SABlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, local_dim=128, r=12, k=-1, rate=0.5, aggr='max',
    ):
        super().__init__()

        embed_dim = in_dim + 3
        local_nn = nn.Sequential(
            nn.Linear(embed_dim, local_dim),
            nn.BatchNorm1d(local_dim),
            nn.ReLU(),
        )

        global_nn = nn.Sequential(
            nn.Linear(local_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

        self.conv = AtomConvDynamic(
            local_nn=local_nn, global_nn=global_nn, rate=rate, r=r, k=k, aggr=aggr,
            use_global_concat=False,
        )

    def forward(self, x, pos, lrf):
        x, pos, lrf = self.conv(x, pos, lrf)
        return x, pos, lrf

class FPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=3):
        """
        Args:
            in_dim (int): the concated dimension of the outputs from the
                interpolation and the corresponding abstraction layers
        """
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.k = k
        self.nn = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x, pos_x, pos_y, y=None):
        k = min(self.k, pos_y.size(0))
        _y = knn_interpolate(x, pos_x, pos_y, k=k)
        if y is None:
            y = _y
        else:
            y = torch.cat((_y, y), dim=-1)

        y = self.nn(y)
        return y

class AtomGCN(nn.Module):
    def __init__(
        self, in_dim, out_dim, src_graphs,
        num_sa=1, sa_dims=(128,), rates=(0.5,), rs=(12,), ks=(-1,), fp_dims=(128,),
        use_fp=True,
    ):
        """
        AtomGCN starts with pooling raw features from source graphs to the query
        nodes, then it applies set abstraction and then feature propagtion to
        update embeddings of the query nodes.

        Args:
            src_graphs (List[Tuple]): src graphs
            num_sa (int): number of set abstraction blocks
            rates (Tuple[float]): the sampling rate for each SA block
            rs (Tuple[float]): the radius for each SA block
            ks (Tuple[int]): the max number of neighbors for each SA block
        """
        super().__init__()

        self.src_block = MGGBlock(in_dim, out_dim, src_graphs,)
        local_dim = out_dim

        # build point++ using atomconv
        self.sa_blocks = None
        self.fp_blocks = None

        if num_sa > 0:
            self.sa_blocks = nn.ModuleList()
            assert (
                num_sa == len(rates) and num_sa == len(rs) and
                num_sa == len(ks) and num_sa == len(sa_dims) and
                num_sa == len(fp_dims)
            )

            _in_dim = local_dim
            for sa_dim, rate, r, k in zip(sa_dims, rates, rs, ks):
                sa = SABlock(_in_dim, sa_dim, r=r, rate=rate, k=k)
                self.sa_blocks.append(sa)
                _in_dim = sa_dim

            if use_fp: # build feature propagation
                self.fp_blocks = nn.ModuleList()
                _in_fp_dim = sa_dims[-1]
                for i in range(num_sa):
                    _sa_dim = sa_dims[-i-2] if i != num_sa - 1 else local_dim
                    fp_dim = fp_dims[i]
                    fp = FPBlock(_sa_dim + _in_fp_dim, fp_dim)
                    self.fp_blocks.append(fp)
                    _in_fp_dim = fp_dim

        self._init_weights()

    def _init_weights(self):
        pass

    def forward(self, x, pos, lrf, src_graphs,):
        x = self.src_block(x, src_graphs, pos=pos, lrf=lrf)
        if self.sa_blocks is None:
            return x

        if self.fp_blocks is not None:
            ags = [(x,pos)] # abstracted graphs

        for sa in self.sa_blocks:
            x, pos, lrf = sa(x, pos, lrf)
            if self.fp_blocks is not None:
                ags.append((x, pos,))

        if self.fp_blocks is None:
            return x, pos, lrf

        x, pos_x = ags[-1]
        for i, fp in enumerate(self.fp_blocks):
            y, pos_y = ags[-i-2]
            x = fp(x, pos_x, pos_y, y)
            pos_x = pos_y
        return x

if __name__ == '__main__':
    import torch_geometric.data as td

    pos_j = torch.randn(4,3)
    x_j = torch.Tensor([[-1,], [-1,], [1,], [1,]])
    src_graph = td.Batch.from_data_list([td.Data(pos=pos_j, x=x_j,)])

    pos_i = torch.randn(2,3)
    x_i = torch.Tensor([[0,], [0,]])
    xlrf = torch.randn((2,3,3))
    xtgt = td.Batch.from_data_list([td.Data(pos=pos_i, x=x_i, lrf=xlrf)])

    conv_module = AtomGCN(1, 128, [(1, 0, 64, 128),])
    print(conv_module(xtgt.x, xtgt.pos, xtgt.lrf, (src_graph,)).size())
