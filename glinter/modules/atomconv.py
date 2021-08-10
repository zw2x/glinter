import torch
from torch import Tensor

from torch_cluster import radius, fps
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops

class AtomConv(MessagePassing):
    def __init__(
        self, local_nn=None, global_nn=None, gate_nn=None,
        aggr='max', use_pos=False, use_concat=False, use_nor=False, **unused,
    ):
        """
        use_pos (bool) : use pos and lrf when computing messages
        use_global_concat (bool) :
            concat query residuals with query outputs before passing to global_nn
        """
        super().__init__(node_dim=0, aggr=aggr)
        self.local_nn = local_nn
        self.global_nn = global_nn
        self.gate_nn = gate_nn

        self.use_pos = use_pos
        self.use_concat = use_concat
        self.use_nor = use_nor

    def __lift__(self, src, edge_index, dim, node_dim=None):
        if node_dim is None:
            node_dim = self.node_dim

        if isinstance(edge_index, Tensor):
            index = edge_index[dim]
        else:
            raise ValueError(
                f'the type of edge_index "{type(edge_index)}" is not allowed'
            )

        return src.index_select(node_dim, index)

    def forward(
        self, x, edge_index, edge_embed=None, pos=None, lrf=None, nor=None,
    ):
        if isinstance(x, tuple):
            x_j, x_i = x
        else:
            x_j, x_i = x, x

        j, i = 0, 1
        if x_j is not None:
            _x_j = self.__lift__(x_j, edge_index, j) # source
        else:
            _x_j = None
        if x_i is not None:
            _x_i = self.__lift__(x_i, edge_index, i) # target
        else:
            _x_i = None

        if self.use_pos:
            if isinstance(pos, tuple):
                pos_j, pos_i = pos
            else:
                pos_j, pos_i = pos, pos
            _pos_j = self.__lift__(pos_j, edge_index, j) # source pos
            _pos_i = self.__lift__(pos_i, edge_index, i) # target pos
            _lrf_i = self.__lift__(lrf, edge_index, i, node_dim=0) # target lrf
        else:
            pos_i, _pos_i, _pos_j, _lrf_i = pos, None, None, None

        if self.use_nor:
            _nor_j = self.__lift__(nor, edge_index, j) # source nor
        else:
            _nor_j = None

        out = self.message(
            _x_i, _x_j,
            edge_embed=edge_embed, pos_i=_pos_i, pos_j=_pos_j, lrf_i=_lrf_i,
            nor_j=_nor_j
        )

        out = self.aggregate(
            out, edge_index[i], dim_size=x_i.size(self.node_dim),
        )

        y = self.update(out, x_i)

        return y, pos_i, lrf

    def message(
        self, x_i=None, x_j=None, edge_embed=None, pos_i=None, pos_j=None,
        lrf_i=None, nor_j=None,
    ):
        # build x
        _y = []
        if self.use_concat and x_i is not None:
            _y.append(x_i)
        if x_j is not None:
            _y.append(x_j)
        if edge_embed is not None:
            _y.append(edge_embed)
        # add pos
        if self.use_pos and pos_j is not None and pos_i is not None:
            _pos = torch.matmul((pos_j - pos_i).unsqueeze(-2), lrf_i).squeeze(-2)
            _y.append(_pos)
        # add nor
        if self.use_nor and nor_j is not None:
            _nor = torch.matmul(nor_j.unsqueeze(-2), lrf_i).squeeze(-2)
            _y.append(_nor)

        if len(_y) > 1:
            y = torch.cat(_y, dim=-1)
        elif len(_y) == 1:
            y = _y[0]
        else:
            y = None

        if self.local_nn is not None:
            f = self.local_nn(y)
        else:
            f = y

        if self.gate_nn is not None:
            g = self.gate_nn(y)
            f = g * f

        return f

    def update(self, f, x):
        if self.global_nn is not None:
            y = self.global_nn(f)
        else:
            y = f

        return y

class AtomConvDynamic(AtomConv):
    def __init__(
        self, local_nn=None, global_nn=None, gate_nn=None,
        aggr='max', rate=1, r=12, k=-1, use_global_concat=False, **unused,
    ):
        super().__init__(
            local_nn=local_nn, global_nn=global_nn, gate_nn=gate_nn, aggr=aggr,
            use_pos=True, use_global_concat=use_global_concat,
        )
        self.rate = rate
        self.r = r
        self.k = k

    def forward(self, x, pos, lrf, **unused):

        if isinstance(x, tuple):
            x_j, x_i = x
        else:
            x_j, x_i = x, x

        if isinstance(pos, tuple):
            pos_j, pos_i = pos
        else:
            pos_j, pos_i = pos, pos

        j, i = 0, 1

        # build sampled graph
        if self.rate < 1 and self.rate > 0:
            idx = fps(pos_i, ratio=self.rate, random_start=self.training)
            # if self.rate < 0.5:
            #     idx = fps(pos_i, ratio=self.rate, random_start=self.training)
            # else:
            #     idx = spaced_sampling(pos_i, rate=self.rate,)
            pos_i = pos_i[idx]
            x_i = x_i[idx]
            lrf_i = lrf[idx]
        else:
            lrf_i = lrf

        if self.k <= 0:
            k = x_j.size(self.node_dim)
        else:
            k = self.k

        col, row = radius(pos_j, pos_i, self.r, max_num_neighbors=k)
        edge_index = torch.stack((row, col,), dim=0) # (src, tgt)
        if self.k > 0:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=pos_i.size(0))

        _x_j = self.__lift__(x_j, edge_index, j) # source
        _x_i = self.__lift__(x_i, edge_index, i) # target

        _pos_j = self.__lift__(pos_j, edge_index, j) # source pos
        _pos_i = self.__lift__(pos_i, edge_index, i) # target pos

        _lrf_i = self.__lift__(lrf_i, edge_index, i, node_dim=-3) # target lrf

        out = self.message(
            _x_i, _x_j, pos_i=_pos_i, pos_j=_pos_j, lrf_i=_lrf_i,
        )
        out = self.aggregate(
            out, edge_index[i], dim_size=x_i.size(self.node_dim),
        )
        x_i = self.update(out, x_i)

        return x_i, pos_i, lrf_i

def spaced_sampling(pos, rate=1, minlen=16,):
    idx = torch.arange(
        pos.size(0), dtype=torch.long, device=pos.device,
    )
    if rate >= 1:
        return idx
    step = int(1 / (1 - rate))
    if step < 2:
        return idx
    mask = torch.ones(pos.size(0), dtype=torch.bool, device=pos.device)
    neg = torch.arange(
        0, pos.size(0), step=step, dtype=torch.long, device=pos.device,
    )
    mask[neg] = 0
    idx = idx[mask]
    return idx

if __name__ == '__main__':
    import torch_geometric.data as td

    xpos_j = torch.randn(4,3)
    x_j = torch.Tensor([[-1,], [-1,], [1,], [1,]])
    xsrc = td.Batch.from_data_list([td.Data(pos=xpos_j, x=x_j,)])

    xpos_i = torch.randn(2,3)
    x_i = torch.Tensor([[0,], [0,]])
    xlrf = torch.randn((2,3,3))
    xtgt = td.Batch.from_data_list([td.Data(pos=xpos_i, x=x_i, lrf=xlrf)])

    print(xpos_j, xpos_i)
    conv = AtomConvDynamic(rate=0.8)
    print(conv((xsrc.x, xtgt.x), (xsrc.pos, xtgt.pos), xtgt.lrf))
