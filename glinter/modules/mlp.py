import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self, inplane, layers, 
        use_final_activation=False,
    ):
        super().__init__()
        self.activation = F.relu

        self.layers = layers
        self.mlp = nn.ModuleList()
        for outplane in self.layers:
            m = [ 
                nn.Conv1d(inplane, outplane, 1,),
                nn.BatchNorm1d(outplane),
            ]
            self.mlp.append(nn.Sequential(*m))
            inplane = outplane

        self._len = len(self.layers)
        self.use_final_activation = use_final_activation

    def forward(self, x, return_all=False):
        out = []

        for i, m in enumerate(self.mlp):
            x = m(x)
            if self.activation is not None:
                if i < self._len - 1 or self.use_final_activation:
                    x = self.activation(x)
            if return_all:
                out.append(x)

        if return_all:
            return out

        return x
