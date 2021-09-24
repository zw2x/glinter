from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from glinter.model.atomgcn import AtomGCN
# from glinter.module.conv import make_layer, ResNet, BasicBlock2d, Bottleneck2d
from glinter.esm_embed import load_esm_model
from glinter.modules.atomgcn import AtomGCN
from glinter.modules.conv import make_layer, ResNet, BasicBlock2d, Bottleneck2d

# copied from esm
def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized

def build_eval_str_list(sep=',', cast=float):
    def _eval_str_list(s):
        return [ cast(_) for _ in s.split(sep) ]
    return _eval_str_list

class MSAModel(nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--node-embed-dim', type=int, default=43,)
        parser.add_argument('--num-1d-layers', type=int, default=1,)
        parser.add_argument('--rates', type=build_eval_str_list(), default=[0.5],)
        parser.add_argument('--rs', type=build_eval_str_list(), default=[12],)
        parser.add_argument(
            '--row-attn-op', type=str, choices=[
                'lower_tri', 'upper_tri', 'sym', 'apc',
            ],
            default='sym',
        )

    def __init__(
        self, args, esm_embed=None, prepend_bos=False, gen_esm=False
    ):
        super().__init__()
        self.args = args
        
        self._gen_esm = gen_esm
        if self._gen_esm:
            assert esm_embed is not None

        self.esm_embed = None
        self.prepend_bos = prepend_bos

        embed_dim = 0
        if 'esm' in args.feature:
            assert esm_embed is not None
            self.esm_embed = esm_embed
            if not self._gen_esm:
                embed_dim += 144

        elif 'pickled-esm' in args.feature:
            embed_dim += 144

        if 'ccm' in args.feature:
            embed_dim += 1 # using ccm instead of msa embeddings

        encoder_1d, _encoder_1d_dim = self._build_encoder_1d()
        self.encoder_1d = encoder_1d
        embed_dim += _encoder_1d_dim * 2

        if not self._gen_esm:
            _conv1 = make_layer(BasicBlock2d, embed_dim, 96, 16)
            self.resnet = ResNet([_conv1,]) 
            self.fc = nn.Conv2d(96, 2, kernel_size=1)


    def _build_encoder_1d(self,):
        encoder_1d = None
        embed_dim = self.args.node_embed_dim
        output_dim = 0
        num_layers = self.args.num_1d_layers
        src_graphs = []

        if 'ca-embed' in self.args.feature:
            _local_dim = 128
            output_dim = 128
            encoder_1d = nn.Sequential(
                nn.Conv1d(embed_dim, _local_dim, 5, padding=2),
                nn.BatchNorm1d(_local_dim),
                nn.ReLU(),
                nn.Conv1d(_local_dim, _local_dim, 5, padding=2),
                nn.BatchNorm1d(_local_dim),
                nn.ReLU(),
                nn.Conv1d(_local_dim, _local_dim, 5, padding=2),
                nn.BatchNorm1d(_local_dim),
                nn.ReLU(),
                nn.Conv1d(_local_dim, output_dim, 5, padding=2),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
            )

        if self.args.feature.use('coordinate-ca-graph','distance-ca-graph'):
            _local_dim = 128
            src_graphs.append(
                dict(
                    node_dim=embed_dim,
                    use_pos=self.args.feature.use('coordinate-ca-graph'),
                    tgt_dim=_local_dim,
                    use_concat=True,
                    edge_dim=(
                        1 if self.args.feature.use('distance-ca-graph') else 0
                    ),
                ),
            )

        if self.args.feature.use('atom-graph'):
            _local_dim = 128
            src_graphs.append(
                dict(
                    node_dim=33,
                    use_pos=True,
                    tgt_dim=_local_dim,
                    use_concat=True,
                    edge_dim=1,
                ),
            )

        if self.args.feature.use('surface-graph'):
            _local_dim = 128
            src_graphs.append(
                dict(
                    node_dim=0,
                    use_nor=True,
                    use_pos=True,
                    use_concat=True,
                    tgt_dim=_local_dim,
                ),
            )

        if len(src_graphs) > 0:
            if num_layers > 1:
                assert num_layers - 1 == len(self.args.rates)
                assert num_layers - 1 == len(self.args.rs)
                ks = [ -1 ] * (num_layers - 1)
                sa_dims = [ _local_dim ] * (num_layers - 1)
                fp_dims = [ _local_dim ] * (num_layers - 1)
            else:
                ks = None
                sa_dims = None
                fp_dims = None

            output_dim = 128
            encoder_1d = AtomGCN(
                embed_dim, output_dim, tuple(src_graphs), num_sa=num_layers-1, 
                use_fp=True, rates=self.args.rates, rs=self.args.rs, ks=ks,
                sa_dims=sa_dims, fp_dims=fp_dims,
            )

        return encoder_1d, output_dim

    def forward(self, data):
        x = None
        if self.esm_embed is not None:
            try:
                with torch.no_grad():
                    self.esm_embed.eval()
                    msa = data['msa']
                    x = self.esm_embed(msa)['row_attentions']
                
                if self.prepend_bos:
                    x = x[..., 1:, 1:]

                B, L, N, K, K = x.size()
                x = x.view(B, L*N, K, K)

                reclen = int(data['reclen'])
                liglen = int(data['liglen'])

                _op = self.args.row_attn_op
    
                if _op == 'lower_tri':
                    x = x[:, :, :reclen, reclen:]
                elif _op == 'upper_tri':
                    x = x[:, :, reclen:, :receln].transpose(-2,-1)
                elif _op == 'sym':
                    x = (
                        x[:, :, :reclen, reclen:] + 
                        x[:, :, reclen:, :reclen].transpose(-2,-1)
                    )
                elif _op == 'apc': # sym, then apc
                    x = apc(
                        x + x.transpose(-2,-1)
                    )[:, :, :reclen, reclen:]
                else:
                    raise ValueError()

                if x.size(-1) != int(liglen):
                    print(x.size(), msa.size(), reclen, liglen)
                    raise RuntimeError('shape mismatch in concated msa')

            except RuntimeError as e:
                raise e

            if self._gen_esm:
                assert B == 1
                x = x.squeeze(0)
                return x

        if 'pickled-esm' in self.args.feature:
            assert x is None
            x = data['esm']
 
        if 'ccm' in self.args.feature:
            y = data['ccm']
            if x is not None:
                x = torch.cat((x, y), dim=1)
            else:
                x = y

        if self.encoder_1d is not None:
            y_rec, y_lig = self.encoder_1d_forward(data)
            y_rec = y_rec[:, :, data['recidx'][0]]
            y_lig = y_lig[:, :, data['ligidx'][0]]
            reclen, liglen = y_rec.size(-1), y_lig.size(-1)
            y = torch.cat(
                (
                    y_rec.unsqueeze(3).expand(-1, -1, -1, liglen),
                    y_lig.unsqueeze(2).expand(-1, -1, reclen, -1),
                ),
                dim=1,
            )
            if x is not None:
                x = torch.cat((x, y), dim=1)
            else:
                x = y
        
        g = self.resnet(x)
        logits = self.fc(g)
        lprobs = F.log_softmax(logits, dim=1).permute(0,2,3,1)

        return lprobs

    def encoder_1d_forward(self, data):
        if 'ca-embed' in self.args.feature:
            y_rec = self.encoder_1d(data['rec_embed'])
            y_lig = self.encoder_1d(data['lig_embed'])
            assert y_rec.size(0) == 1 and y_lig.size(0) == 1

        rec_graphs, lig_graphs = [], []
        if self.args.feature.use('distance-ca-graph', 'coordinate-ca-graph'):
            rec_cag = data['rec_cag']
            lig_cag = data['lig_cag']
            rec_graphs.append(rec_cag)
            lig_graphs.append(lig_cag)
            if self.args.feature.use('atom-graph'):
                rec_graphs.append(data['rec_atg'])
                lig_graphs.append(data['lig_atg'])

            if self.args.feature.use('surface-graph'):
                rec_graphs.append(data['rec_sug'])
                lig_graphs.append(data['lig_sug'])

            y_rec = self.encoder_1d(
                rec_cag.x, rec_cag.pos, rec_cag.lrf, rec_graphs,
            )
            y_rec = y_rec.unsqueeze(0).transpose(1,2)
            y_lig = self.encoder_1d(
                lig_cag.x, lig_cag.pos, lig_cag.lrf, lig_graphs,
            )
            y_lig = y_lig.unsqueeze(0).transpose(1,2)

        return y_rec, y_lig

if __name__ == '__main__':
    import argparse
    import pickle
    from glinter.dataset.dimer_dataset import DimerDataset
    from glinter.dataset.collater import Collater
    from glinter.models.checkpoint_utils import load_state
    # torch.manual_seed(123)
    parser = argparse.ArgumentParser()

    DimerDataset.add_args(parser)
    MSAModel.add_args(parser)
    parser.add_argument('--ckpt-path', type=Path)
    parser.add_argument('--generate-esm-attention', action='store_true')
    args, _ = parser.parse_known_args()

    if args.generate_esm_attention:
        args = parser.parse_args()
        esm_embed, alphabet = load_esm_model(args.ckpt_path)
        model = MSAModel(
            args, esm_embed=esm_embed, prepend_bos=alphabet.prepend_bos, gen_esm=True
        )
        dataset = DimerDataset(args, esm_alphabet=alphabet)
    else:
        if args.ckpt_path is not None:
            assert args.ckpt_path.exists(), f"{args.ckpt_path} does not exist"
            state = load_state(args.ckpt_path)
            model = MSAModel(args)
            model.load_state_dict(state, strict=True)
        dataset = DimerDataset(args,)
    collater = Collater()
    model.eval()
    for i in range(len(dataset)):
        name = dataset.dimers[i][0]
        if args.generate_esm_attention:
            batch = collater([ dataset.get_msa(i) ])
            fname = args.dimer_root.parent.joinpath(name + '.esm.npz')
            with torch.no_grad():
                output = model(batch['data']).cpu().numpy().astype(np.float16)
            np.savez_compressed(fname, esm=output)
        else:
            batch = collater([ dataset[i] ])
            output = {'model':{}}
            with torch.no_grad():
                output['model']['output'] = model(batch['data']).cpu()
            output['model']['recidx'] = batch['data']['recidx'].cpu()
            output['model']['ligidx'] = batch['data']['ligidx'].cpu()
            fname = args.dimer_root.parent.joinpath(name + '.out.pkl')
            with open(fname, 'wb') as handle:
                pickle.dump(output, handle)
