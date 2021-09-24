import torch

def load_tt(ttpath):
    _AA = []
    with open(ttpath, 'rt') as h:
        for l in h:
            l = l.strip()
            if l and not l.startswith('#'):
                i, j = l.split('\t')
                _AA.append((int(i), int(j)))

    AA = torch.zeros(max(_[0]+1 for _ in _AA), dtype=torch.long)
    for i, j in _AA:
        AA[i] = j
    return AA

def load_msa(
    dten, max_row, recidx=None, ligidx=None, esm_alphabet=None, precut=True,
):
    """
    Args:
        dten (dict): dimer-tensor
        max_row (int): the max number of rows
        esm_alphabet (Alphabet, optional): the esm model's alphabet used to 
            transform input tokens
    Returns:
    """
    _msa = dten['msa']
    if precut:
        if not dten['concated']:
            msa = torch.cat((_msa[:, recidx], _msa[:, ligidx]), dim=-1)
        else:
            ligbeg = dten['reclen']
            _idx = torch.cat((recidx, ligidx + ligbeg), dim=0)
            msa = _msa[:, _idx]
    else:
        if not dten['concated']:
            msa = torch.cat((_msa, _msa), dim=-1)
        
    # TODO: sampling
    if max_row > 0:
        msa = msa[:max_row]

    # prepend cls_idx and append eos_idx
    if esm_alphabet is not None:
        if esm_alphabet.prepend_bos:
            msa = torch.cat(
                (
                    torch.LongTensor([esm_alphabet.cls_idx]).expand(
                        msa.size(0), 1
                    ),
                    msa,
                ),
                dim=-1,
            )
        if esm_alphabet.append_eos:
            msa = torch.cat(
                (
                    msa,
                    torch.LongTensor([esm_alphabet.eos_idx]).expand(
                        msa.size(0), 1
                    ),
                ),
                dim=-1,
            )
    return msa 
