import re
import torch

__all__ = [
    'cigar_to_index', 'show_aln', 'get_alniden',
]

def show_aln(alnidx, qseq, tseq):
    i1, i2 = alnidx
    _qseq, _tseq = [], []
    for i, j in zip(i1, i2):
        _qseq.append(qseq[i]) 
        _tseq.append(tseq[j])
    print(''.join(_qseq))
    print(''.join(_tseq))

def get_alniden(alnidx, qseq, tseq):
    i1, i2 = alnidx
    n = 0
    for i, j in zip(i1, i2):
        if qseq[i] == tseq[j]:
            n += 1
    return n / len(i1)

def cigar_to_index(cigar, qbeg=0, tbeg=0):
    """
    Args:
        cigar (str) : MID 
            M: match,
            I: insertion (i.e. char in query, '-' in target),
            D: deletion (i.e. char in target, '-' in query).

    Returns:
        index (torch.LongTensor, 2 x L) : L is the length of the alignment.
            query, target = index
    """
    cigar = re.split('([MID])', cigar)
    assert cigar[-1] == ''
    cigar = cigar[:-1]
    q, t = [], []
    qcur, tcur = qbeg, tbeg
    for i in range(0, len(cigar), 2):
        k = int(cigar[i])
        w = cigar[i+1]
        if w == 'M':
            _len = torch.arange(k, dtype=torch.long)
            _q = qcur + _len
            _t = tcur + _len
            qcur = _q[-1] + 1
            tcur = _t[-1] + 1
            q.append(_q)
            t.append(_t)
        elif w == 'I':
            qcur = qcur + k
        elif w == 'D':
            tcur = tcur + k
    return torch.cat(
        (torch.cat(q, dim=0).view(1, -1), torch.cat(t, dim=0).view(1, -1)),
        dim=0,
    )
