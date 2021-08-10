from collections import OrderedDict

__all__ = [ 'read_seqs' ]

def read_seqs(p, topk=-1, pick_longer=True, ignore_header=False):
    seqs = OrderedDict()
    header = None
    _seq = []

    def _add_seq():
        nonlocal header, _seq, seqs
        if header is None or len(_seq) == 0:
            return
        if topk > 0 and len(seqs) == topk:
            return
        _seq = ''.join(_seq)
        if header not in seqs:
            seqs[header] = _seq
        else:
            if _seq != seqs[header]:
                print(header)
                if pick_longer and len(_seq) > len(seqs[header]):
                    seqs[header] = _seq
        header = None
        _seq = []

    with open(p, 'rt') as ph:
        for line in ph:
            l = line.strip()
            if l.startswith('>'):
                if topk > 0 and len(seqs) == topk:
                    break
                _add_seq()
                if len(seqs) == 0 or not ignore_header:
                    # preserve the header of the query (the first sequence),
                    # even if ignore_headers is True (in order to use get_len)
                    header = l[1:]
                else:
                    header = len(seqs)
            elif header is not None:
                _seq.append(l)
        else:
            _add_seq()

    return seqs
