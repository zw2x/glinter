from pathlib import Path

__all__ = [
    'get_chainid',
    'read_ents',
    'read_seqmap',
    'read_models',
]

def get_chainid(chain):
    chainid = chain.get_id()
    if chainid.strip() == '':
        chainid = '*'
    return chainid

def get_chaincode(name):
    if len(name) > 5:
        code, mid, cid = name.split('_')
        ccode = code + cid
    elif len(name) == 5:
        ccode = name
    else:
        raise RuntimeError(f'cannot parse {name}')
    return ccode
 
def read_ents(path, split_name=True, return_both=False, key=0):
    ents = set()
    if return_both:
        split_name = True
        models = set()
    with open(path, 'rt') as h:
        for l in h:
            l = l.strip()
            if l:
                fields = l.split('\t')
                mname = fields[key]
                if split_name:
                    rec, lig = mname.split(':')
                    ents.add(rec)
                    ents.add(lig)
                    if return_both:
                        models.add(mname)
                else:
                    ents.add(mname)

    if return_both:
        return models, ents
    else:
        return ents

def read_seqmap(path, refseqs=None, only_names=False):
    """
    Args:
        refseqs (dict, optional) : used to get the mapped sequences,
            None if sequences are mapped later stages
    """
    _map = dict()
    with open(path, 'rt') as h:
        lines = h.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        fields = line.split('\t') 
        k = fields[0]
        ref = fields[1]

        # assuming queries are unique
        assert k not in _map

        if only_names:
            _map[k] = ref
            continue

        qbeg = int(fields[-3])
        tbeg = int(fields[-2])
        cigar = fields[-1]

        refseq = refseqs[ref] if refseqs is not None else None

        _map[k] = dict(
            ref=ref,
            qbeg=qbeg,
            tbeg=tbeg,
            cigar=cigar,
            refseq=refseq
        )
        
    return _map

def read_models(path):
    models = dict()
    with open(path, 'rt') as h:
        lines = h.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        pdbpair, seqpair = fields[:2]
        assert pdbpair not in models
        models[pdbpair] = seqpair

    return models
