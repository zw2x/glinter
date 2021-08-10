import string
from copy import copy

import numpy as np
import torch

__all__ = [
    'ATOMS', 'ATOM_ONES', 'encode_atoms',
    'SS8', 'SS8_ONES', 'encode_ss8',
    'AA1', 'AA1_ONES', 'encode_aa1', 'three_to_one', 'one_to_three',
    'seq_encode',
]

"""
Basic utils
""" 
def _build_dict(it):
    _dict = dict( (w,i) for i, w in enumerate(it) )
    _ones = torch.eye(len(_dict), dtype=torch.float32)
    return _dict, _ones

def _encode(sent, _dict, ones=None, oov=None):
    v = torch.LongTensor(
        [ 
            _dict[w]
            if w in _dict or oov is None else oov(w, _dict) for w in sent
        ],
    )
    if ones is not None:
        v = ones[v]
    return v

"""
Atoms
"""
ATOMS, ATOM_ONES = _build_dict([
    'CA', 'N' , 'C' , 'CB', 'O', 'NX', 'CX', 'OX', 'SX', 'HX', 'X',
])

# ATOMS = dict( (w, i) for i, w in enumerate([
#     'N',
#     'C',
#     'O',
#     'S',
#     'X',
# ]))

# out of vocabulary
def _atom_oov(w, _dict):
    return _dict.get(w[0]+'X', ATOMS['X'])

def encode_atoms(atoms, onehot=False,):
    if onehot:
        ones = ATOM_ONES
    else:
        ones = None

    return _encode(atoms, ATOMS, ones=ones, oov=_atom_oov)

"""
Secondary structure symbols
"""
SS8, SS8_ONES = _build_dict('HBEGITS-')

def encode_ss8(ss, onehot=False,):
    if onehot:
        ones = SS8_ONES
    else:
        ones = None

    return _encode(ss, SS8, ones=ones,)

"""
Amino acids
"""
aa3 = [
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR", "X",
]

aa1 = "ACDEFGHIKLMNPQRSTVWYX"

AA1, AA1_ONES = _build_dict(aa1)

def _aa1_oov(w, _dict):
    return AA1['X']

def encode_aa1(aa, onehot=False,):
    if onehot:
        ones = AA1_ONES
    else:
        ones = None

    return _encode(aa, AA1, ones=ones, oov=_aa1_oov)

d3_to_d1 = {}
d1_to_d3 = {}

for a1, a3 in zip(aa1, aa3):
    d3_to_d1[a3] = a1
    d1_to_d3[a1] = a3

def three_to_one(s):
    return d3_to_d1.get(s, 'X')

def one_to_three(s):
    return d1_to_d3.get(s, 'X')


"""
Sequence encode ( from ascii.string.upper_case to number )
"""
WORDS = string.ascii_uppercase + '-'
NUM = bytes.maketrans(WORDS.encode('latin1'), bytearray(range(len(WORDS))))

def seq_encode(s):
    s = s.encode('latin1').translate(NUM)
    s = torch.LongTensor(copy(np.frombuffer(s, dtype=np.uint8)))
    return s
