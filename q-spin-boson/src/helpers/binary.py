import math
import numpy as np
from itertools import product

from settings.types import Enc

def to_binary(i, length):
    b = bin(i)[2:]  # without 0b...
    return str(0) * (length - len(b)) + b


def state_to_binary(state):
    _bin = to_binary(np.nonzero(state)[1], 5)
    return _bin[::-1]


def sb_sequence(numbers=0, length=None, rev=False):
    if length:
        numbers = 2 ** length
    else:
        length = math.ceil(math.log2(numbers))
    sb_seq = []
    for i in range(numbers):
        b = to_binary(i, length)
        if rev:
            sb_seq.append(b[::-1])
        else:
            sb_seq.append(b)
    return sb_seq


def unary_sequence(numbers=3, length=None):
    if length:
        numbers = length
    un_seq = []
    for i in range(numbers):
        zero_before = str(0) * i
        zero_after = str(0) * (numbers - i - 1)
        this_unary = zero_before + str(1) + zero_after
        un_seq.append(this_unary)
    return un_seq


def gray_sequence(numbers=3, length=None, rev=False):
    if not length:
        length = math.ceil(math.log2(numbers))
    gray_seq = []
    n = int(length)
    for i in range(1 << n):
        gray = i ^ (i >> 1)
        gray_string = "{0:0{1}b}".format(gray, n)
        if rev:
            gray_seq.append(gray_string[::-1])
        else:
            gray_seq.append(gray_string)
    return gray_seq


def get_seq(encoding: Enc, numbers=None, length=None, rev=False):
    if encoding == Enc.BINARY:
        seq = sb_sequence(numbers=numbers, length=length, rev=rev)
    elif encoding in [Enc.SPLITUNARY, Enc.FULLUNARY]:
        seq = unary_sequence(numbers=numbers, length=length)
    elif encoding == Enc.GRAY:
        seq = gray_sequence(numbers=numbers, length=length, rev=rev)
    return seq


def fillin_counts(cnts):
    """Fill in missing counts (possible combinations of 0s, 1s) with =0."""
    qubits = len(list(cnts.keys())[0])
    # unique combinations of 0s and 1s
    uc = [''.join(str(i) for i in c) for c in product([0, 1], repeat=qubits)]
    for c in uc:
        if c not in cnts: cnts[c] = 0
    return cnts