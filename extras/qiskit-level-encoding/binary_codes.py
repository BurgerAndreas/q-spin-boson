#

import math


def sb_sequence(numbers=None, length=3):
    if numbers is not None:
        length = math.ceil(math.log2(numbers))
    else:
        numbers = 2 ** length
    sb_seq = []
    for i in range(numbers):
        b = bin(i)[2:]  # without 0b...
        l_bin = len(b)
        b = str(0) * (length - l_bin) + b
        sb_seq.append(b)
    return sb_seq, length


def unary_sequence(numbers=None, length=8):
    if numbers is not None:
        length = numbers
    else:
        numbers = length
    un_seq = []
    for i in range(numbers):
        zero_before = str(0) * i
        zero_after = str(0) * (numbers - i - 1)
        this_unary = zero_before + str(1) + zero_after
        un_seq.append(this_unary)
    return un_seq, length


def gray_sequence(numbers=None, length=3):
    if numbers is not None:
        length = math.ceil(math.log2(numbers))
    gray_seq = []
    n = int(length)
    for i in range(0, 1 << n):
        gray = i ^ (i >> 1)
        gray_seq.append("{0:0{1}b}".format(gray, n))
    return gray_seq, length


def encoding_sequence(encoding='gray', numbers=None, length=3):
    if encoding == 'gray':
        sequence, length = gray_sequence(numbers, length)
    elif encoding in ['sb', 'standard_binary']:
        sequence, length = sb_sequence(numbers, length)
    elif encoding in ['unary', 'one_hot']:
        sequence, length = unary_sequence(numbers, length)
    else:
        print('in call of encoding_sequence(): invalid encoding')
    return sequence, length
