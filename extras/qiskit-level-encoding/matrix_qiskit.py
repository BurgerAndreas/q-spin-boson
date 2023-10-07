# Andreas Burger
# February 2022
#
# Input: Hamiltonian (Numpy matrix)
# Output: Hamiltonian as Pauli operators (Qiskit Opflow)
# For each non-zero element in Hamiltonian
# convert row and col number to binary
# and then into pauli strings.
# Use Sympy for symbolic calculations
# Convert Pauli Hamiltonian into qiskit-readable format

# library imports
from sympy import *
import numpy as np

# file imports
import binary_codes as bc


# o[out][inc] = <out|o|inc> ~ |out><inc|
def bin_to_pauli(out, inc, unary=False):  # input: two strings
    # symbolic variables
    px = IndexedBase('sigma^x', commutative=False)
    py = IndexedBase('sigma^y', commutative=False)
    pz = IndexedBase('sigma^z', commutative=False)
    p_string = 1
    for c in range(len(out)):
        if out[c] == '1':
            if inc[c] == '1':  # 11
                p_string *= Rational(1, 2) * (1 - pz[c])
            elif inc[c] == '0':  # 10
                p_string *= Rational(1, 2) * (px[c] - (I * py[c]))
            else:
                print('invalid inc')
        elif out[c] == '0':
            if inc[c] == '1':  # 01
                p_string *= Rational(1, 2) * (px[c] + (I * py[c]))
            elif inc[c] == '0':  # 00
                if unary:
                    continue  # only consider terms 11, 01, 10. Ignore 00
                p_string *= Rational(1, 2) * (1 + pz[c])
            else:
                print('invalid inc')
        else:
            print('invalid out')
    return p_string


def matrix_to_pauli(matrix, dim_h, en_seq, encoding):
    matrix_pauli = 0
    for row_num in range(dim_h):
        for col_num in range(dim_h):
            if np.abs(matrix[row_num, col_num]) == 0:
                continue  # ignore
            else:
                out_code = en_seq[col_num]
                inc_code = en_seq[row_num]
                if encoding in ['unary', 'one_hot']:
                    elem_pauli = bin_to_pauli(out=out_code, inc=inc_code, unary=True)
                else:
                    elem_pauli = bin_to_pauli(out=out_code, inc=inc_code)
                matrix_pauli += elem_pauli * matrix[row_num, col_num]
    return matrix_pauli


# print in qiskit operatorflow format
# coeff_number * I^I^X^I^Z^...^I        s.t. length = num qubits
def pauli_to_qiskit(term, qubit_num):
    q_ham = expand(term)
    # replace pretty coeff by number coeffs
    q_ham = N(q_ham, 3)
    # make to string
    q_string = str(q_ham)
    #print('q_string', q_string)
    # reorder to be in sequence
    terms = q_string.split(' ')
    reordered_q_string = ''
    for term in terms:
        if term == '+' or term == '-':
            reordered_q_string = reordered_q_string + ' ' + term + ' '
            continue
        # reorder variables
        variables = term.split('*')
        reordered_term = ''
        # get numerical factor
        for variable in variables:
            if '.' in variable:
                reordered_term = reordered_term + variable
                break
        # go through position wise
        for qubit in range(qubit_num):
            first_at_qubit = True
            for variable in variables:
                if ('[' + str(qubit) + ']') in variable:
                    if not first_at_qubit:
                        # operator composition
                        reordered_term = reordered_term + '@' + variable
                    else:
                        first_at_qubit = False
                        # if not at the beginning of term
                        if (len(reordered_term) > 0) and (reordered_term[-1] != ' '):
                            reordered_term = reordered_term + '*' + variable
                        else:
                            reordered_term = reordered_term + variable
            # if none found, fit in identity
            if first_at_qubit:
                # if not at the beginning of term
                if (len(reordered_term) > 0) and (reordered_term[-1] != ' '):
                    reordered_term = reordered_term + '*' + 'I[' + str(qubit) + ']'
                else:
                    reordered_term = reordered_term + 'I[' + str(qubit) + ']'
        # add to q_string
        reordered_q_string = reordered_q_string + reordered_term
    # reordered
    q_string = reordered_q_string
    #print('reordered', q_string)
    # replace position indices
    for p in range(qubit_num):
        q_string = q_string.replace('I[' + str(p) + ']', 'I^')
        q_string = q_string.replace('sigma^x[' + str(p) + ']', 'X^')
        q_string = q_string.replace('sigma^y[' + str(p) + ']', 'Y^')
        q_string = q_string.replace('sigma^z[' + str(p) + ']', 'Z^')
    #print('replaced', q_string)
    # clean up
    q_string = ' ' + q_string
    q_string = q_string + ' '
    q_string = q_string.replace('^ ', ') ')
    q_string = q_string.replace('^*', '^')
    q_string = q_string.replace('^@', '@')
    q_string = q_string.replace('*', '*(')
    for c in ['I', 'X', 'Y', 'Z']:
        q_string = q_string.replace(' ' + c, ' (' + c)
        q_string = q_string.replace(' -' + c, ' -(' + c)
    #print('final', q_string)
    #
    return q_string


# encoding = 'sb', 'gray', 'unary'
def matrix_to_qiskit(matrix, encoding='gray'):
    if matrix.shape[0] != matrix.shape[1]:
        print('Error in call to matrix_to_qiskit(): matrix not square')
    # calculate dimensionality of the problem
    dim_h = matrix.shape[0]
    # set encoding
    en_seq, qubit_num = bc.encoding_sequence(encoding, numbers=dim_h)

    m_as_pauli = matrix_to_pauli(matrix, dim_h, en_seq, encoding)
    m_as_qiskit = pauli_to_qiskit(m_as_pauli, qubit_num)
    return m_as_qiskit, qubit_num


