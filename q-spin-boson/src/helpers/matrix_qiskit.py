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

#
import re as regex
from sympy import *
import numpy as np
import pickle

#
import binaryCodes as bc


class MatrixToQiskit:
    def __init__(self, matrix=None, spins=1, harm_oscis=1, excitations=4,
                 encoding='fullgray'):
        # settings
        self.precision_prefactor = 5
        # set variables
        self.ss = spins * 2  # possible spin states = size of spin basis
        self.harm_oscis = harm_oscis  # number bosons in oscillator (0, ..., harm_oscis-1)
        self.excitations = excitations  # number excitations / bosons in oscillator (0, ..., excitations-1)
        self.encoding = encoding  # 'splitunary', 'fullunary' 'sb', 'gray', 'fullsb', 'fullgray'
        # calculate dimensionality of the problem
        self.dimh = matrix.shape[0]
        # set encoding
        self.dim_enc = self.dimh
        if self.encoding == 'fullsb':
            self.en_seq = bc.sb_sequence(numbers=self.dim_enc)
        elif self.encoding == 'fullunary':
            self.en_seq = bc.unary_sequence(numbers=self.dim_enc)
        elif self.encoding == 'fullgray':
            self.en_seq = bc.gray_sequence(numbers=self.dim_enc)
        else:
            print('invalid encoding flag')
        self.qubit_num = len(self.en_seq[0])
        # symbolic variables
        self.px = IndexedBase('sigma^x', commutative=False)
        self.py = IndexedBase('sigma^y', commutative=False)
        self.pz = IndexedBase('sigma^z', commutative=False)

    # o[out][inc] = <out|o|inc> ~ |out><inc|
    def bin_to_pauli(self, out, inc, unary=False):  # input: two strings
        p_string = 1
        for c in range(len(out)):
            if out[c] == '1':
                if inc[c] == '1':  # 11
                    p_string *= Rational(1, 2) * (1 - self.pz[c])
                elif inc[c] == '0':  # 10
                    p_string *= Rational(1, 2) * (self.px[c] - (I * self.py[c]))
                else:
                    print('invalid inc')
            elif out[c] == '0':
                if inc[c] == '1':  # 01 # watch interaction with python imaginary j
                    p_string *= Rational(1, 2) * (self.px[c] + (I * self.py[c]))
                elif inc[c] == '0':  # 00
                    if unary:
                        continue  # only consider terms 11, 01, 10. Ignore 00
                    p_string *= Rational(1, 2) * (1 + self.pz[c])
                else:
                    print('invalid inc')
            else:
                print('invalid out')
        return p_string

    def matrix_to_pauli(self, matrix):
        matrix_pauli = 0
        for row_num in range(self.dimh):
            for col_num in range(self.dimh):
                if np.abs(matrix[row_num, col_num]) == 0:
                    continue  # ignore
                else:
                    out_code = self.en_seq[col_num]
                    inc_code = self.en_seq[row_num]
                    if self.encoding in ['fullunary']:
                        elem_pauli = self.bin_to_pauli(out=out_code, inc=inc_code, unary=True)
                    else:
                        elem_pauli = self.bin_to_pauli(out=out_code, inc=inc_code)
                    # todo interaction with python imaginary j ?
                    matrix_pauli += elem_pauli * matrix[row_num, col_num]
        return matrix_pauli

    # print in qiskit operatorflow format
    # coeff_number * I^I^X^I^Z^...^I        s.t. length = num qubits
    def pauli_to_qiskit(self, term):
        q_ham = simplify(term)
        q_ham = expand(q_ham)
        print(latex(q_ham))
        # print('before', q_ham)
        # replace variables by numbers
        # replace pretty coeffs by number coeffs
        q_hami = q_ham
        q_hami = N(q_hami, self.precision_prefactor)
        # print('q_ham with numbers', q_ham)
        # make to string
        q_string = str(q_hami)
        # reorder to be in sequence
        # only works under the assumption, that the right order is qubit[0] to qubit[max]
        # which is alright since different sites commute anyway ?
        # print('before reordering', q_string)
        terms = q_string.split(' ')
        # print('terms', terms)
        reordered_q_string = ''
        for term in terms:
            if term == '+' or term == '-':
                reordered_q_string = reordered_q_string + ' ' + term + ' '
                continue
            # reorder variables
            vars = term.split('*')
            # print('vars', vars)
            reordered_term = ''
            # ignore term if prefactor is e^-...
            e_pos = vars[0].find('e')
            if e_pos != -1:  # e found
                if vars[0][e_pos + 1] == '-':
                    if vars[0][e_pos + 2:] in ['11', '12', '13', '14', '15', '16', '17', '18']:
                        if reordered_q_string != '':  # delete previous + or -
                            reordered_q_string = reordered_q_string[:-3]
                        continue  # ignore this term
            # get numerical factor
            for var in vars:
                if '.' in var:
                    reordered_term = reordered_term + var
                    break
            # go through position wise
            for qubit in range(self.qubit_num):
                first_at_qubit = True
                for var in vars:
                    if ('[' + str(qubit) + ']') in var:
                        if not first_at_qubit:
                            # TODO is this how you operator composition ?
                            reordered_term = reordered_term + '@' + var
                        else:
                            first_at_qubit = False
                            # if not at the beginning of term
                            if (len(reordered_term) > 0) and (reordered_term[-1] != ' '):
                                reordered_term = reordered_term + '*' + var
                            else:
                                reordered_term = reordered_term + var
                # if none found, fit in identity
                if first_at_qubit:
                    # if not at the beginning of term
                    if (len(reordered_term) > 0) and (reordered_term[-1] != ' '):
                        reordered_term = reordered_term + '*' + 'I[' + str(qubit) + ']'
                    else:
                        reordered_term = reordered_term + 'I[' + str(qubit) + ']'
            # add to q_string
            reordered_q_string = reordered_q_string + reordered_term
        # print('reordered', reordered_q_string)
        q_string = reordered_q_string
        for p in range(self.qubit_num):
            q_string = q_string.replace('I[' + str(p) + ']', 'I^')
            q_string = q_string.replace('sigma^x[' + str(p) + ']', 'X^')
            q_string = q_string.replace('sigma^y[' + str(p) + ']', 'Y^')
            q_string = q_string.replace('sigma^z[' + str(p) + ']', 'Z^')
        q_string = ' ' + q_string
        q_string = q_string + ' '
        q_string = q_string.replace('^ ', ') ')
        q_string = q_string.replace('^*', '^')
        q_string = q_string.replace('^@', '@')
        q_string = q_string.replace('*', '*(')
        for c in ['I', 'X', 'Y', 'Z']:
            q_string = q_string.replace(' ' + c, ' (' + c)
        # print('replaced middle\t', q_string)
        #
        return q_string

    def matrix_to_qiskit(self, m):
        m_as_pauli = self.matrix_to_pauli(m)
        m_as_qiskit = self.pauli_to_qiskit(m_as_pauli)
        return m_as_qiskit
