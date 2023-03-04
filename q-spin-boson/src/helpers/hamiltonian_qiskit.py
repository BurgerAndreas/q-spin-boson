#
from logging import warning
import re as regex
import sympy as sym
import numpy as np
import pickle

from settings.types import Enc, Model
from settings.parameters import Paras
from settings.conventions import pz_convention
from helpers.binary import get_seq
import hamiltonian_matrix as hm

# Sermionic operators
PXS = sym.IndexedBase('sigma^x', commutative=False)
PYS = sym.IndexedBase('sigma^y', commutative=False)
# PZS = 1, 0, 0, -1 | PZS = -1, 0, 0, 1
PZS = sym.IndexedBase('sigma^z', commutative=False)

CMPLX = 1j  # or sym.I doesn't make a difference



def bin_to_pauli(out, inc, unary=False, start=0):
    """
    o[out][inc] = <out|o|inc> ~ |out><inc|
    input: two strings of 0s and 1s
    start = start qubit
    """
    p_string = 1
    for c in range(len(out)):
        if out[c] == '1':
            if inc[c] == '1':  # 11
                p_string *= sym.Rational(1, 2) * (1 - PZS[start + c])
            elif inc[c] == '0':  # 10
                # sym.I or 1j
                p_string *= sym.Rational(1, 2) * (PXS[start + c] - (1j * PYS[start + c]))
            else:
                print('invalid inc')
        elif out[c] == '0':
            if inc[c] == '1':  # 01 # watch interaction with python imaginary j
                p_string *= sym.Rational(1, 2) * (PXS[start + c] + (1j * PYS[start + c]))
            elif inc[c] == '0':  # 00
                if unary:
                    continue  # only consider terms 11, 01, 10. Ignore 00
                else:
                    p_string *= sym.Rational(1, 2) * (1 + PZS[start + c])
            else:
                print('invalid inc')
        else:
            print('invalid out')
    return p_string


def matrix_to_pauli(matrix, enc, en_seq, start=0, aslatex=False):
    """Turn a matrix into a pauli string"""
    matrix = np.asarray(matrix)
    if len(en_seq) != len(matrix[0]):
        print('enc length', len(en_seq), 'but matrix shape', np.shape(matrix))
    matrix_pauli = 0
    for row_num in range(len(matrix[0])):
        for col_num in range(len(matrix[1])):
            if np.abs(matrix[row_num, col_num]) == 0:
                continue  # ignore
            out_code = en_seq[col_num]
            inc_code = en_seq[row_num]
            elem_pauli = (
                bin_to_pauli(
                    out=out_code, inc=inc_code, unary=True, start=start
                )
                if enc in [Enc.SPLITUNARY, Enc.FULLUNARY]
                else bin_to_pauli(out=out_code, inc=inc_code, start=start)
            )
            matrix_pauli += elem_pauli * matrix[row_num, col_num]
    return latexify(matrix_pauli) if aslatex else matrix_pauli


def latexify(matrix_pauli):
    q_ham = sym.simplify(matrix_pauli)
    q_ham = sym.expand(q_ham)
    q_ham = sym.N(q_ham, 3)
    return sym.latex(q_ham)


def pauli_to_qiskit(term, qubit_num, aslatex=False):
    """Clean-up a pauli string into a qiskit operatorflow format.
    coeff_number * I^I^X^I^Z^...^I s.t. length = num qubits
    """
    if aslatex:
        return latexify(term)
    q_ham = sym.simplify(term)
    q_ham = sym.expand(q_ham)
    # print('before', q_ham)
    # replace variables by numbers
    # replace pretty coeffs by number coeffs
    q_hami = q_ham
    q_hami = sym.N(q_hami, 8)
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
        if term in ['+', '-']:
            reordered_q_string = f'{reordered_q_string} {term} '
            continue
        # reorder variables
        variables = term.split('*')
        # print('variables', variables)
        reordered_term = ''
        # ignore term if prefactor is e^-...
        e_pos = variables[0].find('e')
        if (
            e_pos != -1
            and variables[0][e_pos + 1] == '-'
            and variables[0][e_pos + 2 :]
            in ['11', '12', '13', '14', '15', '16', '17', '18']
        ):
            if reordered_q_string != '':  # delete previous + or -
                reordered_q_string = reordered_q_string[:-3]
            continue  # ignore this term
        # get numerical factor
        for _v in variables:
            if '.' in _v:
                reordered_term += _v
                break
        # go through position wise
        for qubit in range(qubit_num):
            first_at_qubit = True
            for _v in variables:
                if f'[{str(qubit)}]' in _v:
                    if first_at_qubit:
                        first_at_qubit = False
                        # if not at the beginning of term
                        reordered_term = (
                            f'{reordered_term}*{_v}'
                            if (len(reordered_term) > 0)
                            and (reordered_term[-1] != ' ')
                            else reordered_term + _v)
                    else:
                        # operator composition
                        reordered_term = f'{reordered_term}@{_v}'
            if first_at_qubit:
                if (len(reordered_term) > 0) and (reordered_term[-1] != ' '):
                    reordered_term = f'{reordered_term}*I[{str(qubit)}]'
                else:
                    reordered_term = f'{reordered_term}I[{str(qubit)}]'
        # add to q_string
        reordered_q_string = reordered_q_string + reordered_term
    # print('reordered', reordered_q_string)
    q_string = reordered_q_string
    for p in range(qubit_num):
        q_string = q_string.replace(f'I[{str(p)}]', 'I^')
        q_string = q_string.replace(f'sigma^x[{str(p)}]', 'X^')
        q_string = q_string.replace(f'sigma^y[{str(p)}]', 'Y^')
        q_string = q_string.replace(f'sigma^z[{str(p)}]', 'Z^')
    q_string = f' {q_string} '
    q_string = q_string.replace('^ ', ') ')
    q_string = q_string.replace('^*', '^')
    q_string = q_string.replace('^@', '@')
    q_string = q_string.replace('*', '*(')
    for c in ['I', 'X', 'Y', 'Z']:
        q_string = q_string.replace(f' {c}', f' ({c}')
    return q_string


def transpose_ham_string(ham_string):
    """Transpose a hamiltonina in qiskit opflow format.
    input: hamiltonian PauliOpSum string
    return hamiltonian transpose PauliOpSum string
    H -> H^T = Y -> -Y
    """
    sgn = '+'
    num = ''
    pauli = ''
    new_ham = ''
    for op in ham_string.split():
        if op in ['-', '+']:
            sgn = op
        else:
            num = op.split('*')[0]
            pauli_string = op.split('*')[1][1:-1]
            for pauli in pauli_string.split('^'):
                if pauli == 'Y':
                    sgn = '-' if sgn == '+' else '+'
            new_ham += f' {sgn} {num}*({pauli_string})'
    new_ham = new_ham[1:]  # remove first space
    # print(ham_string)
    # print(new_ham)
    return new_ham


def hamiltonian_as_paulis(model=Model.SB1S, n_bos=4, paras=Paras.SB1S, 
                          enc=Enc.SB, aslatex=False):
    """Get hamiltonian as pauli strings."""
    # get encoding
    en_seq = get_seq(enc, numbers=n_bos)
    n_qubits_boson = len(en_seq[0])
    """ Qubits """
    if model.name[:4] == 'SB1S':  # S - HO
        s1, s2 = 0, 0
        first_qubit_boson = 1
        n_qubits = n_qubits_boson + 1
    elif model.name[:4] == 'SB2S':  # S - HO - S
        s1, s2 = n_qubits_boson + 1, 0
        first_qubit_boson = 1
        n_qubits = n_qubits_boson + 2
    elif model == Model.JC3S:  # S - S - S - HO
        s1, s2, s3 = 0, 1, 2
        first_qubit_boson = 3
        n_qubits = n_qubits_boson + 3
    elif model == Model.JC2S:  # S - S - HO
        s1, s2 = n_qubits_boson + 1, 0
        first_qubit_boson = 1
        n_qubits = n_qubits_boson + 2
    # bosonic operators to Pauli
    a_matrix = hm.build_a(n_bos=n_bos)
    a_pauli = matrix_to_pauli(a_matrix, enc, en_seq, first_qubit_boson)
    ad_matrix = hm.build_ad(n_bos=n_bos)
    ad_pauli = matrix_to_pauli(ad_matrix, enc, en_seq, first_qubit_boson)
    n_matrix = hm.build_ada(n_bos=n_bos)
    n_pauli = matrix_to_pauli(n_matrix, enc, en_seq, first_qubit_boson)
    q_matrix = a_matrix + ad_matrix
    q_pauli = matrix_to_pauli(q_matrix, enc, en_seq, first_qubit_boson)
    # hamiltonian
    h_pauli = 0
    if model == Model.SB1SPZ: # H-S
        # S1
        h_pauli += PZS[s1] * 0.5 * pz_convention
        h_pauli += PXS[s1] * 0.5 * paras.value[0]  # epsilon
        # HO
        h_pauli += n_pauli * paras.value[1]  # omega
        # S1-HO
        h_pauli += q_pauli * PZS[s1] * pz_convention * paras.value[2]  # lambda
    elif model == Model.SB1SJC: # H-S
        # S1
        h_pauli += PZS[s1] * 0.5 * pz_convention
        h_pauli += PXS[s1] * 0.5 * paras.value[0]  # epsilon
        # HO
        h_pauli += n_pauli * paras.value[1]  # omega
        # S1-HO
        h_pauli += ad_pauli * 0.5*(PXS[s1] - (CMPLX*PYS[s1])) * paras.value[2]  # lambda
        h_pauli += a_pauli * 0.5*(PXS[s1] + (CMPLX*PYS[s1])) * paras.value[2]  # lambda
    elif model == Model.SB1S: # H-S
        # S1
        h_pauli += PZS[s1] * 0.5 * pz_convention
        h_pauli += PXS[s1] * 0.5 * paras.value[0]  # epsilon
        # HO
        h_pauli += n_pauli * paras.value[1]  # omega
        # S1-HO
        h_pauli += q_pauli * PXS[s1] * paras.value[2]  # lambda
    elif model == Model.SB2S: # S-H-S
        # S1, S2
        h_pauli += PZS[s1] * 0.5 * pz_convention
        h_pauli += PZS[s2] * 0.5 * pz_convention
        h_pauli += PXS[s1] * 0.5 * paras.value[0]  # epsilon
        h_pauli += PXS[s2] * 0.5 * paras.value[0]  # epsilon
        # HO
        h_pauli += n_pauli * paras.value[1]  # omega
        # S1-HO, S2-HO
        h_pauli += q_pauli * PXS[s1] * paras.value[2]  # lambda
        h_pauli += q_pauli * PXS[s2] * paras.value[2]  # lambda
    elif model == Model.JC3S:
        # Three spins, no hopping, full hilbert space, post selection
        # S1, S2, S3
        s1, s2, s3 = 0, 1, 2
        h_pauli += PZS[s1] * 0.5 * pz_convention
        h_pauli += PZS[s2] * 0.5 * pz_convention
        h_pauli += PZS[s3] * 0.5 * pz_convention
        # HO
        h_pauli += n_pauli * paras.value[1]  # omega
        # S1-HO, S2-HO, S3-HO
        h_pauli += ad_pauli * 0.5*(PXS[s1] + (CMPLX*PYS[s1])) * paras.value[2]  # lambda
        h_pauli += a_pauli * 0.5*(PXS[s1] - (CMPLX*PYS[s1])) * paras.value[2]  # lambda
        # S2
        h_pauli += ad_pauli * 0.5*(PXS[s2] + (CMPLX*PYS[s2])) * paras.value[2]  # lambda
        h_pauli += a_pauli * 0.5*(PXS[s2] - (CMPLX*PYS[s2])) * paras.value[2]  # lambda
        # S3
        h_pauli += ad_pauli * 0.5*(PXS[s3] + (CMPLX*PYS[s3])) * paras.value[2]  # lambda
        h_pauli += a_pauli * 0.5*(PXS[s3] - (CMPLX*PYS[s3])) * paras.value[2]  # lambda
    elif model == Model.JC2S: # S-H-S
        # S1, S2
        h_pauli += PZS[s1] * 0.5 * pz_convention
        h_pauli += PZS[s2] * 0.5 * pz_convention
        # HO
        h_pauli += n_pauli * paras.value[1]  # omega
        # S1-HO
        h_pauli += a_pauli * 0.5 * (PXS[s1] + (CMPLX*PYS[s1])) * paras.value[2]  # lambda
        h_pauli += ad_pauli * 0.5 * (PXS[s1] - (CMPLX*PYS[s1])) * paras.value[2]  # lambda
        # S2-HO
        h_pauli += a_pauli * 0.5 * (PXS[s2] + (CMPLX*PYS[s2])) * paras.value[2]  # lambda
        h_pauli += ad_pauli * 0.5 * (PXS[s2] - (CMPLX*PYS[s2])) * paras.value[2]  # lambda
    if aslatex:
        return latexify(h_pauli, enc)
    return pauli_to_qiskit(h_pauli, n_qubits)
