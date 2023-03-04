import sympy as sympy
import numpy as np
import math as math
import functools as ft  # XX = ft.reduce(np.kron, [A, B, C, D, E])

from convention import pz, px, py, gs0, ex1, pm, pp, pppm, pmpp
from settings.types import Model, Env, H, Enc
from settings.parameters import Paras


def build_a(n_bos=4):
    a_matrix = np.zeros([n_bos, n_bos])
    for n in range(n_bos - 1):
        a_matrix[n, n + 1] = math.sqrt(n + 1)
    return a_matrix


# a dagger
def build_ad(n_bos=4):
    ad_matrix = np.zeros([n_bos, n_bos])
    for n in range(n_bos - 1):
        ad_matrix[n + 1, n] = math.sqrt(n + 1)
    return ad_matrix


# number operator
def build_ada(n_bos=4):
    ada_matrix = np.zeros([n_bos, n_bos])
    for n in range(n_bos):
        ada_matrix[n, n] = n
    return ada_matrix


def hamiltonian_as_matrix(model=Model.SB1S, n_bos=4, paras=Paras.SB1S):
    spin_id = np.eye(2)  # identity on spin space
    # build bosonic operators
    a = build_a(n_bos)
    ad = build_ad(n_bos)
    ada = build_ada(n_bos)
    q = a + ad
    boson_id = np.eye(n_bos)  # identity on boson space

    ''' Spin-Boson '''
    def sb1s():  # S-HO
        term1 = np.kron(pz, boson_id) * 0.5
        term2 = np.kron(px, boson_id) * 0.5 * paras.value[0]  # epsilon
        term3 = np.kron(spin_id, ada) * paras.value[paras][1]  # omega
        term4 = np.kron(px, q) * paras.value[paras][2]  # lambda
        return term1 + term2 + term3 + term4

    def sb1spz():  # S-HO
        # S1
        ham = np.kron(pz, boson_id) * 0.5
        ham += np.kron(px, boson_id) * 0.5 * paras.value[paras][0]  # epsilon
        # HO
        ham += np.kron(spin_id, ada) * paras.value[paras][1]  # omega
        # S1-HO
        ham += np.kron(pz, q) * paras.value[paras][2]  # lambda
        return ham

    def sb1sjc():  # S-HO
        # RIght
        #print('Using Hamiltonian Matrix', 'sb1sjc')
        ham = np.kron(pz, boson_id) * 0.5
        ham += np.kron(px, boson_id) * 0.5 * paras.value[paras][0]  # epsilon
        ham += np.kron(spin_id, ada) * paras.value[paras][1]  # omega
        ham += ft.reduce(np.kron, [pm, ad]) * paras.value[paras][2]  # lambda
        ham += ft.reduce(np.kron, [pp, a]) * paras.value[paras][2]  # lambda
        return ham

    # S1 - HO - S2, spin-boson
    def sb2s():
        # S1, S2
        term11 = np.kron(pz, np.kron(boson_id, spin_id)) * 0.5
        term12 = np.kron(spin_id, np.kron(boson_id, pz)) * 0.5
        term21 = np.kron(px, np.kron(boson_id, spin_id)) * 0.5 * paras.value[paras][0]  # epsilon
        term22 = np.kron(spin_id, np.kron(boson_id, px)) * 0.5 * paras.value[paras][0]  # epsilon
        # HO
        term3 = np.kron(spin_id, np.kron(ada, spin_id)) * paras.value[paras][1]  # omega
        # S1-HO, HO-S2
        term41 = np.kron(px, np.kron(q, spin_id)) * paras.value[paras][2]  # lambda
        term42 = np.kron(spin_id, np.kron(q, px)) * paras.value[paras][2]  # lambda
        return term11 + term12 + term21 + term22 + term3 + term41 + term42


    """ Post Selection """
    # with post selection
    #      HO
    # S1 - S2 - S3, Jaynes-Cummings with conservation
    def jc3s(_hopping=False):  # S-S-S-H
        ham = 0
        # S1, S2, S3
        ham += ft.reduce(np.kron, [pz, spin_id, spin_id, boson_id]) * 0.5
        ham += ft.reduce(np.kron, [spin_id, pz, spin_id, boson_id]) * 0.5
        ham += ft.reduce(np.kron, [spin_id, spin_id, pz, boson_id]) * 0.5
        # HO
        ham += ft.reduce(np.kron, [spin_id, spin_id, spin_id, ada]) * paras.value[paras][1]  # omega
        # S1-HO, S2-HO, S3-HO
        ham += ft.reduce(np.kron, [pm, spin_id, spin_id, ad]) * paras.value[paras][2]  # lambda
        ham += ft.reduce(np.kron, [pp, spin_id, spin_id, a]) * paras.value[paras][2]  # lambda
        ham += ft.reduce(np.kron, [spin_id, pm, spin_id, ad]) * paras.value[paras][2]  # lambda
        ham += ft.reduce(np.kron, [spin_id, pp, spin_id, a]) * paras.value[paras][2]  # lambda
        ham += ft.reduce(np.kron, [spin_id, spin_id, pm, ad]) * paras.value[paras][2]  # lambda
        ham += ft.reduce(np.kron, [spin_id, spin_id, pp, a]) * paras.value[paras][2]  # lambda
        if _hopping:
            # hopping S1 - S2, S2 - S3
            ham += ft.reduce(np.kron, [pm, pp, spin_id, boson_id]) * paras.value[paras][3]  # v
            ham += ft.reduce(np.kron, [pp, pm, spin_id, boson_id]) * paras.value[paras][3]  # v
            ham += ft.reduce(np.kron, [spin_id, pm, pp, boson_id]) * paras.value[paras][3]  # v
            ham += ft.reduce(np.kron, [spin_id, pp, pm, boson_id]) * paras.value[paras][3]  # v
        return ham

    def jc2s(_hopping=False): # S-S-H
        ham = 0
        # S1, S2, S3
        ham += ft.reduce(np.kron, [pz, boson_id, spin_id]) * 0.5
        ham += ft.reduce(np.kron, [spin_id, boson_id, pz]) * 0.5
        # HO
        ham += ft.reduce(np.kron, [spin_id, ada, spin_id]) * paras.value[paras][1]  # omega
        # S1-HO, S2-HO
        ham += ft.reduce(np.kron, [pm, ad, spin_id]) * paras.value[paras][2]  # lambda
        ham += ft.reduce(np.kron, [pp, a, spin_id]) * paras.value[paras][2]  # lambda
        ham += ft.reduce(np.kron, [spin_id, ad, pm]) * paras.value[paras][2]  # lambda
        ham += ft.reduce(np.kron, [spin_id, a, pp]) * paras.value[paras][2]  # lambda
        if _hopping:
            # hopping S1 - S2
            ham += ft.reduce(np.kron, [pm, boson_id, pp]) * paras.value[paras][3]  # v
            ham += ft.reduce(np.kron, [pp, boson_id, pm]) * paras.value[paras][3]  # v
        return ham


    # model passed
    if model == Model.SB1S: h_matrix = sb1s()
    elif model == Model.SB2S: h_matrix = sb2s()
    elif model == Model.SB1SPZ: h_matrix = sb1spz()
    elif model == Model.SB1SJC: h_matrix = sb1sjc()
    elif model == Model.JC2S: h_matrix = jc2s()
    elif model == Model.JC3F: h_matrix = jc3s()
    else: h_matrix = None

    return h_matrix
