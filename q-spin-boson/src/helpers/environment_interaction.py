import numpy as np
import math as m
import cmath as cmath

from qiskit import Aer, QuantumCircuit


def swap_matrix():
    """Swap matrix.
    0 spin qubit, 1 ancilla.
    theta = m.asin(m.sqrt(1 - m.exp(-gamma * t))).
    See Palma2019.
    """
    return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def pswap_matrix(theta):
    """Pswap matrix.
    0 spin qubit, 1 ancilla.
    Takes |1>=(0,1) to |0>=(1,0).
    See Palma2019.
    """
    return np.array(
        [
            [m.cos(theta) - 1j * m.sin(theta), 0, 0, 0],
            [0, m.cos(theta), -1j * m.sin(theta), 0],
            [0, -1j * m.sin(theta), m.cos(theta), 0],
            [0, 0, 0, m.cos(theta) - 1j * m.sin(theta)],
        ]
    )


def ad_matrix(theta):
    """Amplitude damping matrix.
    0 spin qubit, 1 ancilla.
    takes |1>=(0,1) to |0>=(1,0)
    See Landi2018_QInfo.
    """
    return np.array(
        [[1, 0, 0, 0],
         [0, m.cos(theta), - 1j * m.sin(theta), 0],
         [0, - 1j * m.sin(theta), m.cos(theta), 0],
         [0, 0, 0, 1]]
    )


def adc_circuit(_theta):
    """Amplitude damping circuit.
    0 spin qubit, 1 ancilla.
    takes |1>=(0,1) to |0>=(1,0)
    theta = 2*theta
    See Nielsen2010.
    """
    f, a = 0, 1
    circ = QuantumCircuit(2)
    circ.cry(theta=_theta, target_qubit=a, control_qubit=f)
    circ.cx(target_qubit=f, control_qubit=a)
    return circ


def kraus0_diluted(gamma=1, t=1):
    """Unitary Dilation of Kraus operators for zero temperature.
    0 spin qubit, 1 ancilla.
    takes |1>=(0,1) to |0>=(1,0)
    """
    x = m.sqrt(m.exp(-gamma * t))
    y = np.sqrt(1 - m.exp(-gamma * t))
    # Unitary 1-dilution
    k1_diluted = np.matrix([[1, 0, 0, 0],
                            [0, x, 0, y],
                            [0, 0, -1, 0],
                            [0, y, 0, -x]])
    k2_diluted = np.matrix([[0, y, x, 0],
                            [0, 0, 0, 1],
                            [1, 0, 0, 0],
                            [0, x, -y, 0]])
    return k1_diluted, k2_diluted


def kraus0(gamma=1, t=1):
    """Kraus operators for zero temperature.
    """
    x = np.sqrt(np.exp(-gamma * t))
    y = np.sqrt(1 - np.exp(-gamma * t))
    k1 = np.matrix([[1, 0], [0, x]])
    k2 = np.matrix([[0, y], [0, 0]])
    return k1, k2


def kraust(gamma=1, t=1, lam=0.5):
    """Kraus operators for finite temperature.
    """
    x = np.sqrt(np.exp(-gamma * t))
    y = np.sqrt(1 - np.exp(-gamma * t))
    k1 = np.sqrt(lam) * np.matrix([[1, 0], [0, x]])
    k2 = np.sqrt(lam) * np.matrix([[0, y], [0, 0]])
    k3 = np.sqrt(1 - lam) * np.matrix([[x, 0], [0, 1]])
    k4 = np.sqrt(1 - lam) * np.matrix([[0, 0], [y, 0]])
    return k1, k2, k3, k4
