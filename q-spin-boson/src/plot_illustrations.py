import numpy as np
import matplotlib.pyplot as plt
import math as m
import scipy.linalg
import os

from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, IBMQ, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeJakarta

from settings.paths import DIR_PLOTS_CIRCUIT
from settings.types import Env
from src.model_twolevel import TwoLvlSimulation

h_label = r'U' # r'$e^{-i \hat{H} \Delta t}$'
d_label = r'D' 

def save_circuit_latex(qc, name):
    """Takes a circuit, cleans it up, and saves it as a latex file."""
    # remove numbers
    _qubits = qc.num_qubits
    latex_source = qc.draw('latex_source')
    for number in range(_qubits):
        look_for = "<<<{" + str(number) + "}"
        latex_source = latex_source.replace(look_for,"<<<{}")
    # save
    dir = f'{DIR_PLOTS_CIRCUIT}{name}.tex'
    with open(dir, 'w') as f:
        f.write(latex_source)
    return dir, latex_source, qc


def circuit_adc():
    """Illustration of ADC circuit."""
    qc = QuantumCircuit(QuantumRegister(1, 's'), QuantumRegister(1, 'a'))
    s = 0
    a = 1
    qc.cry(theta=np.pi, control_qubit=s, target_qubit=a, 
               label=r'Ry($\theta$)')
    qc.cx(a, s)
    # qc.unitary(np.eye(2), [s], label='M')
    qc.reset(a)
    return save_circuit_latex(qc, 'adc_latex')


def circuit_dissipation():
    """Illustration of dissipation circuit."""
    qc = QuantumCircuit(QuantumRegister(1, 's'), QuantumRegister(1, 'a'))
    a = 1
    qc.unitary(np.eye(4), [0,1], label='D')
    qc.reset(a)
    return save_circuit_latex(qc, 'd_latex')


def circuit_sb1s():
    """Illustration of single spin Spin-Boson circuit."""
    sys_qubits = [*range(3)]
    all_qubits = [*range(4)]
    qc = QuantumCircuit(QuantumRegister(2, 'b'), 
                          QuantumRegister(1, 's'), 
                          QuantumRegister(1, 'a'))  
    a = qc.num_qubits - 1
    qc.x(2)
    qc.barrier(all_qubits)
    for t in range(3):
        if t != 0:
            qc.reset(a)
        qc.unitary(np.eye(2**3), sys_qubits, label=h_label)
        qc.unitary(np.eye(2**2), [2, 3], label=d_label)
    qc.barrier(all_qubits)
    #qc.measure(sys_qubits, [*range(len(sys_qubits))])
    qc.unitary(np.eye(2), [0], label='M')
    qc.unitary(np.eye(2), [1], label='M')
    qc.unitary(np.eye(2), [2], label='M')
    return save_circuit_latex(qc, 'sb1f_latex')


def circuit_sb2s():
    """Illustration of two spin Spin-Boson circuit."""
    qc = QuantumCircuit(QuantumRegister(1, 'a1'), QuantumRegister(1, 's1'), 
                          QuantumRegister(2, 'b'), 
                          QuantumRegister(1, 's2'), QuantumRegister(1, 'a2'))
    s1 = 1
    a1 = 0
    a2 = 5
    qc.x(s1)
    qc.barrier([*range(1, 5)])
    for t in range(3):
        if t != 0:
            qc.reset(a1)
            qc.reset(a2)
        qc.unitary(np.eye(2**4), range(1, 5), label=h_label)
        qc.unitary(np.eye(2**2), [0, 1], label=d_label)
        qc.unitary(np.eye(2**2), [4, 5], label=d_label)
    qc.barrier([*range(1, 5)])
    #qc.measure([*range(1, 5)], [*range(0, 4)])
    qc.unitary(np.eye(2), [1], label='M')
    qc.unitary(np.eye(2), [2], label='M')
    qc.unitary(np.eye(2), [3], label='M')
    qc.unitary(np.eye(2), [4], label='M')
    return save_circuit_latex(qc, 'sb2f_latex')


def circuit_dissipation_gates(env=Env.ADC, real_device=False):
    """Circuits which implement dissipaton.
    Optionally transpiled onto real device gate set.
    """
    qc = QuantumCircuit(QuantumRegister(1, 's'), QuantumRegister(1, 'a'))
    sim = TwoLvlSimulation(env=env)
    # qc = trotter.add_trotter(qc, hamilton, ham_q, h_matrix, system_qubits, collis, f_a_pairs, t, gamma, reset_a=True)
    if real_device:
        qc = transpile(qc, backend=FakeJakarta(), optimization_level=3)
    else:
        qc = transpile(qc, backend=AerSimulator(), optimization_level=3, 
                         basis_gates=['cx', 'id', 'x', 'sx', 'rz', 'reset'])
    return