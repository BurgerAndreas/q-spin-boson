import numpy as np
from typing import List, Sequence, Any # typechecking with mypy
from numpy.typing import NDArray # typechecking with mypy
import matplotlib.pyplot as plt
import math as m
from dotenv import load_dotenv  # load environment variables from env file
import functools as ft  # XX = ft.reduce(np.kron, [A, B, C, D, E])
import time as time # code timing
from datetime import datetime 

from qiskit import Aer, QuantumCircuit, QuantumRegister, transpile, IBMQ, assemble
from qiskit.visualization import plot_histogram, array_to_latex, plot_gate_map, plot_circuit_layout
from qiskit.opflow import I, X, Y, Z, Zero, One, Plus, Minus, PauliTrotterEvolution, CircuitSampler, PauliOp, PauliSumOp, Suzuki
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.providers.fake_provider import FakeJakarta, FakeToronto
from qiskit.providers.aer import AerSimulator # old qiskit versions
from qiskit.providers.aer.noise import NoiseModel # old qiskit versions
# from qiskit.providers.aer.utils import insert_noise # old qiskit versions
# from qiskit_aer.noise import NoiseModel # new qiskit versions
# from qiskit_aer import AerSimulator # new qiskit versions
# qiskit.ignis.mitigation	qiskit_terra.mitigation
# last version: qiskit==0.36.2 qiskit-ignis==0.7.1 qiskit-ibmq-provider==0.20.1
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.synthesis import QDrift, LieTrotter, SuzukiTrotter

from settings.types import Enc, Env, H, Model, Steps
from settings.parameters import Paras
from settings.conventions import GS0, EX1, PM, PZ, PX, PY
from src.helpers.noise_modeling import modified_noise_model
from src.helpers.binary import get_seq, unary_sequence, to_binary, sb_sequence
from src.model_base import Simulation

class TwoLvlSimulation(Simulation):
    def __init__(self, 
                 model = Model.TWOLVL, 
                 n_bos = None, 
                 env = Env.ADC, 
                 paras = None, 
                 gamma = 1., 
                 enc = None, 
                 h = None, 
                 steps = Steps.LOOP, 
                 dt = 0.3, 
                 eta = 1,
                 noise = .1,
                 initial = None):
        """Spin-boson model with two spins."""
        super().__init__(model, n_bos, env, paras, gamma, enc, h, steps, dt, 
                         eta, noise, initial)
        self.backend = FakeJakarta()
        self.opt_prdctfrml = {0.01: [H.SCNDORD, 0.2], 
                              0.1: [H.FRSTORD, 0.2],
                              1: [H.FRSTORD, 0.3]}
        # ------------------------------------------------------------
        self.model = Model.TWOLVL
        self.n_bos = None
        self.paras = None
        self.enc = None
        self.h = None
        self.set_default_simulation_parameters()

    def set_dimensions(self) -> None:
        """Set dimensions, post selection, and qubit ordering of system."""
        self.d_system = 2
        self.qc_empty = QuantumCircuit(QuantumRegister(1, 's'), 
                                       QuantumRegister(1, 'a'))
        self.n_qubits_system = 1
        self.n_qubits = 2
        self.qubits_system = [0]
        self.spins = [0]
        self.s_a_pairs = [[0, 1]]
        self.ordered_keys = ['0', '1']
        self.post_select = self.ordered_keys
        return
    
    def set_initial_state(self) -> None:
        """Set initial state of system.
        Can be overwritten by passing vector for system."""
        self.i_system: NDArray = ft.reduce(np.kron, [EX1])
        if self.initial is not None:
            if initial.shape != i_system.shape:
                raise ValueError(
                    f'Initial state must have length {2**self.n_qubits_system}!')
            self.i_system = self.initial
        # With environment ancillas
        i_full: NDArray = ft.reduce(np.kron, [self.i_system, GS0])
        i_full_binary_rev = to_binary(
            np.nonzero(i_full)[0][0], self.n_qubits)
        # Qiskit ordering
        self.i_full_binary = i_full_binary_rev[::-1]
        return
    
    def build_operators(self) -> None:
        """Build spin and boson number operators.
        Used for exact reference and labels.
        """
        self.sz_ops = [PZ]
        self.sx_ops = [PX]
        self.sy_ops = [PY]
        self.l_ops = [self.gamma * PM]
        return