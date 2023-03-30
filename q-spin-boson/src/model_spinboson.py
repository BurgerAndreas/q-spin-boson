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


class SSpinBosonSimulation(Simulation):
    def __init__(self, 
                 model = Model.SB1S, 
                 n_bos = 4, 
                 env = Env.ADC, 
                 paras = Paras.SB1S, 
                 gamma = 1., 
                 enc = Enc.BINARY, 
                 h = H.FRSTORD, 
                 steps = Steps.LOOP, 
                 dt = 0.3, 
                 eta = 1,
                 noise = .1,
                 initial = None,
                 optimal_formula = False):
        """Spin-boson model with a single spin."""
        super().__init__(model, n_bos, env, paras, gamma, enc, h, steps, dt, 
                         eta, noise, initial, optimal_formula)
        self.backend = FakeJakarta()
        self.opt_prdctfrml = {0.01: [H.SCNDORD, 0.2], 
                              0.1: [H.FRSTORD, 0.2],
                              1: [H.FRSTORD, 0.3]}
        # ------------------------------------------------------------
        if self.model not in [Model.SB1S, Model.SB1SPZ, Model.SB1SJC]:
            raise ValueError(f"Wrong class for model {self.model}.")
        # ------------------------------------------------------------
        self.set_default_simulation_parameters()
        if optimal_formula:
            self.set_optimal_product_formula()
        self.get_simulation()
    
    def set_dimensions(self) -> None:
        """Set dimensions, post selection, and qubit ordering of system."""
        # Hamiltonian, Pauli: S-HO
        # Circuit: HO-S
        # Keys: S-HO
        self.d_system = 2 * self.n_bos
        if self.enc in [Enc.BINARY, Enc.GRAY]:
            self.n_qubits_system = m.ceil(m.sqrt(self.d_system))
            self.qubits_system = [x for x in range(self.n_qubits_system)]
            n_qubits_bos = m.ceil(m.sqrt(self.n_bos))
            self.spins = [n_qubits_bos]
            self.s_a_pairs = [[self.n_qubits_system-1, self.n_qubits_system]]
            self.qc_empty = QuantumCircuit(QuantumRegister(n_qubits_bos, 'b'),
                                           QuantumRegister(1, 's'),
                                           QuantumRegister(1, 'a'))
        # no need to set ordered keys for binary encoding
        if self.enc == Enc.GRAY:
            ordered_keys = get_seq(self.enc, length=self.n_qubits_system - 1)
            for num, key in enumerate(ordered_keys[:self.n_bos]):
                ordered_keys[num] = f'0{key}'
                ordered_keys.append(f'1{key}')
            self.ordered_keys = ordered_keys
        elif self.enc == Enc.SPLITUNARY:
            self.n_qubits_system = 1 + self.n_bos
            self.qubits_system = [*range(0, self.n_qubits_system)]
            n_qubits_bos = self.bos
            self.spins = [n_qubits_bos]
            self.s_a_pairs = [[self.n_qubits_system-1, self.n_qubits_system]]
            self.qc_empty = QuantumCircuit(QuantumRegister(n_qubits_bos, 'b'),
                                           QuantumRegister(1, 's'),
                                           QuantumRegister(1, 'a'))
            # post selection and ordered keys
            bos_seq = unary_sequence(self.n_bos)
            post_select = []
            for bos_string in bos_seq:
                post_select.append('0' + bos_string)
            for bos_string in bos_seq:
                post_select.append('1' + bos_string)
            self.post_select = post_select
            self.ordered_keys = post_select
        elif self.enc == Enc.FULLUNARY:
            self.n_qubits_system = self.d_system
            self.qubits_system = [*range(0, self.n_qubits_system)]
            # no spin or s-a pair
            self.qc_empty = QuantumCircuit(
                QuantumRegister(self.n_qubits_system, 'sb'))
            self.post_select = unary_sequence(self.d_system)
            self.ordered_keys = unary_sequence(self.n_qubits_system)
        self.n_qubits = self.n_qubits_system + 1
        # no environment interaction
        if self.env == Env.NOENV:
            self.s_a_pairs = None
        return
    
    def set_initial_state(self) -> None:
        """Set initial state of system.
        Can be overwritten by passing vector for system."""
        bb = np.eye(self.n_bos, dtype=int) # boson basis vectors (onehot)
        self.i_system: NDArray = ft.reduce(np.kron, [EX1, bb[0]])
        if self.initial is not None:
            if self.initial.shape != self.i_system.shape:
                raise ValueError(
                    f'Initial state must have length {2**self.n_qubits_system}!')
            self.i_system = self.initial
        # With environment ancillas
        i_full: NDArray = ft.reduce(np.kron, [GS0, self.i_system])
        # Standard binary encoding
        i_full_binary_rev = to_binary(
            np.nonzero(i_full)[0][0], self.n_qubits)
        if self.enc == Enc.GRAY:
            binary_keys_system = sb_sequence(numbers=len(self.ordered_keys))
            ordered_keys_full = [f'0{_key}' for _key in self.ordered_keys]
            binary_keys_full = [f'0{_key}' for _key in binary_keys_system]
            ones_in_binary = binary_keys_full.index(i_full_binary_rev)
            i_full_binary_rev = ordered_keys_full[ones_in_binary]
        elif self.enc == Enc.SPLITUNARY:
            # i_full = np.hstack((GS0, EX1, bb[0]))
            i_full_binary_rev = ''.join(str(int(p)) for p in i_full)
        elif self.enc == Enc.FULLUNARY:
            i_full_binary_rev = ''.join(str(int(p)) for p in i_full[0])
        # qiskit ordering
        self.i_full_binary = i_full_binary_rev[::-1]  
        return

    def build_operators(self) -> None:
        """Build spin and boson number operators.
        Used for exact reference and labels.
        """
        boson_id = np.eye(self.n_bos)
        spin_id = np.eye(2)
        ada = np.zeros([self.n_bos, self.n_bos])
        for _b in range(self.n_bos):
            ada[_b, _b] = _b
        self.sz_ops = [ft.reduce(np.kron, [PZ, boson_id])]
        self.sx_ops = [ft.reduce(np.kron, [PX, boson_id])]
        self.sy_ops = [ft.reduce(np.kron, [PY, boson_id])]
        self.ada = ft.reduce(np.kron, [spin_id, ada])
        self.l_ops = [self.gamma * np.kron(PM, boson_id)]
        return


# class SBHamiltonianSimulation(SpinBosonSimulation):
#     def __init__(self, model):
#         super().__init__(model)


class DSpinBosonSimulation(Simulation):
    def __init__(self, 
                 model = Model.SB2S, 
                 n_bos = 4, 
                 env = Env.ADC, 
                 paras = Paras.SB2S, 
                 gamma = 1., 
                 enc = Enc.BINARY, 
                 h = H.FRSTORD, 
                 steps = Steps.LOOP, 
                 dt = 0.3, 
                 eta = 1,
                 noise = .1,
                 initial = None,
                 optimal_formula = False):
        """Spin-boson model with two spins."""
        super().__init__(model, n_bos, env, paras, gamma, enc, h, steps, dt, 
                         eta, noise, initial, optimal_formula)
        self.backend = FakeJakarta()
        self.opt_prdctfrml = {0.01: [H.SCNDORD, 0.2], 
                              0.1: [H.FRSTORD, 0.2],
                              1: [H.FRSTORD, 0.3]}
        # ------------------------------------------------------------
        self.model = Model.SB2S
        # ------------------------------------------------------------
        self.set_default_simulation_parameters()
        if optimal_formula:
            self.set_optimal_product_formula()
        self.get_simulation()

    def set_dimensions(self) -> None:
        """Set dimensions, post selection, and qubit ordering of system.
        Sets n_qubits_system, qubits_system, qc_empty, post_select,
        ordered_keys.
        """
        # Hamiltonian, Pauli: s1-HO-s2
        # Circuit: s2-HO-s1
        # Keys: s1-HO-s2
        self.d_system = 2 * self.n_bos * 2
        # qubits
        if self.enc in [Enc.BINARY, Enc.GRAY]:
            self.n_qubits_system = m.ceil(m.sqrt(self.d_system))
            self.qubits_system = [*range(1, self.n_qubits_system+1)]
            n_qubits_bos = m.ceil(m.sqrt(self.n_bos))
        if self.enc == Enc.SPLITUNARY:
            self.n_qubits_system = 1 + self.n_bos + 1
            self.qubits_system = [*range(1, self.n_qubits_system+1)]
            n_qubits_bos = self.n_bos
        # circuit
        if self.enc in [Enc.BINARY, Enc.GRAY, Enc.SPLITUNARY]:
            self.spins = [n_qubits_bos+2, 1]
            self.s_a_pairs = [[self.n_qubits_system, self.n_qubits_system+1],
                              [1, 0]]
            self.qc_empty = QuantumCircuit(QuantumRegister(1, 'a1'),
                                            QuantumRegister(1, 's1'),
                                            QuantumRegister(n_qubits_bos, 'b'),
                                            QuantumRegister(1, 's2'),
                                            QuantumRegister(1, 'a2'))
        # post selection and ordering
        # no need to set ordered keys for binary encoding
        if self.enc in [Enc.GRAY, Enc.SPLITUNARY]:
            bos_seq = get_seq(self.enc, length=self.n_qubits_system-2)
            ordered_keys = []
            for key in bos_seq:
                ordered_keys.append(f'0{key}0')
                ordered_keys.append(f'0{key}1')
            for key in bos_seq:
                ordered_keys.append(f'1{key}0')
                ordered_keys.append(f'1{key}1')
            self.ordered_keys = ordered_keys
            self.post_select = ordered_keys
        # full unary
        if self.enc == Enc.FULLUNARY:
            self.n_qubits_system = self.d_system
            self.qubits_system = [*range(1, self.n_qubits_system+1)]
            # no spins or s-a pairs
            self.qc_empty = QuantumCircuit(
                QuantumRegister(self.n_qubits_system, 'sb'))
            self.post_select = unary_sequence(self.d_system)
            self.ordered_keys = unary_sequence(self.n_qubits_system)
        self.n_qubits = 1 + self.n_qubits_system + 1
        # No environment interaction
        if self.env == Env.NOENV:
            self.s_a_pairs = None
        return
    
    def set_initial_state(self) -> None:
        """Set initial state of system.
        Can be overwritten by passing vector for system."""
        bb = np.eye(self.n_bos, dtype=int) # boson basis vectors (onehot)
        self.i_system: NDArray = ft.reduce(np.kron, [EX1, bb[0], GS0])
        if self.initial is not None:
            if self.initial.shape != self.i_system.shape:
                raise ValueError(
                    f'Initial state must have length {2**self.n_qubits_system}!')
            self.i_system = self.initial
        # With environment ancillas
        i_full: NDArray = ft.reduce(np.kron, [GS0, self.i_system, GS0])
        # Standard binary encoding
        i_full_binary_rev = to_binary(
            np.nonzero(i_full)[0][0], self.n_qubits)
        if self.enc == Enc.GRAY:
            binary_keys_system = sb_sequence(numbers=len(self.ordered_keys))
            ordered_keys_full = [f'0{_key}0' for _key in self.ordered_keys]
            binary_keys_full = [f'0{_key}0' for _key in binary_keys_system]
            ones_in_binary = binary_keys_full.index(i_full_binary_rev)
            i_full_binary_rev = ordered_keys_full[ones_in_binary]
        elif self.enc == Enc.SPLITUNARY:
            # i_full = np.hstack((GS0, EX1, bb[0], GS0, GS0))
            i_full_binary_rev = ''.join(str(int(p)) for p in i_full)
        elif self.enc == Enc.FULLUNARY:
            i_full_binary_rev = ''.join(str(int(p)) for p in i_full[0])
        # Qiskit ordering
        self.i_full_binary = i_full_binary_rev[::-1]  
        return
    
    def build_operators(self) -> None:
        """Build spin and boson number operators.
        Used for exact reference and labels.
        """
        boson_id = np.eye(self.n_bos)
        spin_id = np.eye(2)
        ada = np.zeros([self.n_bos, self.n_bos])
        for _b in range(self.n_bos):
            ada[_b, _b] = _b
        self.sz_ops = [ft.reduce(np.kron, [PZ, boson_id, spin_id]),
                       ft.reduce(np.kron, [spin_id, boson_id, PZ])]
        self.sx_ops = [ft.reduce(np.kron, [PX, boson_id, spin_id]),
                       ft.reduce(np.kron, [spin_id, boson_id, PX])]
        self.sy_ops = [ft.reduce(np.kron, [PY, boson_id, spin_id]),
                       ft.reduce(np.kron, [spin_id, boson_id, PY])]
        self.ada = ft.reduce(np.kron, [spin_id, ada, spin_id])
        self.l_ops = [self.gamma * ft.reduce(np.kron, [PM, boson_id, spin_id]),
                      self.gamma * ft.reduce(np.kron, [spin_id, boson_id, PM])]
        return