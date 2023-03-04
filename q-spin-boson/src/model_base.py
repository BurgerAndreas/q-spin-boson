import numpy as np
from typing import List, Sequence, Any # typechecking with mypy
from numpy.typing import NDArray # typechecking with mypy
import matplotlib.pyplot as plt
import math as m
import scipy.linalg # expm
from dotenv import load_dotenv  # load environment variables from env file
import os
import functools as ft  # XX = ft.reduce(np.kron, [A, B, C, D, E])
import time as time # code timing
from datetime import datetime 
import pickle # save/load objects

from qiskit import Aer, QuantumCircuit, QuantumRegister, transpile, IBMQ, assemble
from qiskit.visualization import plot_histogram, array_to_latex, plot_gate_map, plot_circuit_layout
from qiskit.opflow import I, X, Y, Z, Zero, One, Plus, Minus, PauliTrotterEvolution, CircuitSampler, PauliOp, Suzuki
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.providers.fake_provider import FakeManila, FakeVigo, FakeAthens, FakeSingapore, FakeJakarta, FakeToronto
# from qiskit.providers.aer import AerSimulator
# from qiskit.providers.aer.noise import NoiseModel
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
# qiskit.ignis.mitigation	qiskit_terra.mitigation
# last version: qiskit==0.36.2 qiskit-ignis==0.7.1 qiskit-ibmq-provider==0.20.1
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.synthesis import QDrift, LieTrotter, SuzukiTrotter

from settings.types import Enc, Env, H, Model, Steps
from settings.parameters import Paras


# load environment variables from env file
# DIR_ENV = os.path.join(os.path.dirname(__file__), 'settings/paths.env')
DIR_ENV = "q-spin-boson/settings/paths.env"
load_dotenv(DIR_ENV)
DIR_SAVED_MODELS = os.getenv('DIR_SAVED_MODELS', DIR_ENV)
DIR_PLOTS = os.getenv('DIR_PLOTS', DIR_ENV)
DIR_PLOTS_CIRCUIT = os.getenv('DIR_PLOTS_CIRCUIT', DIR_ENV)


class Simulation():
    """Simulation base class. Don't use this class directly, use one of the
    """
    # class wide hyperparameters
    shots = 8 * 1024

    def __init__(self, 
                 model: Model, 
                 n_b: int, 
                 env: Env, 
                 paras: Paras, 
                 gamma: float, 
                 enc: Enc, 
                 h: H, 
                 steps: Steps, 
                 dt: float, 
                 eta: int, 
                 errfctr: float = 1.):
        """Create a Simulation object.
        Args:
            model (str): Model for system. Spin-Boson or Jaynes-Cummings.
            n_b (int): Number of bosons in system.
            env (Env): Model for environment.
            h (H): 1st order (trotter), 2nd order (suzuki) product formula 
                or isometric decomposition.

        """
        # --------------------------------------------------------------------
        # set parameters
        self.model = model
        self.n_b = n_b
        self.env = env
        self.paras = paras
        self.gamma = gamma
        self.h = h
        self.steps = steps
        self.dt = np.round(dt, 2)
        self.eta = eta
        self.errfctr = errfctr
        # --------------------------------------------------------------------
        # parameters with default values
        self.qst = True
        self.initial_state = None
        self.name = None
        # --------------------------------------------------------------------
        # containers for simulation calculations
        self.hamiltonian = None # hamiltonian
        self.noise_model = None # noise model
        self.i_system: NDArray = None # initial system state (w/o environment)
        self.i_full_binary: str = None # initial full state (w/ environment)
        self.qc_empty = None # empty quantum circuit w qubits
        self.post_select: NDArray = None # post selection for measurement
        self.ordered_keys: List[int] = None # ordered keys for measurement
        self.qubits_system: List[int] = None # qubits for system
        self.n_qubits_system: int = None # number of qubits for system
        self.d_system: int = None # dimension of system
        self.spins: List[int] = None # spins of system
        self.s_a_pairs: List[int] = None # spin-ancilla pairs (system, env)
        # --------------------------------------------------------------------
        # containers for simulation results
        # noiseless circuit
        self.evo: NDArray = None # evolution of system
        self.pz: NDArray = None # z measurement on spin(s)
        self.px: NDArray = None # x measurement on spin(s)
        self.py: NDArray = None # y measurement on spin(s)
        self.dm: NDArray = None # density matrix
        self.pzcorr: NDArray = None # connected correlation <pz pz> - <pz><pz>
        self.pxcorr: NDArray = None # connected correlation <px px> - <px><px>
        self.bosons: NDArray = None # boson occupation number
        # error mitigated circuit = noise model + measurement mitigation
        self.evo_em: NDArray = None # evolution of system
        self.pz_em: NDArray = None # z measurement
        self.px_em: NDArray = None # x measurement
        self.py_em: NDArray = None # y measurement
        self.dm_em: NDArray = None # density matrix
        self.pzcorr_em: NDArray = None # connected correlation
        self.pxcorr_em: NDArray = None # connected correlation
        self.bosons_em: NDArray = None # boson occupation number
        # exact reference = qutip linblad master equation solver
        self.evo_exact: NDArray = None # evolution of system
        self.pz_exact: NDArray = None # z measurement
        self.px_exact: NDArray = None # x measurement
        self.py_exact: NDArray = None # y measurement
        self.dm_exact: NDArray = None # density matrix
        self.pzcorr_exact: NDArray = None # connected correlation
        self.pxcorr_exact: NDArray = None # connected correlation
        self.bosons_exact: NDArray = None # boson occupation number
        # infidelity 
        self.infidelity: NDArray = None # exact - noiseless circuit
        self.infidelity_em: NDArray = None # exact - error mitigated circuit
        # post selection quota
        self.post_selection_quota: NDArray = None 
        # --------------------------------------------------------------------
        # do
        self.fix_parameters()
        self.update_name()
        load_status = self.load()
        if load_status == 404:
            self.simulate_qc()
            self.simulate_exact_linblad()
            self.compare_simulations()
            load_status = self.save()


    def fix_parameters(self):
        if self.enc in [Enc.SPLITUNARY, Enc.FULLUNARY]:
            self.qst = False
        if self.h in [H.NOH, H.ISODECOMP]:
            self.enc = Enc.BINARY

    def update_name(self):
        # model
        name = f'{self.model.name}_{self.env.name}_b{self.n_b}'
        # model parameters
        name += f'_{self.paras.name}_g{self.gamma:.2f}'
        # simulation parameters
        name += f'_{self.h.name}'
        if self.steps == Steps.LOOP:
            name += f'_s{self.steps}_d{self.dt:.2f}'
        else:
            name += f'_s{self.steps}_n{self.eta}'
        name += f'_e{self.errfctr:.2f}'
        self.name = name
        return name

    def load(self):
        # check if saved model exists
        path = f'{DIR_SAVED_MODELS}{self.name}.pickle'
        if path.is_file():
            # load model
            self.__dict__.update(pickle.load(
                open(path, mode='rb')
                ).__dict__)
            return 200
        # saved model does not exist
        return 404

    def save(self):
        # save model
        with open(f'{DIR_SAVED_MODELS}{self.name}.pickle', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return 201

    def circuit_to_latex(self):
        return 0

    def circuit_to_image(self):
        return 0

    def simulate_exact_linblad(self):

        # calculate observables from density matrix

        return 0

    def simulate_qc(self):
        """Calculate the model"""
        # Get Noise Model

        # Get Hamiltonian

        # Containers for results

        # Loop over all time steps

            # Set initial state

            # Add Trotter step

            # State tomography

            # measurements
            # z-basis

            # x-basis

            # y-basis

            # two-spin-correlations

        # calculate observables from measurements

        return 0

    def get_hamiltonian(self):
        return 0

    def add_trotter_step(self):
        return 0

    def meas_tomography(self):
        return 0

    def meas_z(self):
        return 0

    def compare_simulations(self):
        # Fidelity
        return 0