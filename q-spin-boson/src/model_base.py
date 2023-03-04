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

from qiskit import Aer, QuantumCircuit, QuantumRegister, transpile, IBMQ, execute
from qiskit.visualization import plot_histogram, array_to_latex, plot_gate_map, plot_circuit_layout
from qiskit.opflow import I, X, Y, Z, Zero, One, Plus, Minus, PauliTrotterEvolution, CircuitSampler, PauliOp, PauliSumOp, Suzuki
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.providers.fake_provider import FakeJakarta, FakeToronto
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
# qiskit.ignis.mitigation	qiskit_terra.mitigation
# last version: qiskit==0.36.2 qiskit-ignis==0.7.1 qiskit-ibmq-provider==0.20.1
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit_experiments.library import StateTomography
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.synthesis import QDrift, LieTrotter, SuzukiTrotter

from settings.types import Enc, Env, H, Model, Steps
from settings.parameters import Paras
from src.helpers.noise_modeling import modified_noise_model
from src.helpers.hamiltonian_matrix import hamiltonian_as_matrix
from src.helpers.hamiltonian_qiskit import matrix_to_pauli, pauli_to_qiskit, hamiltonian_as_paulis
from src.helpers.environment_interaction import swap_matrix, pswap_matrix, ad_matrix, adc_circuit, kraus0_diluted 
from src.helpers.binary import unary_sequence, sb_sequence, get_seq
from src.helpers.error_mitigation import mitigate_error

# load environment variables from env file
# DIR_ENV = os.path.join(os.path.dirname(__file__), 'settings/paths.env')
DIR_ENV = "q-spin-boson/settings/paths.env"
load_dotenv(DIR_ENV)
DIR_SAVED_MODELS = os.getenv('DIR_SAVED_MODELS', DIR_ENV)
DIR_PLOTS = os.getenv('DIR_PLOTS', DIR_ENV)
DIR_PLOTS_CIRCUIT = os.getenv('DIR_PLOTS_CIRCUIT', DIR_ENV)


class Simulation():
    """Simulation base class. 
    Don't use this class directly, use one of the child classes instead.
    """
    # class wide hyperparameters
    shots = 8 * 1024

    def __init__(self, 
                 model: Model, 
                 n_bos: int, 
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
        To load / simulate the model, use the get_simulation() method.
        Args:
            model (str): Model for system. Spin-Boson or Jaynes-Cummings.
            n_bos (int): Number of bosons in system.
            env (Env): Model for environment.
            h (H): 1st order (trotter), 2nd order (suzuki) product formula 
                or isometric decomposition.
        """
        # --------------------------------------------------------------------
        # set parameters
        self.model = model
        self.n_bos = n_bos
        self.env = env
        self.paras = paras
        self.gamma = gamma
        self.enc = enc
        self.h = h
        self.steps = steps
        self.dt = np.round(dt, 2)
        self.eta = eta
        self.errfctr = errfctr
        # --------------------------------------------------------------------
        # parameters
        self.qst = True
        self.initial_state = None
        self.name = None
        # --------------------------------------------------------------------
        # containers for simulation calculations
        self.h_op: PauliSumOp = None # hamiltonian
        self.h_mat: NDArray = None # hamiltonian
        self.backend = None # fake backend
        self.noise_model = None # noise model
        self.i_system: NDArray = None # initial system state (w/o environment)
        self.i_full_binary: str = None # initial full state (w/ environment)
        self.timesteps: NDArray = None # timesteps of evolution
        self.qc_empty = None # empty quantum circuit w qubits
        self.post_select: NDArray = None # post selection for measurement
        self.ordered_keys: List[int] = None # ordered keys for measurement
        self.qubits_system: List[int] = None # qubits for system
        self.n_qubits_system: int = None # number of qubits for system
        self.n_qubits: int = None # total number of qubits
        self.d_system: int = None # dimension of system
        self.spins: List[int] = None # spins of system
        self.s_a_pairs: List[int] = None # spin-ancilla pairs (system, env)
        # --------------------------------------------------------------------
        # containers for simulation results
        # noiseless circuit
        self.evo = []  # evolution of system
        self.sz = []  # z measurement on spin(s)
        self.sx = []  # x measurement on spin(s)
        self.sy = []  # y measurement on spin(s)
        self.dm = []  # density matrix
        self.szcorr = []  # connected correlation <sz sz> - <sz><sz>
        self.sxcorr = []  # connected correlation <sx sx> - <sx><sx>
        self.bosons = []  # boson occupation number
        # error em circuit = noise model + measurement mitigation
        self.evo_em = []  # evolution of system
        self.sz_em = []  # z measurement
        self.sx_em = []  # x measurement
        self.sy_em = []  # y measurement
        self.dm_em = []  # density matrix
        self.szcorr_em = []  # connected correlation
        self.sxcorr_em = []  # connected correlation
        self.bosons_em = []  # boson occupation number
        # exact reference = qutip linblad master equation solver
        self.evo_exact = []  # evolution of system
        self.sz_exact = []  # z measurement
        self.sx_exact = []  # x measurement
        self.sy_exact = []  # y measurement
        self.dm_exact = []  # density matrix
        self.szcorr_exact = []  # connected correlation
        self.sxcorr_exact = []  # connected correlation
        self.bosons_exact = []  # boson occupation number
        # infidelity 
        self.infidelity = []  # exact - noiseless circuit
        self.infidelity_em = []  # exact - error em circuit
        # post selection quota
        self.ps_quota = [] 
        self.ps_quota_em = [] 
        # --------------------------------------------------------------------
        self.load_status: int = 0
        
    def get_simulation(self):
        self.fix_parameters()
        self.update_name()
        self.load_status = self.load()
        if self.load_status == 404:
            self.simulate_qc()
            self.simulate_exact_linblad()
            self.compare_simulations()
            self.load_status = self.save()
        return self

    def fix_parameters(self):
        if self.enc in [Enc.SPLITUNARY, Enc.FULLUNARY]:
            self.qst = False
        if self.h in [H.NOH, H.ISODECOMP]:
            self.enc = Enc.BINARY

    def update_name(self):
        # model
        name = f'{self.model.name}_{self.env.name}_b{self.n_bos}'
        # model parameters
        name += f'_{self.paras.name}_g{self.gamma:.2f}'
        # simulation parameters
        name += f'_{self.h.name}'
        if self.steps == Steps.LOOP:
            name += f'_s{self.steps}_d{self.dt:.2f}'
        else:
            name += f'_s{self.steps}_n{self.eta}'
        name += f'_e{self.errfctr:.2f}_{self.backend.backend_name}'
        self.name = name.lower()
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
    
    def set_default_simulation_parameters(self) -> None:
        """Set initial state, hamiltonian, noise model, backend, etc.
        Called in __init__ of child classes. 
        Parameters can be overwritten by user.
        """
        # Get Noise Model
        if self.errfctr == 1.:
            self.noise_model = AerSimulator.from_backend(self.backend)
        else:
            self.noise_model = AerSimulator(
                noise_model=modified_noise_model(self.backend, self.errfctr))
        # Dimensionality
        self.set_dimensions()
        # Hamiltonian
        self.set_hamiltonian()
        # Initial state
        self.set_initial_state()
        # time steps
        if self.steps == Steps.LOOP:
            self.timesteps = np.arange(0, 2.1, self.dt)
        else:
            self.timesteps = np.arange(0, 2.1, .1)
        return 
    
    def set_hamiltonian(self) -> tuple[NDArray, PauliSumOp]:
        # only directly used with isometric decomposition
        h_mat = hamiltonian_as_matrix(self.model, self.n_bos, self.paras)
        if self.enc == Enc.FULLUNARY:
            enc_seq = unary_sequence(self.d_system)
            h_pauli = matrix_to_pauli(h_mat, self.enc, enc_seq, 0)
            h_op = eval(pauli_to_qiskit(h_pauli, self.d_system))
            # evolution_op = h_op.exp_i()
            # trotterized_op = PauliTrotterEvolution(trotter_mode='trotter', reps=1).convert(evolution_op)
            # qc_e = trotterized_op.to_circuit()
        else:
            h_op = eval(hamiltonian_as_paulis(self.model, self.n_bos, self.paras, self.enc))
        self.h_mat = h_mat
        self.h_op = h_op
        return h_mat, h_op
    
    def set_dimensions(self) -> None:
        """Overwrite in child class"""
        return
    
    def set_initial_state(self) -> None:
        """Overwrite in child class"""
        return

    def simulate_qc(self):
        """Calculate the model"""
        
        # Loop over all time steps
        if self.steps == Steps.LOOP:
            # Set initial state
            qc = self.qc_empty.copy()
            for pos, c in enumerate(self.i_full_bin):
                    if c == '1':
                        qc.x(pos)
            for t_step, t_point in enumerate(self.timesteps):
                print(' Time-step', np.round(t_point, 2))
                # Add Trotter step
                reset_a = t_step > 0 # first timepoint is 0, no evolution
                qc = self.add_trotter_step(qc, t_point, reset_a)
                # State tomography
                if self.qst:
                    qc = self.meas_tomography(qc, self.qubits_system)
                # Measure system qubits
                z_meas = self.meas_system(qc, self.qubits_system, self.n_qubits)
                # Measure spins

                # two-spin-correlations
        else:
            pass

        # calculate observables from measurements

        return 0

    def add_trotter_step(self, qc, t, reset_a=True) -> QuantumCircuit:
        if self.h == H.SCNDORD:
            # same as PTE(Suzuki(order=1)), PTE('suzuki')
            trotterized_op = PauliEvolutionGate(self.h_op, time=t, synthesis=SuzukiTrotter(order=2, reps=1))
            qc.append(trotterized_op, self.qubits_system)
        elif self.h == H.QDRIFT:
            trotterized_op = PauliEvolutionGate(self.h_op, time=t, synthesis=QDrift(reps=1))
            qc.append(trotterized_op, self.qubits_system)
        elif self.h == H.FIRSTORD:
            # trotterized_op = PauliTrotterEvolution(trotter_mode='trotter', reps=1).convert((h*t).exp_i())
            # same as PEG(), PEG(SuzukiTrotter(order=1)), PTE(Suzuki(order=1)), PTE('trotter')
            trotterized_op = PauliEvolutionGate(self.h_op, time=t, synthesis=LieTrotter(reps=1))
            qc.append(trotterized_op, self.qubits_system)
        elif self.h == H.ISODECOMP:
            h_unitary = scipy.linalg.expm(-1j * t * self.h_mat)
            qc.unitary(h_unitary, self.qubits_system, label='H')
        if self.s_a_pairs is not None:
            # see Landi2018_QInfo
            mu = 1 - m.exp(-self.gamma * t)
            theta = m.asin(m.sqrt(mu))
            if self.env == Env.PSWAP:
                pswap_arr = pswap_matrix(theta)
            elif self.env == Env.ADMATRIX:
                ad_arr = ad_matrix(theta)
            elif self.env == 'kraus':
                k1, k2 = kraus0_diluted(self.gamma, t)
            # trotterized_evolution
            for f_a in self.s_a_pairs:
                if reset_a:
                    qc.reset(qubit=f_a[1])
                # trotterized_evolution
                if self.env == Env.PSWAP:
                    qc.unitary(pswap_arr, [f_a[0], f_a[1]], label='pswap')
                elif self.env == Env.ADMATRIX:
                    qc.unitary(ad_arr, [f_a[0], f_a[1]], label='adc')
                elif self.env == 'adc':
                    qc.cry(theta=2*theta, control_qubit=f_a[0], target_qubit=f_a[1])
                    qc.cx(control_qubit=f_a[1], target_qubit=f_a[0])
                elif self.env == 'kraus':
                    qc.unitary(k2, [f_a[0], f_a[1]], label='k2')
                elif self.env == Env.GATEFOLDING:
                    #[('rz', 8), ('sx', 6), ('cx', 2), ('x', 1)
                    # {2 CX, 4 SX, 4 RZ} acting on the spin
                    # {2 CX, 2 SX, 4 RZ, 1 X} acting on the auxiliary
                    # SX, RZ extra on auxiliary while spin idle
                    phi = np.pi
                    qc.barrier(f_a)
                    for _ in range(2):
                        qc.cx(f_a[0], f_a[1])
                        qc.barrier(f_a)
                    for _ in range(5):
                        if _ < 4:
                            qc.rz(phi=phi, qubit=f_a[0])
                        if _ > 0:
                            qc.rz(phi=phi, qubit=f_a[1])
                        qc.barrier(f_a)
                    # x and sx have same gate time
                    for _ in range(4):  # 
                        qc.sx(qubit=f_a[0])
                        if _ > 1:
                            qc.sx(qubit=f_a[1])
                        qc.barrier(f_a)
                    # 1 extra x overall
                    for _ in range(1): # 1
                        qc.x(qubit=f_a[1])
                        qc.barrier(f_a)
        return qc

    def meas_tomography(self, qc, meas_qubits) -> None:
        # Error Mitigation -> use qiskit ignis
        # Lossless tomography
        qc_qst = state_tomography_circuits(qc, meas_qubits)
        job_tomo = execute(qc_qst, AerSimulator(), shots=self.shots)
        tomo_fitter = StateTomographyFitter(job_tomo.result(), qc_qst)
        dm = tomo_fitter.fit(method='lstsq')
        # Noisy tomography
        # generate the calibration circuits
        cal_circuit = QuantumRegister(len(meas_qubits))
        cal_circ_meas, state_labels = complete_meas_cal(qr=cal_circuit)
        # calibration: noise
        cal_noisy = transpile(cal_circ_meas, backend=self.noise_model)
        job_cal_noisy = self.noise_model.run(cal_noisy, shots=self.shots)
        meas_noise_fitter = CompleteMeasFitter(
            job_cal_noisy.result(), state_labels)
        # noisy tomography
        qc_qst_noisy = transpile(qc_qst, backend=self.noise_model)
        job_tomo_noisy = self.noise_model.run(qc_qst_noisy, shots=self.shots)
        # correct data
        correct_tomo_results = meas_noise_fitter.filter.apply(
            job_tomo_noisy.result(), method='least_squares')
        tomo_fitter_em = StateTomographyFitter(correct_tomo_results, qc_qst_noisy)
        dm_em = tomo_fitter_em.fit(method='lstsq')
        # ----------------------------------
        # No Error Mitigation -> use qiskit.experiments
        # qst_data_noisy = StateTomography(qc, measurement_qubits=meas_qubits).run(self.noise_model, shots=self.shots).block_for_results()
        # dm_qiskit_noisy = qst_data_noisy.analysis_results("state").value.data
        # qst_data_ll = StateTomography(qc, measurement_qubits=meas_qubits).run(AerSimulator(),shots=self.shots).block_for_results()
        # dm = qst_data_ll.analysis_results("state").value.data
        # ----------------------------------
        # reorder states
        if self.ordered_keys is not None:
            unordered_keys = sb_sequence(length=len(meas_qubits))
            state_order = [unordered_keys.index(k) for k in self.ordered_keys]
            dm = dm[:, state_order]
            dm = dm[state_order, :]
            dm_em = dm_em[:, state_order]
            dm_em = dm_em[state_order, :]
        self.dm.append(dm)
        self.dm_em.append(dm_em)
        return 0

    def meas_system(self, qc, meas_qubits, n_q):
        backend_ll = AerSimulator()
        n_q_meas = len(meas_qubits)
        qc_m = QuantumCircuit(n_q, n_q_meas)
        qc_m.barrier(meas_qubits)
        qc_m.measure(qubit=meas_qubits, cbit=range(n_q_meas))
        # lhs.compose(rhs)
        qc_e_m = qc_m.compose(qc, qubits=range(n_q), front=True)
        # noiseless
        qc_e_m_ll = transpile(qc_e_m, backend=backend_ll, optimization_level=2)
        job_t = backend_ll.run(qc_e_m_ll, shots=self.shots)  # run_options
        result_t = job_t.result()
        counts_t = result_t.get_counts(qc_e_m_ll)
        # print('measure', counts_t)
        probs, _, quota = self.shots_to_probs(counts_t, n_q_meas)
        self.evo.append(probs)
        self.ps_quota.append(quota)
        _, counts_em, _ = mitigate_error(qc_e_m, n_q_meas, 
                                         self.shots, self.noise_model)
        probs_em, _, quota = self.shots_to_probs(counts_em, n_q_meas)
        self.evo_em.append(probs_em)
        self.ps_quota_em.append(quota)
        return
    
    def shots_to_probs(self, cnts, qubits):
        cnts_select, shots_select = self.get_post_selection(cnts)
        probs = [0 for _ in range(2 ** qubits)]
        labels = ['c' for _ in range(2 ** qubits)]
        if self.enc == Enc.SPLITUNARY:
            probs = [0 for _ in range(len(ordered_keys))]
            labels = ['c' for _ in range(len(ordered_keys))]
        elif self.enc == Enc.FULLUNARY:
            probs = [0 for _ in range(qubits)]
            labels = ['c' for _ in range(qubits)]
        if ordered_keys is None:
            ordered_keys = get_seq(code, length=qubits, rev=False)
        for key in cnts_select:
            if key in ordered_keys:
                pos = ordered_keys.index(key)
                probs[pos] = cnts_select[key] / shots_select
                labels[pos] = key[::-1]
        return probs, labels, shots_select / self.shots
    
    def get_post_selection(self, counts):
        if self.post_select is None:
            return counts, self.shots
        counts_select = {}
        shots_select = 0
        for key in counts:
            if key in self.post_select:
                counts_select[key] = counts[key]
                shots_select += counts[key]
        return counts_select, shots_select
            
    
    def meas_spins(self):
        return 0

    def compare_simulations(self):
        # Fidelity
        return 0