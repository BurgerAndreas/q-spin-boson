import numpy as np
from typing import List, Sequence, Dict, Any # typechecking with mypy
from numpy.typing import NDArray # typechecking with mypy
import matplotlib.pyplot as plt
import math as m
import scipy.linalg # expm
import functools as ft  # XX = ft.reduce(np.kron, [A, B, C, D, E])
from dotenv import load_dotenv  # load environment variables from env file
import os
from pathlib import Path
import time as time # code timing
from datetime import datetime 
import pickle # save/load objects

from qutip import Qobj, Options, mesolve, sigmaz, expect, projection

from qiskit import Aer, QuantumCircuit, QuantumRegister, transpile, IBMQ, execute
from qiskit.visualization import plot_histogram, array_to_latex, plot_gate_map, plot_circuit_layout
from qiskit.opflow import I, X, Y, Z, Zero, One, Plus, Minus, PauliTrotterEvolution, CircuitSampler, PauliOp, PauliSumOp, Suzuki
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.providers.fake_provider import FakeJakarta, FakeToronto
from qiskit.providers.aer import AerSimulator # old qiskit versions
from qiskit.providers.aer.noise import NoiseModel # old qiskit versions
# from qiskit_aer.noise import NoiseModel # new qiskit versions
# from qiskit_aer import AerSimulator # new qiskit versions
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit_experiments.library import StateTomography
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.synthesis import QDrift, LieTrotter, SuzukiTrotter

from settings.types import Enc, Env, H, Model, Steps, Axis
from settings.parameters import Paras
from src.helpers.noise_modeling import modified_noise_model
from src.helpers.hamiltonian_matrix import hamiltonian_as_matrix
from src.helpers.hamiltonian_qiskit import matrix_to_pauli, pauli_to_qiskit, hamiltonian_as_paulis
from src.helpers.environment_interaction import swap_matrix, pswap_matrix, ad_matrix, adc_circuit, kraus0_diluted 
from src.helpers.binary import unary_sequence, sb_sequence, get_seq, fillin_counts
from src.helpers.error_mitigation import mitigate_error
from src.helpers.operators import expct, get_state_label, dm_fidelity

# Set file locations
DIR_FILE = os.path.dirname(os.path.realpath(__file__))
DIR_SAVED_MODELS = os.path.join(DIR_FILE, '../data/saved-models/')
DIR_PLOTS = os.path.join(DIR_FILE, '../data/plots/')
DIR_PLOTS_CIRCUIT = os.path.join(DIR_FILE, '../data/plots-circuits/')
PIC_FILE = 'png'

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
                 noise: float = 1.,
                 initial: NDArray = None):
        """Create a Simulation object.
        To load / simulate the model, use the get_simulation() method.
        Args:
            model (str): Model for system. Spin-Boson or Jaynes-Cummings.
            n_bos (int): Number of bosons in system.
            env (Env): Model for environment.
            h (H): 1st order (trotter), 2nd order (suzuki) product formula 
                or isometric decomposition.
            steps (Steps): Number of steps for trotter or suzuki product formula.
            dt (float): Timestep for trotter or suzuki product formula.
            eta (int): Number of bosons to be measured.
            noise (float): Noise strength for simulation.
            initial (NDArray): initial system state (w/o environment)
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
        self.noise = noise
        self.initial = initial
        # --------------------------------------------------------------------
        # parameters
        self.qst = True
        self.initial_state = None
        self.name = None
        self.opt_prdctfrml: Dict = None # noise -> product formula, timestep
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
        # operators
        self.sz_ops: List[NDArray] = None # z measurement operators
        self.sx_ops: List[NDArray] = None # x measurement operators
        self.sy_ops: List[NDArray] = None # y measurement operators
        self.ada: NDArray = None # boson number operator
        self.l_ops: List[NDArray] = None # lindblad (dissipation) operators
        # --------------------------------------------------------------------
        # containers for simulation results
        # noiseless circuit
        self.evo = []  # evolution of system
        self.sz = []  # z measurement on spin(s)
        self.sx = []  # x measurement on spin(s)
        self.sy = []  # y measurement on spin(s)
        self.s = {Axis.ZAX: self.sz, Axis.XAX: self.sx, Axis.YAX: self.sy}
        self.dm = []  # density matrix
        self.szcorr = []  # connected correlation <sz sz> - <sz><sz>
        self.sxcorr = []  # connected correlation <sx sx> - <sx><sx>
        self.sycorr = []  # connected correlation <sy sy> - <sy><sy>
        self.scorr = {Axis.ZAX: self.szcorr, Axis.XAX: self.sxcorr, 
                      Axis.YAX: self.sycorr}
        self.bosons = []  # boson occupation number
        # error em circuit = noise model + measurement mitigation
        self.evo_em = []  # evolution of system
        self.sz_em = []  # z measurement
        self.sx_em = []  # x measurement
        self.sy_em = []  # y measurement
        self.s_em = {Axis.ZAX: self.sz_em, Axis.XAX: self.sx_em, 
                     Axis.YAX: self.sy_em}
        self.dm_em = []  # density matrix
        self.szcorr_em = []  # connected correlation
        self.sxcorr_em = []  # connected correlation
        self.sycorr_em = []  # connected correlation
        self.scorr_em = {Axis.ZAX: self.szcorr_em, Axis.XAX: self.sxcorr_em,
                        Axis.YAX: self.sycorr_em}
        self.bosons_em = []  # boson occupation number
        # exact reference = qutip linblad master equation solver
        self.evo_exact = []  # evolution of system
        self.sz_exact = []  # z measurement
        self.sx_exact = []  # x measurement
        self.sy_exact = []  # y measurement
        self.s_exact = {Axis.ZAX: self.sz_exact, Axis.XAX: self.sx_exact, 
                        Axis.YAX: self.sy_exact}
        self.dm_exact = []  # density matrix
        self.szcorr_exact = []  # connected correlation
        self.sxcorr_exact = []  # connected correlation
        self.sycorr_exact = []  # connected correlation
        self.scorr_exact = {Axis.ZAX: self.szcorr_exact, 
                            Axis.XAX: self.sxcorr_exact,
                            Axis.YAX: self.sycorr_exact}
        self.bosons_exact = []  # boson occupation number
        # infidelity 
        self.infidelity = []  # exact - noiseless circuit
        self.infidelity_em = []  # exact - error em circuit
        # absolute error
        self.ae = []  # exact - noiseless circuit
        self.ae_em = []  # exact - error em circuit
        # mean absolute error
        self.mae = []  # exact - noiseless circuit
        self.mae_em = []  # exact - error em circuit
        # mean squared error
        self.mse = []  # exact - noiseless circuit
        self.mse_em = []  # exact - error em circuit
        # post selection quota
        self.ps_quota = [] 
        self.ps_quota_em = [] 
        # state labels
        self.labels = []
        # --------------------------------------------------------------------
        self.load_status: int = 0
        
    def __repr__(self):
        # dir(MyClass)
        return f"Simulation({self.name}). Methods: {self.__dir__()}"
    
    def __str__(self):
        return f"Simulation({self.name})"
    
    def get_simulation(self):
        self.fix_parameters()
        self.update_name()
        self.load_status = self.load()
        if self.load_status != 200:
            self.simulate_model()
        return self

    def fix_parameters(self):
        if self.enc in [Enc.SPLITUNARY, Enc.FULLUNARY]:
            self.qst = False
            if self.model in [Model.JC2S, Model.JC3S]:
                raise ValueError('Unary encodings not supported for JC.')
        if self.h in [H.NOH, H.ISODECOMP]:
            self.enc = Enc.BINARY
        if self.env == Env.NOENV or self.gamma == 0.:
            self.gamma = 0.
            self.env = Env.NOENV
        return
    
    def set_optimal_product_formula(self) -> None:
        """Set optimal product formula and dt for Trotterization."""
        if self.noise in self.opt_prdctfrml:
            self.h, self. dt = self.opt_prdctfrml[self.noise]
        else:
            self.h, self. dt = self.opt_prdctfrml[.1]
        return

    def update_name(self):
        # Model
        name = f'{self.model.name}'
        if self.env:
            name += f'_{self.env.name}' # JC does not have environment
        if self.n_bos:
            name += f'_b{self.n_bos}' # TwoLevel does not have bosons
        # Model parameters
        if self.paras:
            name += f'_p{self.paras.name}' # TwoLevel has no parameters
        if self.gamma:
            name += f'_g{self.gamma:.3f}' # JC does not have environment
        # Simulation parameters
        if self.h:
            name += f'_{self.h.value}' # TwoLevel has no Hamiltonian
        if self.steps == Steps.LOOP:
            name += f'_s{self.steps.value}_d{self.dt:.3f}'
        else:
            name += f'_s{self.steps.value}_n{self.eta}'
        name += f'_e{self.noise:.3f}'
        name += f"_{self.backend.backend_name.replace('_', '')}"
        # Custom initial state
        if self.initial:
            name += f"_i{''.join(str(i) for i in self.initial)}"
        # Remove all dots in floats
        self.name = name.lower().replace('.', '')
        return name

    def load(self):
        # Check if saved model exists
        path = f'{DIR_SAVED_MODELS}{self.name}.pickle'
        if Path(path).is_file():
            # load model
            self.__dict__.update(pickle.load(
                open(path, mode='rb')
                ).__dict__)
            return 200
        # Saved model does not exist
        return 404

    def save(self):
        # save model
        with open(f'{DIR_SAVED_MODELS}{self.name}.pickle', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return 201

    def circuit_example(
            self, 
            backend = AerSimulator(), 
            initial: bool = False,
            transpile: bool = False) -> QuantumCircuit:
        """Circuit for one trotter step.
        Time t = 1.
        """
        # Circuit for one trotter step
        qc = self.qc_empty.copy()
        if initial:
            for pos, c in enumerate(self.i_full_binary):
                    if c == '1':
                        qc.x(pos)
        qc = self.add_trotter_step(qc, t=1, reset_a=True)
        if transpile:
            # transpile onto device gates and layout
            qc = transpile(qc, backend=backend, optimization_level=3, 
                        basis_gates=['cx', 'id', 'x', 'sx', 'rz', 'reset'])
        return qc
    
    def get_gates(
            self, 
            backend = AerSimulator(),
            initial: bool = False) -> tuple[int, dict]:
        """Gates necessary for one trotter step."""
        qc = self.circuit_example(backend, initial)
        return qc.depth(), qc.count_ops()

    def save_layout(
            self, 
            backend,
            initial: bool = False) -> None:
        """Qubit connectivity and circuit qubit mapping onto device."""
        # device gate map
        fig = plot_gate_map(backend, 
            filename=f'{DIR_PLOTS_CIRCUIT}{self.name}_gatemap.{PIC_FILE}')
        # fig.tight_layout()
        # fig.show()
        # qubit mapping onto device
        qc = self.circuit_example(backend, initial)
        fig = plot_circuit_layout(qc, backend)
        fig.savefig(fname=f'{DIR_PLOTS_CIRCUIT}{self.name}_layout.{PIC_FILE}')
        # fig.show()
        return 

    def save_circuit_latex(
            self, 
            backend = AerSimulator(),
            initial: bool = False) -> str:
        """Circuit for one trotter step as latex."""
        qc = self.circuit_example(backend, initial)
        latex_source = qc.draw('latex_source', fold=20)
        for number in range(qc.num_qubits):
            look_for = "<<<{" + str(number) + "}"
            latex_source = latex_source.replace(look_for,"<<<{}")
        with open(f'{DIR_PLOTS_CIRCUIT}{self.name}.tex', 'w') as f:
            f.write(latex_source)
        return latex_source

    def save_circuit_image(
            self, 
            backend = AerSimulator(), 
            initial: bool = False) -> None:
        """Circuit for one trotter step as an image."""
        qc = self.circuit_example(backend, initial)
        fig = qc.draw('mpl', 
                      filename=f'{DIR_PLOTS_CIRCUIT}{self.name}.{PIC_FILE}', 
                      style="bw") #  style="bw", "iqx"
        fig.show()
        return
    
    def update_s(self) -> None:
        """Update spin, spin correlation dictionaries."""
        self.s = {Axis.ZAX: self.sz, Axis.XAX: self.sx, Axis.YAX: self.sy}
        self.s_em = {Axis.ZAX: self.sz_em, Axis.XAX: self.sx_em, 
                     Axis.YAX: self.sy_em}
        self.s_exact = {Axis.ZAX: self.sz_exact, Axis.XAX: self.sx_exact, 
                        Axis.YAX: self.sy_exact}
        # spin correlation
        self.scorr = {Axis.ZAX: self.szcorr, Axis.XAX: self.sxcorr, 
                      Axis.YAX: self.sycorr} 
        self.scorr_em = {Axis.ZAX: self.szcorr_em, Axis.XAX: self.sxcorr_em, 
                         Axis.YAX: self.sycorr_em}
        self.scorr_exact = {Axis.ZAX: self.szcorr_exact, 
                            Axis.XAX: self.sxcorr_exact, 
                            Axis.YAX: self.sycorr_exact}
        return
    
    def simulate_model(self) -> None:
        print('-'*40)
        print(f'Simulating model: {self.name}')
        t = time.time()
        self.build_operators()
        self.set_labels()
        self.simulate_qc()
        self.simulate_exact_linblad()
        self.update_s()
        self.compare_simulations()
        self.load_status = self.save()
        print('infidelity   :', np.round(self.infidelity, 3))
        print('infidelity em:', np.round(self.infidelity_em, 3))
        print('Simulation time:', 
              f'{(time.time() - t)/60:.2f}min', 
              f'({time.time() - t:.2f}s)')
        print('-'*40)
        return
    
    def build_operators(self) -> None:
        """Build spin and boson number operators.
        Used for exact reference and labels.
        Overwrite in child class.
        """
        return
    
    def set_labels(self) -> None:
        """Set labels of states for plots."""
        labels = []
        for d in range(self.d_system):
            basis_dm = np.zeros([self.d_system, self.d_system])
            basis_dm[d][d] = 1
            labels.append(get_state_label(basis_dm, self.sz_ops, self.ada))
        self.labels = labels
        return

    def simulate_exact_linblad(self) -> None:
        if self.h_mat is None:
            # TwoLevel has no Hamiltonian
            h_obj = Qobj(np.zeros([self.d_system, self.d_system]))
        else:
            h_obj = Qobj(self.h_mat)
        i_system = np.reshape(self.i_system, [np.size(self.i_system), 1])
        rho0 = Qobj(i_system @ i_system.T)
        if self.env != Env.NOENV and self.l_ops:
            cops = [Qobj(l) for l in self.l_ops]
        else:
            cops = []
        dm_qutip = mesolve(H=h_obj, rho0=rho0, tlist=self.timesteps, c_ops=cops)
        # evolution
        evo_exact_transpose = [
            np.real(expect(dm_qutip.states, projection(self.d_system, st, st))) 
            for st in range(self.d_system)]
        # self.evo_exact = np.transpose(evo_exact_transpose)
        self.evo_exact = [list(x) for x in zip(*evo_exact_transpose)]
        # density matrix
        self.dm_exact = [dm_qutip.states[t_step].data.toarray() 
                         for t_step in range(len(self.timesteps))]
        # calculate observables from density matrix
        for dm_t in self.dm_exact:
            self.sz_exact.append([expct(_sz, dm_t) for _sz in self.sz_ops])
            self.sx_exact.append([expct(_sx, dm_t) for _sx in self.sx_ops])
            self.sy_exact.append([expct(_sy, dm_t) for _sy in self.sy_ops])
            if self.n_bos:
                self.bosons_exact.append(expct(self.ada, dm_t))
        # connected correlation
        if len(self.spins) == 2:
            for dm_t in self.dm_exact:
                for _corr, _ops in zip(
                    [self.szcorr_exact, self.sxcorr_exact, self.sycorr_exact],
                    [self.sz_ops, self.sx_ops, self.sy_ops]):
                    _corr.append(
                        expct(dm_t, _ops[0] @ _ops[1]) 
                        - (expct(dm_t, _ops[0]) * expct(dm_t, _ops[1])))
        return 
    
    def set_default_simulation_parameters(self) -> None:
        """Set initial state, hamiltonian, noise model, backend, etc.
        Called in __init__ of child classes. 
        Parameters can be overwritten by user.
        """
        self.fix_parameters()
        self.update_name()
        # Get Noise Model
        if self.noise == 1.:
            self.noise_model = AerSimulator.from_backend(self.backend)
        else:
            self.noise_model = AerSimulator(
                noise_model=modified_noise_model(self.backend, self.noise))
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
            # trotterized_op = PauliTrotterEvolution(
            #     trotter_mode='trotter', 
            #     reps=1
            #     ).convert(evolution_op)
            # qc_e = trotterized_op.to_circuit()
        else:
            h_op = eval(hamiltonian_as_paulis(self.model, self.n_bos, 
                                              self.paras, self.enc))
        self.h_mat = h_mat
        self.h_op = h_op
        return h_mat, h_op
    
    def set_dimensions(self) -> None:
        """Overwrite in child class"""
        return
    
    def set_initial_state(self) -> None:
        """Overwrite in child class"""
        return

    def simulate_qc(self) -> None:
        """Calculate the model"""
        # Loop over all time steps
        if self.steps == Steps.LOOP:
            qc = self.qc_empty.copy()
            # Set initial state
            for pos, c in enumerate(self.i_full_binary):
                    if c == '1':
                        qc.x(pos)
            for t_step, t_point in enumerate(self.timesteps):
                print(f' Time-step {t_point:.2f}')
                # Add Trotter step
                if t_step > 0:
                    # first timepoint is 0, no evolution
                    qc = self.add_trotter_step(qc, self.dt, reset_a=True)
                # Measure
                self.meas_circuit(qc, self.qubits_system)
        elif self.steps == Steps.NFIXED:
            for t_step, t_point in enumerate(self.timesteps):
                print(f' Time-step {t_point:.2f}')
                # Set initial state
                qc = self.qc_empty.copy()
                for pos, c in enumerate(self.i_full_binary):
                        if c == '1':
                            qc.x(pos)
                # Add Trotter steps
                if t_step > 0:
                    for eta_step in range(self.eta):
                        reset_a = eta_step > 1
                        qc = self.add_trotter_step(qc, t_point/self.eta, reset_a)
                # Measure
                self.meas_circuit(qc, self.qubits_system)
        return
    
    def meas_circuit(self, qc, qubits) -> None:
        """Measure qubits in qc"""
        # State tomography
        if self.qst:
            self.meas_tomography(qc, qubits)
        # Measure system qubits
        self.meas_system(qc, qubits)
        # Measure spins and bosons for observables
        self.meas_spins_bosons(qc)
        return 

    def add_trotter_step(self, qc, t, reset_a=True) -> QuantumCircuit:
        if self.h == H.SCNDORD:
            # same as PTE(Suzuki(order=1)), PTE('suzuki')
            trotterized_op = PauliEvolutionGate(self.h_op, time=t, synthesis=SuzukiTrotter(order=2, reps=1))
            qc.append(trotterized_op, self.qubits_system)
        elif self.h == H.QDRIFT:
            trotterized_op = PauliEvolutionGate(self.h_op, time=t, synthesis=QDrift(reps=1))
            qc.append(trotterized_op, self.qubits_system)
        elif self.h == H.FRSTORD:
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
            elif self.env == Env.KRAUS:
                k1, k2 = kraus0_diluted(self.gamma, t)
            # trotterized_evolution
            for s_a in self.s_a_pairs:
                if reset_a:
                    qc.reset(qubit=s_a[1])
                # trotterized_evolution
                if self.env == Env.PSWAP:
                    qc.unitary(pswap_arr, [s_a[0], s_a[1]], label='pswap')
                elif self.env == Env.ADMATRIX:
                    qc.unitary(ad_arr, [s_a[0], s_a[1]], label='adc')
                elif self.env == Env.ADC:
                    qc.cry(theta=2*theta, control_qubit=s_a[0], target_qubit=s_a[1])
                    qc.cx(control_qubit=s_a[1], target_qubit=s_a[0])
                elif self.env == Env.KRAUS:
                    qc.unitary(k2, [s_a[0], s_a[1]], label='k2')
                elif self.env == Env.GATEFOLDING:
                    #[('rz', 8), ('sx', 6), ('cx', 2), ('x', 1)
                    # {2 CX, 4 SX, 4 RZ} acting on the spin
                    # {2 CX, 2 SX, 4 RZ, 1 X} acting on the auxiliary
                    # SX, RZ extra on auxiliary while spin idle
                    phi = np.pi
                    qc.barrier(s_a)
                    for _ in range(2):
                        qc.cx(s_a[0], s_a[1])
                        qc.barrier(s_a)
                    for _ in range(5):
                        if _ < 4:
                            qc.rz(phi=phi, qubit=s_a[0])
                        if _ > 0:
                            qc.rz(phi=phi, qubit=s_a[1])
                        qc.barrier(s_a)
                    # x and sx have same gate time
                    for _ in range(4):  # 
                        qc.sx(qubit=s_a[0])
                        if _ > 1:
                            qc.sx(qubit=s_a[1])
                        qc.barrier(s_a)
                    # 1 extra x overall
                    for _ in range(1): # 1
                        qc.x(qubit=s_a[1])
                        qc.barrier(s_a)
                elif self.env != Env.NOENV:
                    raise ValueError(f'Unknown environment {self.env}')
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

    def meas_system(self, qc: QuantumCircuit, meas_qubits: List[int]) -> None:
        """Z-basis measurement of the system qubits."""
        qnum = qc.num_qubits
        backend_ll = AerSimulator()
        n_q_meas = len(meas_qubits)
        qc_m = QuantumCircuit(qnum, n_q_meas)
        qc_m.barrier(meas_qubits)
        qc_m.measure(qubit=meas_qubits, cbit=range(n_q_meas))
        qc_e_m = qc_m.compose(qc, qubits=range(qnum), front=True)
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
        ordered_keys = self.ordered_keys
        if ordered_keys is None:
            ordered_keys = get_seq(self.enc, length=qubits, rev=False)
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
            
    def meas_spins_bosons(self, qc: QuantumCircuit) -> None:
        qnum: int = qc.num_qubits
        # ----------------------------------
        # Bosons from previous measurements
        if len(self.spins) == 1 and self.n_bos:
            # Model.SB1S, Model.SB1SJC, Model.SB1SPZ
            for probs, _occ in zip([self.evo[-1], self.evo_em[-1]], 
                                   [self.bosons, self.bosons_em]):
                _occ.append(sum((probs[_b] + probs[_b+self.n_bos])*_b 
                                for _b in range(1, self.n_bos)))
        elif len(self.spins) == 2 and self.n_bos:
            # Model.SB2S, Model.JC2S
            # Bosons from previous measurement
            for probs, _occ in zip([self.evo[-1], self.evo_em[-1]],
                                    [self.bosons, self.bosons_em]):
                _occ.append(sum(
                    (
                        probs[_b * 2]
                        + probs[_b * 2 + 1]
                        + probs[(_b + self.n_bos) * 2]
                        + probs[(_b + self.n_bos) * 2 + 1]
                    )
                    * _b
                    for _b in range(1, self.n_bos)
                ))
        # ----------------------------------
        # Z-basis
        if len(self.spins) == 1:
            # Z-basis from previous measurements
            for probs, _sz in zip([self.evo[-1], self.evo_em[-1]], 
                                  [self.sz, self.sz_em]):
                _sz.append(- np.sum(probs[:self.n_bos]) 
                           + np.sum(probs[self.n_bos:]))
        elif len(self.spins) == 2:
            # Z-basis
            qc_z = qc.copy()
            qc_m = QuantumCircuit(qnum, len(self.spins))
            qc_m.barrier()
            qc_m.measure(qubit=self.spins, cbit=range(len(self.spins)))
            qc_e_m = qc_m.compose(qc_z, qubits=range(qnum), front=True)
            # lossless
            backend_ll = AerSimulator()
            qc_e_m_ll = transpile(qc_e_m, backend=backend_ll, 
                                  optimization_level=2)
            job_ll = backend_ll.run(qc_e_m_ll, shots=self.shots)  # run_options
            result_ll = job_ll.result()
            counts_ll = result_ll.get_counts(qc_e_m_ll)
            # counts_ll, shots_ll = self.get_post_selection(counts_ll)
            shots_ll = self.shots
            counts_ll = fillin_counts(counts_ll)
            # error mitigated
            _, counts_em, _ = mitigate_error(qc_e_m, len(self.spins), 
                                             self.shots, self.noise_model)
            # counts_em, shots_em = self.get_post_selection(counts_em)
            shots_em = self.shots
            counts_em = fillin_counts(counts_em)
            for _cnts, _shots, _sz in zip([counts_ll, counts_em], 
                                          [shots_ll, shots_em],
                                          [self.sz, self.sz_em]):
                _sz.append([
                    -1*(-(_cnts['00'] + _cnts['10']) 
                        + (_cnts['01'] + _cnts['11'])) / _shots,                
                    -1*(-(_cnts['00'] + _cnts['01']) 
                        + (_cnts['10'] + _cnts['11'])) / _shots])
            # Z-basis spin correlation
            for _cnts, _shots, _szc in zip([counts_ll, counts_em], 
                                          [shots_ll, shots_em],
                                          [self.szcorr, self.szcorr_em]):
                # <Sz Sz> = P[00] + P[11] - P[01] - P[10]
                # <Sz><Sz> = {-P[0] + P[1]}_a * {-P[0] + P[1]}_b
                #          = {-P[00] - P[01] + P[10] + P[11]}_a 
                #           * {-P[00] - P[10] + P[01] + P[11]}_b
                _corr = np.real(((_cnts['00'] + _cnts['11']) 
                                 - (_cnts['01'] + _cnts['10'])) / _shots)  
                _avg_0 = np.real((-_cnts['00'] - _cnts['01'] 
                                  + _cnts['10'] + _cnts['11']) / _shots)
                _avg_1 = np.real((-_cnts['00'] - _cnts['10'] 
                                  + _cnts['01'] + _cnts['11']) /_shots)
                _szc.append(_corr - (_avg_0 * _avg_1))
        # ----------------------------------
        # X-basis measurement, spin correlation
        qc_x = qc.copy()
        qc_x.h(self.spins)
        qc_m = QuantumCircuit(qnum, len(self.spins))
        qc_m.barrier()
        qc_m.measure(qubit=self.spins, cbit=range(len(self.spins)))
        qc_e_m = qc_m.compose(qc_x, qubits=range(qnum), front=True)
        # lossless
        backend_ll = AerSimulator()
        qc_e_m_ll = transpile(qc_e_m, backend=backend_ll, optimization_level=2)
        job_ll = backend_ll.run(qc_e_m_ll, shots=self.shots)  # run_options
        result_ll = job_ll.result()
        counts_ll = result_ll.get_counts(qc_e_m_ll)
        # counts_ll, shots_ll = self.get_post_selection(counts_ll)
        shots_ll = self.shots
        counts_ll = fillin_counts(counts_ll)
        # error mitigated
        _, counts_em, _ = mitigate_error(qc_e_m, len(self.spins), 
                                         self.shots, self.noise_model)
        # counts_em, shots_em = self.get_post_selection(counts_em)
        shots_em = self.shots
        counts_em = fillin_counts(counts_em)
        if len(self.spins) == 1:
            for _cnts, _shots, _sx in zip([counts_ll, counts_em], 
                                          [shots_ll, shots_em],
                                          [self.sx, self.sx_em]):
                _sx.append(-1*(-_cnts['0'] + _cnts['1']) / _shots)
        elif len(self.spins) == 2:
            for _cnts, _shots, _sx in zip([counts_ll, counts_em], 
                                          [shots_ll, shots_em],
                                          [self.sx, self.sx_em]):
                _sx.append([
                    -1*(-(_cnts['00'] + _cnts['10']) 
                        + (_cnts['01'] + _cnts['11'])) / _shots,                
                    -1*(-(_cnts['00'] + _cnts['01']) 
                        + (_cnts['10'] + _cnts['11'])) /_shots])
            for _cnts, _shots, _sxc in zip([counts_ll, counts_em], 
                                          [shots_ll, shots_em],
                                          [self.sxcorr, self.sxcorr_em]):
                _corr = np.real(((_cnts['00'] + _cnts['11']) 
                                 - (_cnts['01'] + _cnts['10'])) / _shots)  
                _avg_0 = np.real((-_cnts['00'] - _cnts['01'] 
                                  + _cnts['10'] + _cnts['11']) / _shots)
                _avg_1 = np.real((-_cnts['00'] - _cnts['10'] 
                                  + _cnts['01'] + _cnts['11']) / _shots)
                _sxc.append(_corr - (_avg_0 * _avg_1))
        # ----------------------------------
        # Y-basis measurement, spin correlation
        qc_y = qc.copy()
        qc_y.sdg(self.spins) # S^dagger, only difference from X-basis
        qc_y.h(self.spins)
        qc_m = QuantumCircuit(qnum, len(self.spins))
        qc_m.barrier()
        qc_m.measure(qubit=self.spins, cbit=range(len(self.spins)))
        qc_e_m = qc_m.compose(qc_y, qubits=range(qnum), front=True)
        # lossless
        backend_ll = AerSimulator()
        qc_e_m_ll = transpile(qc_e_m, backend=backend_ll, optimization_level=2)
        job_ll = backend_ll.run(qc_e_m_ll, shots=self.shots)  # run_options
        result_ll = job_ll.result()
        counts_ll = result_ll.get_counts(qc_e_m_ll)
        # counts_ll, shots_ll = self.get_post_selection(counts_ll)
        shots_ll = self.shots
        counts_ll = fillin_counts(counts_ll)
        # error mitigated
        _, counts_em, _ = mitigate_error(qc_e_m, len(self.spins), self.shots, 
                                         self.noise_model)
        # counts_em, shots_em = self.get_post_selection(counts_em)
        shots_em = self.shots
        counts_em = fillin_counts(counts_em)
        if len(self.spins) == 1:
            for _cnts, _shots, _sy in zip([counts_ll, counts_em], 
                                          [shots_ll, shots_em], 
                                          [self.sy, self.sy_em]):
                _sy.append(-1*(-_cnts['0'] + _cnts['1']) / _shots)
        elif len(self.spins) == 2:
            for _cnts, _shots, _sy in zip([counts_ll, counts_em], 
                                          [shots_ll, shots_em], 
                                          [self.sy, self.sy_em]):
                _sy.append([
                    -1*(-(_cnts['00'] + _cnts['10']) 
                        + (_cnts['01'] + _cnts['11'])) / _shots,                
                    -1*(-(_cnts['00'] + _cnts['01']) 
                        + (_cnts['10'] + _cnts['11'])) / _shots])
            for _cnts, _shots, _syc in zip([counts_ll, counts_em], 
                                           [shots_ll, shots_em], 
                                           [self.sycorr, self.sycorr_em]):
                _corr = np.real(((_cnts['00'] + _cnts['11']) 
                                 - (_cnts['01'] + _cnts['10'])) / _shots)  
                _avg_0 = np.real((-_cnts['00'] - _cnts['01'] 
                                  + _cnts['10'] + _cnts['11']) / _shots)
                _avg_1 = np.real((-_cnts['00'] - _cnts['10'] 
                                  + _cnts['01'] + _cnts['11']) / _shots)
                _syc.append(_corr - (_avg_0 * _avg_1))
        
        return 
    
    def compare_simulations(self) -> None:
        """Fidelity"""
        if self.qst:
            for t_step in range(len(self.timesteps)):
                self.infidelity.append(
                    np.real(dm_fidelity(
                    true=self.dm_exact[t_step], 
                    approx=self.dm[t_step], 
                    inv=True)))
                self.infidelity_em.append(
                    np.real(dm_fidelity(
                    true=self.dm_exact[t_step], 
                    approx=self.dm_em[t_step], 
                    inv=True)))
        # absolute error
        self.ae = np.abs(np.array(self.evo) - np.array(self.evo_exact))
        self.ae_em = np.abs(np.array(self.evo_em) - np.array(self.evo_exact))
        # mean squared error (MSE)
        # each point is the MSE of all states at a given time step
        self.mse = np.mean(np.square(self.ae), axis=1)
        self.mse_em = np.mean(np.square(self.ae_em), axis=1)
        # mean absolute error (MAE)
        self.mae = np.mean(self.ae, axis=1)
        self.mae_em = np.mean(self.ae_em, axis=1)
        return 
    
    def check_results(self) -> None:
        # Check if there are results
        print('timesteps', np.shape(self.timesteps))
        print('labels', np.shape(self.labels))
        print('evo', np.shape(self.evo))
        print('evo exact', np.shape(self.evo_exact))
        print('dm', np.shape(self.dm))
        print('dm exact', np.shape(self.dm_exact))
        print('sz', np.shape(self.sz))
        print('sx', np.shape(self.sx))
        print('sy', np.shape(self.sy))
        print('szcorr', np.shape(self.szcorr))
        print('sxcorr', np.shape(self.sxcorr))
        print('sycorr', np.shape(self.sycorr))
        print('bosons', np.shape(self.bosons))
        print('infidelity', np.shape(self.infidelity))
        print('infidelity_em', np.shape(self.infidelity_em))
        return 
    
    def print_results(self) -> None:
        # Print results
        print('timesteps\n', self.timesteps)
        print('evo\n', self.evo)
        print('dm\n', self.dm)
        print('sz\n', self.sz)
        print('sx\n', self.sx)
        print('sy\n', self.sy)
        print('szcorr\n', self.szcorr)
        print('sxcorr\n', self.sxcorr)
        print('sycorr\n', self.sycorr)
        print('bosons\n', self.bosons)
        print('infidelity\n', self.infidelity)
        print('infidelity_em\n', self.infidelity_em)
        return