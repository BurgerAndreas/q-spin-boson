import numpy as np
import matplotlib.pyplot as plt
from typing import List, Sequence, Any # typechecking with mypy
from numpy.typing import NDArray # typechecking with mypy

from settings.types import Model, Env, H, Axis
from src.model_base import Simulation
from src.model_general import simulation

DTS_DEFAULT = [0.1, 0.2, 0.3, 0.4, 0.5]
NOISES_DEFAULT = [0.01, 0.1, 1.]
GAMMAS_DEFAULT = [0., 0.5, 1., 1.5, 2., 2.5]

def plot_ifid_vs_dt_env(model: Model = Model.SB1S, dts: List[float] = None):
    """
    Time-averaged infidelity over the timestep size dt in noiseless simulations.
    Compares w/o environment (Hamiltonian only).
    Fig 4 in the paper.
    """
    if dts is None:
        dts = DTS_DEFAULT
    fig, ax = plt.subplots()
    ifid_avg_h_fo = [] # first order
    ifid_avg_env_fo = []
    ifid_avg_h_so = [] # second order
    ifid_avg_env_so = []
    for dt in dts:
        # Hamiltonian only
        sim = simulation(model=model, env=Env.NOENV, h=H.FRSTORD, dt=dt)
        sim.get_simulation()
        ifid_avg_h_fo.append(np.sum(sim.infidelity) / 
                             np.shape(sim.infidelity)[0])
        sim = simulation(model=model, env=Env.NOENV, h=H.SCNDORD, dt=dt)
        sim.get_simulation()
        ifid_avg_h_so.append(np.sum(sim.infidelity) / 
                             np.shape(sim.infidelity)[0])
        # Environment
        sim = simulation(model=model, env=Env.ADC, h=H.FRSTORD, dt=dt)
        sim.get_simulation()
        ifid_avg_env_fo.append(np.sum(sim.infidelity) / 
                               np.shape(sim.infidelity)[0])
        sim = simulation(model=model, env=Env.ADC, h=H.SCNDORD, dt=dt)
        sim.get_simulation()
        ifid_avg_env_so.append(np.sum(sim.infidelity) / 
                               np.shape(sim.infidelity)[0])
    ax.plot(dts, ifid_avg_h_fo, label='Hamiltonian only (first order)')
    ax.plot(dts, ifid_avg_env_fo, label='Hamiltonian + environment (first order)')
    ax.plot(dts, ifid_avg_h_so, label='Hamiltonian only (second order)')
    ax.plot(dts, ifid_avg_env_so, label='Hamiltonian + environment (second order)')
    ax.set_xlabel(r'Timestep size dt')
    ax.set_ylabel(r'Time-averaged infidelity')
    # ax.set_title(r'Time-averaged infidelity over the timestep size dt in noiseless simulations')
    ax.legend()
    return fig


def plot_ifid_vs_dt_noises(model: Model = Model.SB1S, 
                           env: Env = Env.ADC, 
                           dts: List[float] = None, 
                           noises: List[float] = None):
    """
    Time-averaged infidelity over the timestep size dt at different noise levels.
    Compares first and second order under noise.
    Fig 5(a) in the paper.
    """
    if dts is None:
        dts = DTS_DEFAULT
    if noises is None:
        noises = NOISES_DEFAULT
    fig, ax = plt.subplots()
    ifid_avg_fo = [] # first order
    ifid_avg_so = [] # second order
    for dt in dts:
        for noise in noises:
            sim = simulation(model=model, env=env, h=H.FRSTORD, dt=dt, 
                             noise=noise)
            sim.get_simulation()
            ifid_avg_fo.append(np.sum(sim.infidelity_em) / 
                               np.shape(sim.infidelity_em)[0])
            sim = simulation(model=model, env=env, h=H.SCNDORD, dt=dt, 
                             noise=noise)
            sim.get_simulation()
            ifid_avg_so.append(np.sum(sim.infidelity_em) / 
                               np.shape(sim.infidelity_em)[0])
    for n_noise, noise in enumerate(noises):
        ax.plot(dts, ifid_avg_fo[n_noise::len(noises)], 
                label=f'First order, noise={noise}')
        ax.plot(dts, ifid_avg_so[n_noise::len(noises)], 
                label=f'Second order, noise={noise}')
    ax.set_xlabel(r'Timestep size dt')
    ax.set_ylabel(r'Time-averaged infidelity')
    ax.set_title(r'Time-averaged infidelity over the timestep size dt at different noise levels')
    return fig


def plot_ifid_vs_noise(model: Model = Model.SB1S, 
                        env: Env = Env.ADC,
                        h: H = H.SCNDORD, 
                        dt: float = 0.2,
                        noises: List[float] = None):
    """
    Time-averaged and final infidelity over the noise level.
    Scaling of infidelity with noise.
    Compares time-averaged and final infidelity.
    Fig 5(b) in the paper.
    """
    if noises is None:
        noises = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.]
    fig, ax = plt.subplots()
    ifid_avg = []
    ifid_final = []
    for noise in noises:
        sim = simulation(model=model, env=env, h=h, dt=dt, 
                         noise=noise)
        sim.get_simulation()
        ifid_avg.append(np.sum(sim.infidelity_em) / 
                        np.shape(sim.infidelity_em)[0])
        ifid_final.append(sim.infidelity_em[-1])
    ax.plot(noises, ifid_avg, label='Time-averaged infidelity')
    ax.plot(noises, ifid_final, label='Final infidelity')
    ax.set_xlabel(r'Noise level')
    ax.set_ylabel(r'Infidelity')
    ax.set_title(r'Time-averaged and final infidelity over the noise level')
    return fig


def plot_ifid_vs_time(model: Model = Model.SB1S, 
                        env: Env = Env.ADC, 
                        dt: float = 0.2,
                        noises: List[float] = None):
    """
    Infidelity over time at different noise levels.
    Compares first and second order under noise.
    Fig 6 in the paper.
    """
    if noises is None:
        noises = NOISES_DEFAULT
    fig, ax = plt.subplots()
    ifid_fo = [] # first order
    ifid_so = [] # second order
    for noise in noises:
        sim = simulation(model=model, env=env, h=H.FRSTORD, dt=dt, 
                         noise=noise)
        sim.get_simulation()
        ifid_fo.append(sim.infidelity)
        sim = simulation(model=model, env=env, h=H.SCNDORD, dt=dt, 
                         noise=noise)
        sim.get_simulation()
        ifid_so.append(sim.infidelity)
    for n_noise, noise in enumerate(noises):
        ax.plot(ifid_fo[n_noise], label=f'First order, noise={noise}')
        ax.plot(ifid_so[n_noise], label=f'Second order, noise={noise}')
    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Infidelity')
    ax.set_title(r'Infidelity over time at different noise levels')
    return fig


def plot_ifid_vs_gamma(model: Model = Model.SB1S, 
                        env: Env = Env.ADC, 
                        dt: float = 0.2,
                        h: H = H.SCNDORD,
                        noise: float = 0.01,
                        gammas: List[float] = None):
    """
    Time-averaged infidelity over dissipative rate w/o noise.
    Fig 7(a) in the paper.
    """ 
    if gammas is None:
        gammas = GAMMAS_DEFAULT
    fig, ax = plt.subplots()
    ifid_avg = []
    ifid_avg_noise = []
    for gamma in gammas:
        sim = simulation(model=model, env=env, h=h, dt=dt, 
                         noise=noise, gamma=gamma)
        sim.get_simulation()
        ifid_avg.append(np.sum(sim.infidelity) / 
                        np.shape(sim.infidelity)[0])
        ifid_avg_noise.append(np.sum(sim.infidelity_em) / 
                              np.shape(sim.infidelity_em)[0])
    ax.plot(gammas, ifid_avg, label='Time-averaged infidelity')
    ax.plot(gammas, ifid_avg_noise, label='Time-averaged infidelity with noise')
    ax.set_xlabel(r'Dissipative rate $\gamma$')
    ax.set_ylabel(r'Time-averaged infidelity')
    ax.set_title(r'Time-averaged infidelity over dissipative rate w/o noise')
    return fig


def plot_ifid_vs_time_gammas(model: Model = Model.SB1S, 
                        env: Env = Env.ADC, 
                        dt: float = 0.2,
                        h: H = H.SCNDORD,
                        noise: float = 0.01,
                        gammas: List[float] = None):
    """
    Infidelity over time at different dissipative rates gamma.
    Fig 7(b) in the paper.
    """ 
    if gammas is None:
        gammas = GAMMAS_DEFAULT
    fig, ax = plt.subplots()
    ifid = []
    for gamma in gammas:
        sim = simulation(model=model, env=env, h=h, dt=dt, 
                         noise=noise, gamma=gamma)
        sim.get_simulation()
        ifid.append(sim.infidelity)
    for n_gamma, gamma in gammas:
        ax.plot(ifid[n_gamma], label=f'Gamma={gamma}')
    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Infidelity')
    ax.set_title(r'Infidelity over time at different dissipative rates gamma')
    return fig


def plot_bosons(model: Model = Model.SB1S, 
                env: Env = Env.ADC, 
                noises: List[float] = None,
                bosons: List[int] = [4]):
    """
    Average bosonic occupation over time at different noises.
    Fig 8(a) in the paper.
    """ 
    if noises is None:
        noises = NOISES_DEFAULT
    fig, ax = plt.subplots()
    for n_boson, boson in enumerate(bosons):
        for noise in noises:
            sim = simulation(model=model, env=env, noise=noise, n_boson=boson)
            sim.set_optimal_product_formula()
            sim.get_simulation()
            ax.plot(sim.timesteps, sim.bosons_em, 
                    label=f'Boson={boson}, noise={noise}')
        sim = simulation(model=model, env=env, noise=min(noises), n_boson=boson)
        sim.set_optimal_product_formula()
        sim.get_simulation()
        ax.plot(sim.timesteps, sim.bosons_exact, label=f'Boson={boson}, exact')
    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Bosonic occupation')
    ax.set_title(r'Average bosonic occupation over time at different noises')
    return fig


def plot_spin(model: Model = Model.SB1S, 
                env: Env = Env.ADC, 
                noises: List[float] = None,
                saxis: Axis = Axis.ZAX):
    """
    Average spin in x, y, or z over time at different noises.
    Fig 8(b) in the paper.
    """ 
    if noises is None:
        noises = NOISES_DEFAULT
    fig, ax = plt.subplots()
    #
    for noise in noises:
        sim = simulation(model=model, env=env, noise=noise)
        sim.set_optimal_product_formula()
        sim.get_simulation()
        ax.plot(sim.timesteps, sim.s_em[saxis], label=f'Noise={noise}')
    sim = simulation(model=model, env=env, noise=min(noises))
    sim.set_optimal_product_formula()
    sim.get_simulation()
    ax.plot(sim.timesteps, sim.s_exact[saxis], label='Exact')
    ax.set_xlabel(r'Time')
    ax.set_ylabel(f'Spin {saxis.name.lower()}')
    ax.set_title(r'Average spin over time at different noises')
    return fig


def plot_spincorrelation(model: Model = Model.SB2S, 
                env: Env = Env.ADC, 
                noises: List[float] = None,
                saxis: Axis = Axis.ZAX):
    """
    Average spin correlation in x, y, or z over time at different noises.
    Fig 9 in the paper.
    """ 
    if noises is None:
        noises = NOISES_DEFAULT
    fig, ax = plt.subplots()
    #
    for noise in noises:
        sim = simulation(model=model, env=env, noise=noise)
        sim.set_optimal_product_formula()
        sim.get_simulation()
        ax.plot(sim.timesteps, sim.scorr_em[saxis], label=f'Noise={noise}')
    sim = simulation(model=model, env=env, noise=min(noises))
    sim.set_optimal_product_formula()
    sim.get_simulation()
    ax.plot(sim.timesteps, sim.scorr_exact[saxis], label='Exact')
    ax.set_xlabel(r'Time')
    ax.set_ylabel(f'Spin correlation {saxis.name.lower()}')
    ax.set_title(r'Average spin correlation over time at different noises')
    return fig