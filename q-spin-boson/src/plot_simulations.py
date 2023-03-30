import numpy as np
import matplotlib.pyplot as plt
from typing import List, Sequence, Any # typechecking with mypy
from numpy.typing import NDArray # typechecking with mypy

from settings.parameters import Paras
from settings.types import Model, Env, H, Axis
from src.model_base import Simulation
from src.model_general import simulation

from settings.plot_styles import save_legend_extra, get_blues, \
    extended_palette
from settings.paths import PIC_FILE, DIR_PLOTS

DTS_DEFAULT = [0.1, 0.2, 0.3, 0.4, 0.5]
NOISES_DEFAULT = [0.01, 0.1, 1.]
GAMMAS_DEFAULT = [0., 0.5, 1., 1.5, 2., 2.5]


def plot_states(sim: Simulation, noisy = True, exact = True):
    """Plot state evolution over time.
    Optionally plot noiseless or noisy evolution.
    Optionally plot exact evolution.
    """
    # easier slicing
    evo = np.asarray(sim.evo_em) if noisy else np.asarray(sim.evo) 
    fig, ax = plt.subplots()
    lines = []
    labels = []
    for s in range(np.shape(evo)[1]):
        l, = ax.plot(sim.timesteps, evo[:, s], color=extended_palette[s])
        lines.append(l)
        labels.append(sim.labels[s])
    if exact:
        evo_exact = np.asarray(sim.evo_exact)
        for s in range(np.shape(evo_exact)[1]):
            ax.plot(sim.timesteps, evo_exact[:, s],
                    linestyle='dashed', color=extended_palette[s])
    fname = f'states_{sim.name}_{"noisy" if noisy else "noiseless"}{"_exact" if exact else ""}'
    legend_fig = save_legend_extra(lines, labels, fname)
    ax.set_xlabel('Time')
    ax.set_ylabel('States')
    ax.set_title(f'Noiseless state evolution {sim.model.value}')
    loc = f'{DIR_PLOTS}{fname}.{PIC_FILE}'
    fig.savefig(loc, format=PIC_FILE)
    return loc, legend_fig


def plot_ifid_vs_dt_env(model: Model = Model.SB1S, dts: List[float] = None):
    """
    Time-averaged infidelity over the timestep size in noiseless simulations.
    Compares w/o environment (Hamiltonian).
    Fig 4 in the paper.
    """
    if dts is None:
        dts = DTS_DEFAULT
    fig, ax = plt.subplots()
    ifid_avg_h_o1 = [] # 1st order
    ifid_avg_env_o1 = []
    ifid_avg_h_o2 = [] # 2nd order
    ifid_avg_env_o2 = []
    for dt in dts:
        # Hamiltonian
        sim = simulation(model=model, env=Env.NOENV, h=H.FRSTORD, dt=dt)
        ifid_avg_h_o1.append(np.mean(sim.infidelity))
        sim = simulation(model=model, env=Env.NOENV, h=H.SCNDORD, dt=dt)
        ifid_avg_h_o2.append(np.mean(sim.infidelity))
        # Environment
        sim = simulation(model=model, env=Env.ADC, h=H.FRSTORD, dt=dt)
        ifid_avg_env_o1.append(np.mean(sim.infidelity))
        sim = simulation(model=model, env=Env.ADC, h=H.SCNDORD, dt=dt)
        ifid_avg_env_o2.append(np.mean(sim.infidelity))
    # plot
    l1, = ax.plot(dts, ifid_avg_h_o1, c=extended_palette[0])
    l2, = ax.plot(dts, ifid_avg_env_o1, c=extended_palette[1])
    l3, = ax.plot(dts, ifid_avg_h_o2, c=extended_palette[0], ls='dashed')
    l4, = ax.plot(dts, ifid_avg_env_o2, c=extended_palette[1], ls='dashed')
    fname = f'ifid_vs_dt_env_{model.value}'
    legend_fig = save_legend_extra(
        [l1, l2, l3, l4], 
        ['Hamiltonian (1st order)', 
        'Hamiltonian + environment (1st order)', 
        'Hamiltonian (2nd order)', 
        'Hamiltonian + environment (2nd order)'],
        fname)
    ax.set_xlabel(r'Timestep size')
    ax.set_ylabel(r'Time-averaged infidelity')
    ax.set_title(r'Time-averaged infidelity over the timestep size in noiseless simulations')
    loc = f'{DIR_PLOTS}{fname}.{PIC_FILE}'
    fig.savefig(loc, format=PIC_FILE)
    return loc, legend_fig


def plot_ifid_vs_dt_noises(model: Model = Model.SB1S, 
                           env: Env = Env.ADC, 
                           dts: List[float] = None, 
                           noises: List[float] = None):
    """
    Time-averaged infidelity over the timestep size at different noise levels.
    Compares first and 2nd order under noise.
    Fig 5(a) in the paper.
    """
    if dts is None:
        dts = DTS_DEFAULT
    if noises is None:
        noises = NOISES_DEFAULT
    noises.sort()
    fig, ax = plt.subplots()
    ifid_avg_o1 = [] # 1st order
    ifid_avg_o2 = [] # 2nd order
    for dt in dts:
        for noise in noises:
            sim = simulation(model=model, env=env, h=H.FRSTORD, dt=dt, 
                             noise=noise)
            ifid_avg_o1.append(np.mean(sim.infidelity_em))
            sim = simulation(model=model, env=env, h=H.SCNDORD, dt=dt, 
                             noise=noise)
            ifid_avg_o2.append(np.mean(sim.infidelity_em))
    # plot
    lines = []
    labels = []
    cs = get_blues(len(noises))
    for n_noise, noise in enumerate(noises):
        l, = ax.plot(dts, ifid_avg_o1[n_noise::len(noises)], 
                     c=cs[n_noise])
        lines.append(l)
        labels.append(f'1st order, noise={noise}')
        l, = ax.plot(dts, ifid_avg_o2[n_noise::len(noises)],
                     c=cs[n_noise], ls='dashed') 
        lines.append(l)
        labels.append(f'2nd order, noise={noise}')
    fname = f'ifid_vs_dt_noises_{model.value}_{env.value}'
    legend_fig = save_legend_extra(lines, labels, fname)
    ax.set_xlabel(r'Timestep size')
    ax.set_ylabel(r'Time-averaged infidelity')
    ax.set_title(r'Time-averaged infidelity over the timestep size at different noise levels')
    loc = f'{DIR_PLOTS}{fname}.{PIC_FILE}'
    fig.savefig(loc, format=PIC_FILE)
    return loc, legend_fig


def plot_ifid_vs_noise(model: Model = Model.SB1S, 
                        env: Env = Env.ADC,
                        h: H = H.SCNDORD, 
                        dt: float = 0.2,
                        noises: List[float] = None):
    """Time-averaged and final infidelity over the noise level.
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
        ifid_avg.append(np.mean(sim.infidelity_em))
        ifid_final.append(sim.infidelity_em[-1])
    # plot
    ls = 'dashed' if sim.h == H.SCNDORD else 'solid'
    l1, = ax.plot(noises, ifid_avg, ls=ls)
    l2, = ax.plot(noises, ifid_final, ls=ls)
    fname = f'ifid_vs_noise_{model.value}_{env.value}_{h.value}'
    legend_fig = save_legend_extra(
        [l1, l2], 
        ['Time-averaged infidelity', 'Final infidelity'],
        fname)
    ax.set_xlabel(r'Noise level')
    ax.set_ylabel(r'Infidelity')
    ax.set_title(r'Time-averaged and final infidelity over the noise level')
    loc = f'{DIR_PLOTS}{fname}.{PIC_FILE}'
    fig.savefig(loc, format=PIC_FILE)
    return loc, legend_fig


def plot_ifid_vs_time(model: Model = Model.SB1S, 
                        env: Env = Env.ADC, 
                        dt: float = 0.2,
                        noises: List[float] = None):
    """Infidelity over time at different noise levels.
    Compares first and 2nd order under noise.
    Fig 6 in the paper.
    """
    if noises is None:
        noises = NOISES_DEFAULT
    noises.sort()
    fig, ax = plt.subplots()
    ifid_o1 = [] # 1st order
    ifid_o2 = [] # 2nd order
    for noise in noises:
        sim = simulation(model=model, env=env, h=H.FRSTORD, dt=dt, 
                         noise=noise)
        ifid_o1.append(sim.infidelity)
        sim = simulation(model=model, env=env, h=H.SCNDORD, dt=dt, 
                         noise=noise)
        ifid_o2.append(sim.infidelity)
    # plot
    lines = []
    labels = []
    cs = get_blues(len(noises))
    for n_noise, noise in enumerate(noises):
        l, = ax.plot(ifid_o1[n_noise], c=cs[n_noise])
        lines.append(l)
        labels.append(f'1st order, noise={noise}')
        l, = ax.plot(ifid_o2[n_noise], c=cs[n_noise], ls='dashed')
        lines.append(l)
        labels.append(f'2nd order, noise={noise}')
    fname = f'ifid_vs_time_noises_{model.value}_{env.value}_{dt:.2f}'
    fname = fname.replace('.', '')
    legend_fig = save_legend_extra(lines, labels, fname)
    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Infidelity')
    ax.set_title(r'Infidelity over time at different noise levels')
    loc = f'{DIR_PLOTS}{fname}.{PIC_FILE}'
    fig.savefig(loc, format=PIC_FILE)
    return loc, legend_fig


def plot_ifid_vs_gamma(model: Model = Model.SB1S, 
                        env: Env = Env.ADC, 
                        dt: float = 0.2,
                        h: H = H.SCNDORD,
                        noise: float = 0.01,
                        gammas: List[float] = None):
    """Time-averaged infidelity over dissipative rate w/o noise.
    Fig 7(a) in the paper.
    """ 
    if gammas is None:
        gammas = GAMMAS_DEFAULT
    fig, ax = plt.subplots()
    ifid_avg = []
    ifid_avg_noise = []
    gammas.sort()
    for gamma in gammas:
        sim = simulation(model=model, env=env, h=h, dt=dt, 
                         noise=noise, gamma=gamma)
        ifid_avg.append(np.mean(sim.infidelity))
        ifid_avg_noise.append(np.mean(sim.infidelity_em))
    # plot
    ls = 'dashed' if h == H.SCNDORD else 'solid'
    l1, = ax.plot(gammas, ifid_avg, ls=ls)
    l2, = ax.plot(gammas, ifid_avg_noise, ls=ls)
    fname = f'ifid_vs_gamma_{model.value}_{env.value}_{h.value}_{dt:.2f}'
    fname = fname.replace('.', '')
    legend_fig = save_legend_extra(
            [l1, l2], 
            ['Time-averaged infidelity', 'Time-averaged infidelity with noise'],
            fname)
    ax.set_xlabel(r'Dissipative rate $\gamma$')
    ax.set_ylabel(r'Time-averaged infidelity')
    ax.set_title(r'Time-averaged infidelity over dissipative rate w/o noise')
    loc = f'{DIR_PLOTS}{fname}.{PIC_FILE}'
    fig.savefig(loc, format=PIC_FILE)
    return loc, legend_fig


def plot_ifid_vs_time_gammas(model: Model = Model.SB1S, 
                        env: Env = Env.ADC, 
                        dt: float = 0.2,
                        h: H = H.SCNDORD,
                        noise: float = 0.01,
                        gammas: List[float] = None):
    """Infidelity over time at different dissipative rates gamma.
    Fig 7(b) in the paper.
    """ 
    if gammas is None:
        gammas = GAMMAS_DEFAULT
    fig, ax = plt.subplots()
    ifid = []
    sim = None
    for gamma in gammas:
        sim = simulation(model=model, env=env, h=h, dt=dt, 
                         noise=noise, gamma=gamma)
        ifid.append(sim.infidelity)
    # plot
    lines = []
    labels = []
    cs = get_blues(len(gammas))
    ls = 'dashed' if h == H.SCNDORD else 'solid'
    for n_gamma, gamma in enumerate(gammas):
        l, = ax.plot(sim.timesteps, ifid[n_gamma], c=cs[n_gamma], ls=ls)
        lines.append(l)
        labels.append(f'Gamma={gamma}')
    fname = f'ifid_vs_time_gammas_{model.value}_{env.value}_{h.value}_{dt:.2f}'
    fname = fname.replace('.', '')
    legend_fig = save_legend_extra(lines, labels, fname)
    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Infidelity')
    ax.set_title(r'Infidelity over time at different dissipative rates gamma')
    loc = f'{DIR_PLOTS}{fname}.{PIC_FILE}'
    fig.savefig(loc, format=PIC_FILE)
    return loc, legend_fig


def plot_bosons(model: Model = Model.SB1S, 
                env: Env = Env.ADC, 
                bosons: List[int] = None, 
                noises: List[float] = None):
    """
    Average bosonic occupation over time at different noises.
    Fig 8(a) in the paper.
    """ 
    if bosons is None:
        bosons = [4]
    if noises is None:
        noises = NOISES_DEFAULT
    noises.sort()
    fig, ax = plt.subplots()
    lines = []
    labels = []
    cs = get_blues(len(noises))
    for boson in bosons:
        for n_noise, noise in enumerate(noises):
            sim = simulation(model=model, env=env, noise=noise, n_bos=boson, 
                     optimal_formula=True)
            ls = 'dashed' if sim.h == H.SCNDORD else 'solid'
            l, = ax.plot(sim.timesteps, sim.bosons_em, ls=ls, c=cs[n_noise])
            lines.append(l)
            labels.append(f'Boson={boson}, noise={noise}')
        # exact
        sim = simulation(model=model, env=env, noise=min(noises), n_bos=boson, 
                         optimal_formula=True)
        l, = ax.plot(sim.timesteps, sim.bosons_exact, ls='dotted', c=cs[0])
    fname = f'bosons_{model.value}_{env.value}'
    legend_fig = save_legend_extra(lines, labels, fname)
    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Bosonic occupation')
    ax.set_title(r'Average bosonic occupation over time at different noises')
    loc = f'{DIR_PLOTS}{fname}.{PIC_FILE}'
    fig.savefig(loc, format=PIC_FILE)
    return loc, legend_fig


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
    lines = []
    labels = []
    cs = get_blues(len(noises))
    for n_noise, noise in enumerate(noises):
        sim = simulation(model=model, env=env, noise=noise, 
                     optimal_formula=True)
        ls = 'dashed' if sim.h == H.SCNDORD else 'solid'
        l, = ax.plot(sim.timesteps, sim.s_em[saxis], ls=ls, c=cs[n_noise])
        lines.append(l)
        labels.append(f'Noise={noise}')
    # exact
    sim = simulation(model=model, env=env, noise=min(noises), 
                     optimal_formula=True)
    l, = ax.plot(sim.timesteps, sim.s_exact[saxis], linestyle='dotted', c=cs[0])
    fname = f'spin_{saxis.value}_{model.value}_{env.value}'
    legend_fig = save_legend_extra(lines, labels, fname)
    ax.set_xlabel(r'Time')
    ax.set_ylabel(f'Spin {saxis.value.lower()}')
    ax.set_title(r'Average spin over time at different noises')
    loc = f'{DIR_PLOTS}{fname}.{PIC_FILE}'
    fig.savefig(loc, format=PIC_FILE)
    return loc, legend_fig


def plot_spincorrelation(model: Model = Model.SB2S, 
                env: Env = Env.ADC, 
                paras: Paras = Paras.SB2S,
                noises: List[float] = None,
                saxis: Axis = Axis.ZAX):
    """Average spin correlation in x, y, or z over time at different noises.
    Fig 9 in the paper.
    """ 
    if noises is None:
        noises = NOISES_DEFAULT
    fig, ax = plt.subplots()
    #
    lines = []
    labels = []
    cs = get_blues(len(noises))
    for n_noise, noise in enumerate(noises):
        sim = simulation(model=model, env=env, noise=noise, 
                     optimal_formula=True)
        ls = 'dashed' if sim.h == H.SCNDORD else 'solid'
        l, = ax.plot(sim.timesteps, sim.scorr_em[saxis], ls=ls, c=cs[n_noise])
        lines.append(l)
        labels.append(f'Noise={noise}')
    # exact
    sim = simulation(model=model, env=env, noise=min(noises), 
                     optimal_formula=True)
    l, = ax.plot(sim.timesteps, sim.scorr_exact[saxis], linestyle='dashed', 
                 c=cs[0])
    fname = f'spincorrelation_{saxis.value}_{model.value}_{env.value}'
    legend_fig = save_legend_extra(lines, labels, fname)
    ax.set_xlabel(r'Time')
    ax.set_ylabel(f'Spin correlation {saxis.value.lower()}')
    ax.set_title(r'Average spin correlation over time at different noises')
    loc = f'{DIR_PLOTS}{fname}.{PIC_FILE}'
    fig.savefig(loc, format=PIC_FILE)
    return loc, legend_fig