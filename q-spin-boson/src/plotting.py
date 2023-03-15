import numpy as np
import matplotlib.pyplot as plt
from typing import List, Sequence, Any # typechecking with mypy
from numpy.typing import NDArray # typechecking with mypy

from src.model_base import Simulation

# print('Time-averaged infidelity noiseless:', np.sum(fidelity_evo) / np.shape(fidelity_evo)[0])

def plot_ifid_vs_dt_models(sims: List[Simulation], ):
    """
    Time-averaged infidelity over the timestep size dt in noiseless simulations.
    Takes in multiple models.
    """
    return fig


def plot_ifid_vs_dt_noises(dts: List[float], noises: List[float]):
    """
    Time-averaged infidelity over the timestep size dt at different noise levels.
    Takes in a single model.
    """
    return fig


def plot_ifid_vs_noise(noises: List[float]):
    """
    Time-averaged and final infidelity over the noise level.
    Takes in a single model.
    """
    return fig


def plot_ifid_vs_time(noises: List[float]):
    """
    Infidelity over time at different noise levels.
    Takes in a single model.
    """
    return fig


def plot_ifid_vs_gamma(gammas: List[float]):
    """
    Time-averaged infidelity over dissipative rate w/o noise.
    Takes in a single model.
    """ 
    return fig


def plot_ifid_vs_time_gammas(gammas: List[float]):
    """
    Infidelity over time at different dissipative rates gamma.
    Takes in a single model.
    """ 
    return fig


def plot_bosons(bosons: List[int]):
    """
    Average bosonic occupation over time at different noises.
    Takes in a single model.
    """ 
    return fig


def plot_spin(saxis: Axis):
    """
    Average spin in x, y, or z over time at different noises.
    Takes in a single model.
    """ 
    return fig


def plot_spincorrelation(saxis: Axis):
    """
    Average spin correlation in x, y, or z over time at different noises.
    Takes in a single model.
    """ 
    return fig