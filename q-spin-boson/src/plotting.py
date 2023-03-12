

# print('Time-averaged infidelity noiseless:', np.sum(fidelity_evo) / np.shape(fidelity_evo)[0])

def plot(sims: List[Simulation], ):
    """
    Time-averaged infidelity over the timestep size dt in noiseless simulations.
    Takes in multiple models.
    """
    return fig


def plot(dts: List[float], noises: List[float]):
    """
    Time-averaged infidelity over the timestep size dt at different noise levels.
    Takes in a single model.
    """
    return fig


def plot(noises: List[float]):
    """
    Time-averaged and final infidelity over the noise level.
    Takes in a single model.
    """
    return fig


def plot(noises: List[float]):
    """
    Infidelity over time at different noise levels.
    Takes in a single model.
    """
    return fig


def plot(gammas: List[float]):
    """
    Time-averaged infidelity over dissipative rate w/o noise.
    Takes in a single model.
    """ 
    return fig


def 7b(gammas: List[float]):
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