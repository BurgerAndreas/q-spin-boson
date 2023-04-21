import numpy as np
import math
import scipy.linalg # expm
import matplotlib.pyplot as plt

def spin_boson_unitary_6x6(
        dt: float,
        epsilon: float,
        omega: float,
        lambd: float
) -> np.ndarray:
    """Unitary matrix for the spin-boson model with a 6x6 matrix representation.
    One spin and three bosonic levels (0, 1, 2).
    Args:
        dt: timestep size of one unitary
        loops: number of passes through unitary
        epsilon: spin x-field strength
        omega: harmonic oscillator (boson) frequency
        lambd: spin-boson coupling strength
    """
    # Define the Hamiltonian matrix
    h_mat = np.array([
        [-0.5, 0, 0, 0.5*epsilon, lambd, 0],
        [0, -0.5+omega, 0, lambd, 0.5*epsilon, math.sqrt(2)*lambd],
        [0, 0, -0.5+(2*omega), 0, math.sqrt(2)*lambd, 0.5*epsilon],
        [0.5*epsilon, lambd, 0, 0.5, 0, 0],
        [lambd, 0.5*epsilon, math.sqrt(2)*lambd, 0, 0.5+omega, 0],
        [0, math.sqrt(2)*lambd, 0.5*epsilon, 0, 0, 0.5+(2*omega)]
    ])
    # Calculate the unitary matrix
    return scipy.linalg.expm(-1j * h_mat * dt)

def spin_boson_simulation(
        t: float = 6,
        loops: int = 3,
        epsilon: float = 1,
        omega: float = 1,
        lambd: float = 1,
        init_state: np.ndarray = np.array([1, 0, 0, 0, 0, 0])
) -> tuple[np.ndarray, np.ndarray]:
    """Time evolution of the spin-boson model.
    Args:
        t: total time
        loops: number of passes through unitary
        epsilon: spin x-field strength
        omega: harmonic oscillator (boson) frequency
        lambd: spin-boson coupling strength
        init_state: initial state (spin x boson)
    """
    # dt: timestep size of one unitary
    dt = t / loops
    # get unitary
    unitary = spin_boson_unitary_6x6(dt, epsilon, omega, lambd)
    timesteps = np.arange(0, t+dt, dt)
    probabilities = np.zeros([loops+1, np.shape(unitary)[0]])
    # loop over unitary
    probabilities[0, :] = np.abs(init_state) ** 2
    for t_step in range(1, loops+1):
        init_state = unitary @ init_state
        probabilities[t_step, :] = np.abs(init_state) ** 2
    return timesteps, probabilities

def plot(timesteps, probabilities, name=None):
    fig, ax = plt.subplots()
    ax.plot(timesteps, probabilities)
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    if name:
        ax.set_title(name)
    ax.legend([r'$\vert\downarrow 0\rangle$', r'$\vert\downarrow 1\rangle$', 
               r'$\vert\downarrow 2\rangle$', r'$\vert\uparrow 0\rangle$', 
               r'$\vert\uparrow 1\rangle$', r'$\vert\uparrow 2\rangle$'])
    return fig

def plot_all(
        t: float = 6,
        epsilon: float = 1,
        omega: float = 1,
        lambd: float = 1,
        init_state: np.ndarray = np.array([1, 0, 0, 0, 0, 0])
    ) -> None:
    """Plot all the spin-boson simulations loop steps.
    Args:
        t: total time
        loops: number of passes through unitary
        epsilon: spin x-field strength
        omega: harmonic oscillator (boson) frequency
        lambd: spin-boson coupling strength
        init_state: initial state (spin x boson)
    """
    for loops in [60, 1, 2, 3]:
        name = f'{loops} loops, $\epsilon$={epsilon}, $\omega$={omega}, $\lambda$={lambd}'
        fig = plot(*spin_boson_simulation(t, loops, epsilon, omega, lambd, init_state), name)
    return