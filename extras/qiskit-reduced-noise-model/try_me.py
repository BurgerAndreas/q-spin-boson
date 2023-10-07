from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeToronto
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

from noise_modeling import modified_noise_model

def noise_test(reduced_level=0.1):
  """
  Test 'modified_noise_model()' from 'noise_modeling.py'.
  Run simulations of a circuit with 
  (a) no noise (noiseless, ideal)
  (b) current IBM device noise levels
  (c) reduced noise levels
  and compare results.
  
  reduced_level (float): reduces gate infidelity and readout error 
  to = reduced_level * current_IBM_device.
  (Default: 0.1).
  """

  # Construct a quantum circuit
  circ = QuantumCircuit(3, 3)
  circ.h(0)
  circ.cx(0, 1)
  circ.cx(1, 2)
  circ.measure([0, 1, 2], [0, 1, 2])

  """ Noiseless results """
  # Noisy backend
  ideal_backend = AerSimulator()
  # Transpile the circuit for the noisy basis gates
  tcirc = transpile(circ, ideal_backend)
  # Execute noisy simulation and get counts
  result_ideal = ideal_backend.run(tcirc).result()
  counts_ideal = result_ideal.get_counts(0)
  # Plot
  plot_histogram(counts_ideal, title="Ideal, noiseless counts for 3-qubit GHZ state")
  plt.show()


  """ Noisy results """
  # Noisy backend
  device_backend = FakeToronto()
  sim_noisy = AerSimulator.from_backend(device_backend)
  # Transpile the circuit for the noisy basis gates
  tcirc = transpile(circ, sim_noisy)
  # Execute noisy simulation and get counts
  result_noise = sim_noisy.run(tcirc).result()
  counts_noise = result_noise.get_counts(0)
  # Plot
  plot_histogram(counts_noise, title="Counts for 3-qubit GHZ state with device noise model")
  plt.show()


  """ Reduced noise results """
  reduced_noise_model = modified_noise_model(_backend=FakeToronto(), gate_error_factor=reduced_level)
  sim_reduced_noise = AerSimulator(noise_model=reduced_noise_model)
  circ_t = transpile(circ, sim_reduced_noise)
  # Run and get reduced_noise counts
  result_reduced_noise = sim_reduced_noise.run(circ_t).result()
  counts_reduced_noise = result_reduced_noise.get_counts(0)
  # Plot
  plot_histogram(counts_reduced_noise, title="Counts for 3-qubit GHZ state with reduced noise model")
  plt.show()

  return counts_ideal, counts_noise, counts_reduced_noise
    

_counts_ideal, _counts_noise, _counts_reduced_noise = noise_test(reduced_level=0.1)
print('Noiseless, correct counts:\n', _counts_ideal)
print('Noisy, current device counts:\n', _counts_noise)
print('Reduced-noise counts:\n', _counts_reduced_noise)


    
