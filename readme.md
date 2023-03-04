# Simulating the Spin-Boson Model on a Quantum Computer

Minimal version of the code used in my master thesis and in the paper [Digital Quantum Simulation of the Spin-Boson Model under Markovian Open-System Dynamics](https://www.mdpi.com/1099-4300/24/12/1766).

## Installation

### Conda

```bash
pip install qiskit 'qiskit[visualization]' qiskit-ignis==0.7.1 mypy pylint
conda install -c conda-forge matplotlib seaborn python-dotenv -y
```

### Docker

todo

## Future Work

### Acceptance Criteria

- [ ] Type hints
- [ ] Docstrings

### Next Steps

- [ ] Simulation calculations
- [ ] meas spins, correlations
- [ ] meas bosons
- [ ] exact lindblad
- [ ] fidelity
- [ ] nfixed


- [ ] Circuit to latex
- [ ] Circuit to image
- [ ] Plotting
- [ ] Docker

### Done

cleaned up (not typechecked)

- [x] environment_interaction
- [x] hamiltonian_qiskit
- [x] hamiltonian_matrix
- [x] conventions
- [x] binary

typechecked
