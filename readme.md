# Simulating the Spin-Boson Model on a Quantum Computer

Minimal version of the code used in my master thesis and in the paper [Digital Quantum Simulation of the Spin-Boson Model under Markovian Open-System Dynamics](https://www.mdpi.com/1099-4300/24/12/1766).

## Installation


### Conda

<https://qiskit.org/documentation/release_notes.html>

```bash
pip install qiskit==0.36.2 qiskit-terra==0.20.2 qiskit-aer==0.10.4 qiskit-ignis==0.7.1 qiskit-ibmq-provider==0.19.1 qiskit-experiments
```

```bash
pip install qiskit 'qiskit[visualization]' qiskit-ignis==0.7.1 qiskit-experiments mypy pylint
conda install -c conda-forge matplotlib seaborn python-dotenv jupyter notebook qutip -y
```

### Docker

todo

## Future Work

### Acceptance Criteria

- [ ] Type hints
- [ ] Docstrings

### Next Steps

- [ ] SB2S
- [ ] JC2S
- [ ] SB1SJC (inheret from SB1S)
- [ ] SB1SPZ (inheret from SB1S)

- [ ] Circuit to latex
- [ ] Circuit to image
- [ ] Plotting
- [ ] Docker

## Fixes

in

```path
<conda-environment>/lib/python3.10/site-packages/qiskit/ignis/verification/tomography/basis/circuits.py
/Users/a-burger-zeb/opt/anaconda3/envs/qiskit-sf/lib/python3.10/site-packages/qiskit/ignis/verification/tomography/basis/circuits.py
```

line 467, function `_tomography_circuits`

```python
# Add circuit being tomographed
# prep += circuit # REMOVED
# # meas_qubits, measured_qubits
prep = circuit.compose(prep, measured_qubits, front=True) # ADDED
# Generate Measurement circuit
for meas_label in meas_labels:
    meas = QuantumCircuit(*registers)
    if meas_label is not None:
        meas.barrier(*qubit_registers)
        for j in range(num_qubits):
            # meas += measurement(meas_label[j], # REMOVED
            #                     meas_qubits[j], 
            #                     clbits[j]) 
            meas_q = measurement(meas_label[j], # ADDED
                                    meas_qubits[j], 
                                    clbits[j])
            meas = meas.compose(meas_q, meas_qubits[j], clbits[j]) 
    # circ = prep + meas # REMOVED
    # meas_qubits, measured_qubits
    circ = prep.compose(meas, measured_qubits) # ADDED
```
