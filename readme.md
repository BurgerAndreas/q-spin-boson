# Simulating the Spin-Boson Model on a Quantum Computer

Minimal version of the code used in my master thesis and in the paper [Digital Quantum Simulation of the Spin-Boson Model under Markovian Open-System Dynamics](https://www.mdpi.com/1099-4300/24/12/1766).

## Installation

### Pip

<https://qiskit.org/documentation/release_notes.html>

```bash
pip install -r requirements.txt
```

### Conda

```bash
conda install -c conda-forge pip matplotlib seaborn python-dotenv jupyter notebook qutip -y

python -m pip install qiskit==0.36.2 qiskit-terra==0.20.2 qiskit-aer==0.10.4 qiskit-ignis==0.7.1 qiskit-ibmq-provider==0.19.1 qiskit-experiments pylatexenc matplotlib seaborn python-dotenv jupyter notebook qutip mypy pylint
```

## Structure

```bash
q-spin-boson/
|-- saved-models/ # saved simulations (pickle files)
|   |-- ...
|
|-- settings/ # basically code which is not functions
|   |-- ...
|
|-- src/
|   |-- helpers/ # functions. called by model_base.py
|   |   |-- ...
|   |
|   |-- model_base.py # simulation base class. called by model_<...>.py
|   |-- model_<...>.py # simulations. called by main.py
|   |-- plotting.py # plot simulations
|
|-- main.py
```

## Todos

[ ] jupyter notebook env variables
[ ] __init__.py
[ ] matplotlib show plots as an env setting
[ ] code timing
[ ] docker
