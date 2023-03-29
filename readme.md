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
conda install -c conda-forge pip matplotlib seaborn python-dotenv jupyter notebook qutip ipympl -y

python -m pip install qiskit==0.36.2 qiskit-terra==0.20.2 qiskit-aer==0.10.4 qiskit-ignis==0.7.1 qiskit-ibmq-provider==0.19.1 qiskit-experiments pylatexenc matplotlib seaborn python-dotenv jupyter notebook qutip mypy pylint
```

### Docker

Start docker daemon / desktop app

```bash
docker build -t myimage -f dockerfile .
docker images

# docker run -it --name mycontainer myimage
# this will save changes made inside the container to your local machine
docker run -it -v $(pwd)/q-spin-boson/data:/app/data --name mycontainer myimage

docker ps -a
docker rm -f mycontainer
```

## File Structure

```bash
q-spin-boson/
|-- data/
|   |-- saved-models/ # saved simulations (pickle files)
|   |-- plots/ 
|   |-- plots-circuits/ 
|
|-- settings/ # code which is not functions
|   |-- ...
|
|-- src/
|   |-- helpers/ # functions called by model_base.py
|   |   |-- ...
|   |
|   |-- model_base.py # simulation base class. called by model_<...>.py
|   |-- model_<...>.py # simulation subclasses. called by main.py
|   |-- plot_simulations.py # plot simulations
|   |-- plot_illustrations.py # circuit illustrations
|
|-- main.py
|-- paper.ipynb
|-- masterthesis.ipynb
```

## Class Structure

```bash
Simulation # base class
|-- SSpinBosonSimulation # Single spin spin-boson simulation
|   # Different hamiltonians
|
|-- DSpinBosonSimulation # Double spin spin-boson simulation
|
|-- JCSpinBosonSimulation # Jaynes-Cummings simulation 
|
|-- TwoLevelSimulation # Spin system simulation 
|
simulation() # Function to get any simulation subclass
```
