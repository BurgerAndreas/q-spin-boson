from qiskit import QuantumCircuit, QuantumRegister, Aer, transpile, assemble
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.test.mock import FakeJakarta, FakeToronto
from qiskit_aer import AerSimulator

# https://qiskit.org/textbook/ch-quantum-hardware/measurement-error-mitigation.html


def mitigate_error(circuit, qubits, shots=8*1024, noise_model=None):
    if noise_model is None:
        noise_model = AerSimulator.from_backend(FakeToronto())
    # calibration: measurements for basis states
    cal_circuit = QuantumRegister(qubits)
    meas_cal, state_labels = complete_meas_cal(qr=cal_circuit)
    # calibration: noise
    cal_noisy = transpile(meas_cal, backend=noise_model)
    result_cal_noisy = noise_model.run(cal_noisy, shots=shots).result()
    meas_noise_fitter = CompleteMeasFitter(result_cal_noisy, state_labels)
    # noisy simulation
    qc_noisy = transpile(circuit, backend=noise_model)
    results_noisy = noise_model.run(qc_noisy, shots=shots).result()
    counts_noisy = results_noisy.get_counts()
    # error mitigation
    meas_noise_filter = meas_noise_fitter.filter
    mitigated_results = meas_noise_filter.apply(results_noisy)
    counts_mitigated = mitigated_results.get_counts()
    return counts_noisy, counts_mitigated, qc_noisy

