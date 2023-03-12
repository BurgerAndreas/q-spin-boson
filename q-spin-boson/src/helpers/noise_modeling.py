from warnings import warn, catch_warnings, filterwarnings
from numpy import allclose, inf, exp

from qiskit.quantum_info import average_gate_fidelity
from qiskit.providers.fake_provider import FakeToronto
# old qiskit versions
from qiskit.providers.aer.noise import NoiseModel, ReadoutError, QuantumError
from qiskit.providers.aer.noise.device.parameters import readout_error_values, gate_param_values, \
    thermal_relaxation_values, _NANOSECOND_UNITS
from qiskit.providers.aer.noise.device import basic_device_readout_errors, readout_error_values
from qiskit.providers.aer.noise.errors import thermal_relaxation_error, depolarizing_error
# new qiskit versions
# from qiskit_aer.noise import NoiseModel, ReadoutError, QuantumError
# from qiskit_aer.noise.device.parameters import readout_error_values, gate_param_values, \
#     thermal_relaxation_values, _NANOSECOND_UNITS
# from qiskit_aer.noise.device import basic_device_readout_errors, readout_error_values
# from qiskit_aer.noise.errors import thermal_relaxation_error, depolarizing_error


# taken from
# https://qiskit.org/documentation/_modules/qiskit/providers/aer/noise/device/models.html

# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


# used to be basic_device_gate_errors
def modified_noise_model(_backend=FakeToronto(),
                         gate_error_factor=1.,
                         gate_length_factor=None,
                         readout_error_factor=None,
                         gate_error=True,
                         thermal_relaxation=True,
                         gate_lengths=None,
                         gate_length_units='ns',
                         temperature=0,
                         warnings=True,
                         _relax_info=False,
                          _thermal_regime_info=False, 
                          _infidelity_ratio_info=False):
    """Builds NoiseModel with reduced noise.
    Based on QuantumErrors derived from a device's BackendProperties.
    First set thermal_relaxation, then adds depolarizing_error 
    to reach gate_error.
    gate_error is the target average gate infidelity.

    If non-default values are used gate_lengths should be a list
    of tuples ``(name, qubits, value)`` where ``name`` is the gate
    name string, ``qubits`` is either a list of qubits or ``None``
    to apply gate time to this gate one any set of qubits,
    and ``value`` is the gate time in nanoseconds.

    Qubit temperature enters only via the excited state population.
    If t=0, excited_population=0.

    Qubit frequency enters only via the excited state population.    
    Frequency is stored in relax_params.
    If freq=inf, excited_population=0.

    excited_population is computed via (freq, temperature).

    Args:
        _backend (ibmq.ibmqbackend.IBMQBackend or fake_provider.backends): 
            fake or real device backend
            (qiskit.providers.ibmq.ibmqbackend.IBMQBackend 
            or qiskit.providers.fake_provider.backends)
        gate_error_factor (float): reduces gate infidelity 
            by reducing depolarising.
            (bit-flip, phase-flip, both = Pauli X, Z, Y). (Default: 1).
        gate_length_factor (float): reduces thermal relaxation in gates.
            If None set to same as gate_error_factor. 
            (Default: None).
        readout_error_factor (float): reduces readout (measurement, bit-flip) 
            error. 
            If None set to same as gate_error_factor. 
            (Default: None).
        gate_error (bool): Include depolarizing gate errors (Default: True).
        gate_length_factor (float): reduces thermal relaxation (Default: 1).
        thermal_relaxation (Bool): Include thermal relaxation errors
            (Default: True).
        gate_lengths (list): Override device gate times with custom
            values. If None use gate times from
            backend properties 
            noise.device.gate_length_values(_backend.properties())
            (Default: None).
        gate_length_units (str): Time units for gate length values in 
            gate_lengths.
            Can be 'ns', 'ms', 'us', or 's' (Default: 'ns').
        temperature (double): qubit temperature in milli-Kelvin (mK)
                              (Default: 0).
        warnings (bool): Display warnings (Default: True).
        _relax_info (bool): Information on infidelity due to thermal relaxation 
            (Default: False).

    Returns:
        list: A list of tuples ``(label, qubits, QuantumError)``, for gates
        with non-zero quantum error terms, where `label` is the label of the
        noisy gate, `qubits` is the list of qubits for the gate.
    """
    properties = _backend.properties()
    # TODO match if set to none
    if gate_length_factor is None:
        gate_length_factor = gate_error_factor
    if readout_error_factor is None:
        readout_error_factor = gate_error_factor
    # TODO
    # Initilize empty errors
    depol_error = None
    relax_error = None
    # Generate custom gate time dict
    custom_times = {}
    relax_params = []
    if thermal_relaxation:
        # If including thermal relaxation errors load
        # T1, T2, and frequency values from properties
        relax_params = thermal_relaxation_values(properties)
        # If we are specifying custom gate times include
        # them in the custom times dict
        if gate_lengths:
            for name, qubits, value in gate_lengths:
                # Convert all gate lengths to nanosecond units
                time = value * _NANOSECOND_UNITS[gate_length_units]
                if name in custom_times:
                    custom_times[name].append((qubits, time))
                else:
                    custom_times[name] = [(qubits, time)]
    # Get the device gate parameters from properties
    device_gate_params = gate_param_values(properties)

    # TODO
    # Is T_2 < T_1 or 2T_1 > T_2 > T_1 ?
    if _thermal_regime_info:
        for name, qubits, gate_length, error_param in device_gate_params:
            regime_onebiggertwo = 0
            regime_twobiggerone = 0
            for qubit in qubits:
                t1, t2, freq = relax_params[qubit]
                if t2 < t1:
                    regime_onebiggertwo += 1
                    print('Qubit', qubit, 't2 < t1')
                else:
                    regime_twobiggerone += 1
                    print('Qubit', qubit, 't2 > t1')
            #print("Total:", 't2 < t1:', regime_onebiggertwo, 
            # 't1 < t2:', regime_twobiggerone)
            #break
    # TODO

    # Construct quantum errors
    errors = []
    # TODO only for printing _relax_info
    _num = 0  
    # TODO only for _infidelity_ratio_info
    # averaged over gates and qubits
    ifid_depol_total = 0
    ifid_thermal_total = 0
    for name, qubits, gate_length, error_param in device_gate_params:
        _num += 1
        # TODO reduces time for thermal relaxation to happen
        relax_time = gate_length * gate_length_factor
        # TODO
        # Check for custom gate time
        if name in custom_times:
            # Override with custom value
            warn(" Overwriting (modified) gate_length by custom values "
                 + "passed to modified_noise_model.", UserWarning)
            filtered = [
                val for q, val in custom_times[name]
                if q is None or q == qubits
            ]
            if filtered:
                # get first value
                relax_time = filtered[0]
        # Get relaxation error
        if thermal_relaxation:
            relax_error = _device_thermal_relaxation_error(
                qubits, relax_time, relax_params, temperature,
                thermal_relaxation)

        # Get depolarizing error channel
        if gate_error:
            with catch_warnings():
                filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    module="qiskit.providers.aer.noise.errors.errorutils"
                )
                # TODO reduces error_param
                if error_param is not None:
                    error_param *= gate_error_factor
                
                # TODO information on infidelity due to thermal relaxation
                # rz, sx, x, id
                if _relax_info and (name in ['x']) and (qubits == [1]):
                    print('Gate_error_factor', gate_error_factor)
                    print('name, qubits, gate_length, error_param: ', 
                          name, qubits, gate_length, error_param)
                    depol_error = _device_depolarizing_error(
                        qubits, error_param, relax_error, warnings=warnings, 
                        _relax_info=True)
                    return depol_error
                # TODO                
                else:
                    depol_error = _device_depolarizing_error(
                        qubits, error_param, relax_error, warnings=warnings)
        
            # TODO
            # add to average
            if _infidelity_ratio_info:
                depol_error, ifid_depol, ifid_thermal = \
                    _device_depolarizing_error(
                        qubits, error_param, relax_error, warnings=warnings, 
                        _infidelity_ratio_info=_infidelity_ratio_info)
                if name in ['cx', 'cnot', 'CX', 'CNOT']:
                    print('CNOT!')
                    ifid_depol_total += ifid_depol
                    ifid_thermal_total += ifid_thermal
                else:
                    ifid_depol_total += ifid_depol
                    ifid_thermal_total += ifid_thermal
            # TODO

        # Combine errors
        if depol_error is None and relax_error is None:
            # No error for this gate
            pass
        elif depol_error is not None and relax_error is None:
            # Append only the depolarizing error
            errors.append((name, qubits, depol_error))
        elif relax_error is not None and depol_error is None:
            # Append only the relaxation error
            errors.append((name, qubits, relax_error))
        else:
            # Append a combined error of depolarizing error
            # followed by a relaxation error
            combined_error = depol_error.compose(relax_error)
            errors.append((name, qubits, combined_error))
    # before: return errors
    # Instead add QuantumError objects to NoiseModel
    # initialize NoiseModel
    _noise_model = NoiseModel(
                basis_gates=NoiseModel.from_backend(_backend).basis_gates)
    for _obj in errors:
        _noise_model.add_quantum_error(error=_obj[2], instructions=_obj[0], 
                                       qubits=_obj[1])
    # add Measurement error to NoiseModel
    _qubit_num = _backend.configuration().num_qubits
    # Todo reduce readout error
    if readout_error_factor > 0.:
        for _readout_error in basic_device_readout_errors(properties):
            #qerror_readout = QuantumError(_readout_error[1].to_instruction())
            #_noise_model.add_quantum_error(error=qerror_readout, instructions='measure', qubits=_readout_error[0])
            if readout_error_factor == 1.:
                _noise_model.add_readout_error(error=_readout_error[1], 
                                               qubits=_readout_error[0])
            else:
                # qubit = _readout_error[0][0]
                #re_probs = basic_device_readout_errors(properties)[_readout_error[0][0]][1].to_dict()['probabilities']
                re_values = readout_error_values(properties)[_readout_error[0][0]]
                new_re_probs = [[1 - (re_values[0] * readout_error_factor), 
                                 (re_values[0] * readout_error_factor)],
                                [(re_values[1] * readout_error_factor), 
                                 1 - (re_values[1] * readout_error_factor)]]
                new_re = ReadoutError(new_re_probs)
                _noise_model.add_readout_error(error=new_re, 
                                               qubits=_readout_error[0])
    # TODO
    # return the average
    if _infidelity_ratio_info:
        return ifid_depol_total, ifid_thermal_total
    # TODO
    return _noise_model


def _device_depolarizing_error(qubits,
                               error_param,
                               relax_error=None,
                               warnings=True,
                               _relax_info=False,
                               _infidelity_ratio_info=False):
    """Construct a depolarizing_error for device"""

    # We now deduce the depolarizing channel error parameter in the
    # presence of T1/T2 thermal relaxation. We assume the gate error
    # parameter is given by e = 1 - F where F is the average gate fidelity,
    # and that this average gate fidelity is for the composition
    # of a T1/T2 thermal relaxation channel and a depolarizing channel.

    # For the n-qubit depolarizing channel E_dep = (1-p) * I + p * D, where
    # I is the identity channel and D is the completely depolarizing
    # channel. To compose the errors we solve for the equation
    # F = F(E_dep * E_relax)
    #   = (1 - p) * F(I * E_relax) + p * F(D * E_relax)
    #   = (1 - p) * F(E_relax) + p * F(D)
    #   = F(E_relax) - p * (dim * F(E_relax) - 1) / dim

    # Hence we have that the depolarizing error probability
    # for the composed depolarization channel is
    # p = dim * (F(E_relax) - F) / (dim * F(E_relax) - 1)
    if relax_error is not None:
        relax_fid = average_gate_fidelity(relax_error)
        relax_infid = 1 - relax_fid
        # TODO
        if _relax_info:
            print('relax_fid', relax_fid)
            print('relax_infid', relax_infid)
            return relax_infid
        # TODO
    else:
        relax_fid = 1
        relax_infid = 0
    if error_param is not None and error_param > relax_infid:
        num_qubits = len(qubits)
        dim = 2 ** num_qubits
        error_max = dim / (dim + 1)
        # Check if reported error param is un-physical
        # The minimum average gate fidelity is F_min = 1 / (dim + 1)
        # So the maximum gate error is 1 - F_min = dim / (dim + 1)
        error_param = float(error_param)
        error_max = float(error_max)
        if error_param > error_max:
            if warnings:
                warn('Device reported a gate error parameter greater than '
                     + 'maximum allowed value. Truncating to maximum value.')
            error_param = error_max
        # Model gate error entirely as depolarizing error
        num_qubits = len(qubits)
        dim = 2 ** num_qubits
        depol_param = dim * (error_param - relax_infid) / (dim * relax_fid - 1)
        max_param = 4 ** num_qubits / (4 ** num_qubits - 1)
        if depol_param > max_param:
            if warnings:
                warn('Device model returned a depolarizing error parameter '
                     + 'greater than maximum allowed value (%f > %f). '
                     + 'Truncating to maximum value.')
            depol_param = min(depol_param, max_param)
        # TODO
        if _infidelity_ratio_info:
            print('Thermal / Depolarizing Infidelity:', 
                  round(relax_infid / (error_param-relax_infid), 2))
            depol_error = depolarizing_error(
                depol_param, num_qubits, standard_gates=None)
            depol_infid = (error_param-relax_infid)
            return depol_error, depol_infid, relax_infid
        # TODO
        return depolarizing_error(
            depol_param, num_qubits, standard_gates=None)
    # TODO
    # if error_param < relax_infid:
    # no depolarizing error
    # but thermal is kept
    if _infidelity_ratio_info:
        prec = 5
        depol_infid = 0
        if error_param is None:
            print('No depolarizing. error_param is None.', 
                'depol_infid', round(depol_infid, prec), 
                'relax_infid:', round(relax_infid, prec))
        else:
            if error_param < relax_infid:
                print('No depolarizing. error_param < relax_infid.', 
                    'error_param:', round(error_param, prec), 
                    'depol_infid', round(depol_infid, prec), 
                    'relax_infid', round(relax_infid, prec))
            else:
                print('No errors.')
        return None, depol_infid, relax_infid
    # TODO
    return None


def _device_thermal_relaxation_error(qubits,
                                     gate_time,
                                     relax_params,
                                     temperature,
                                     thermal_relaxation=True):
    """Construct a thermal_relaxation_error for device"""
    # Check trivial case
    if not thermal_relaxation or gate_time is None or gate_time == 0:
        return None

    # Construct a tensor product of single qubit relaxation errors
    # for any multi qubit gates
    first = True
    error = None
    for qubit in qubits:
        t1, t2, freq = relax_params[qubit]
        t2 = _truncate_t2_value(t1, t2)
        population = _excited_population(freq, temperature)
        if first:
            error = thermal_relaxation_error(t1, t2, gate_time, population)
            first = False
        else:
            single = thermal_relaxation_error(t1, t2, gate_time, population)
            error = error.expand(single)
    return error


def _truncate_t2_value(t1, t2):
    """Return t2 value truncated to 2 * t1 (for t2 > 2 * t1)"""
    new_t2 = t2
    if t2 > 2 * t1:
        new_t2 = 2 * t1
        warn("Device model returned an invalid T_2 relaxation time greater than"
             f" the theoretical maximum value 2 * T_1 ({t2} > 2 * {t1})."
             " Truncating to maximum value.", UserWarning)
    return new_t2


def _excited_population(freq, temperature):
    """Return excited state population"""
    population = 0
    if freq != inf and temperature != 0:
        # Compute the excited state population from qubit
        # frequency and temperature
        # Boltzman constant  kB = 8.617333262-5 (eV/K)
        # Planck constant h = 4.135667696e-15 (eV.s)
        # qubit temperature temperatue = T (mK)
        # qubit frequency frequency = f (GHz)
        # excited state population = 1/(1+exp((2*h*f*1e9)/(kb*T*1e-3)))
        exp_param = exp((95.9849 * freq) / abs(temperature))
        population = 1 / (1 + exp_param)
        if temperature < 0:
            # negative temperate implies |1> is thermal ground
            population = 1 - population
    return population
