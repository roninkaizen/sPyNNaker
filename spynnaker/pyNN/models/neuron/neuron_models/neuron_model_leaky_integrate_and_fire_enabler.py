from pacman.executor.injection_decorator import inject_items
from pacman.model.decorators import overrides
from spynnaker.pyNN.models.neural_properties import NeuronParameter
from .neuron_model_leaky_integrate import NeuronModelLeakyIntegrate
from spynnaker.pyNN.utilities import utility_calls

from data_specification.enums import DataType

import numpy
from enum import Enum


class _LIF_TYPES(Enum):
    REFRACT_COUNT = (1, DataType.INT32)
    V_RESET = (2, DataType.S1615)
    TAU_REFRACT = (3, DataType.INT32)
    V_MAX = (4, DataType.S1615)
    V_ENABLE = (5, DataType.S1615)

    def __new__(cls, value, data_type):
        obj = object.__new__(cls)
        obj._value_ = value
        obj._data_type = data_type
        return obj

    @property
    def data_type(self):
        return self._data_type


class NeuronModelLeakyIntegrateAndFireEnabler(NeuronModelLeakyIntegrate):

    def __init__(
            self, n_neurons, v_init, v_rest, tau_m, cm, i_offset, v_reset,
            tau_refrac, v_max, v_enable):
        NeuronModelLeakyIntegrate.__init__(
            self, n_neurons, v_init, v_rest, tau_m, cm, i_offset)
        self._v_reset = utility_calls.convert_param_to_numpy(
            v_reset, n_neurons)
        self._tau_refrac = utility_calls.convert_param_to_numpy(
            tau_refrac, n_neurons)
        self._v_max = v_max
        self._v_enable = v_enable
        self._countdown_to_refactory_period = \
            utility_calls.convert_param_to_numpy(0, n_neurons)

        self._my_units = {'v_reset': 'mV', 'tau_refac': 'ms', 'v_max': 'mV',
                          'v_enable': 'mV'}

    @property
    def v_max(self):
        return self._v_max

    @v_max.setter
    def v_max(self, v_max):
        self._v_max = utility_calls.convert_param_to_numpy(
            v_max, self._n_neurons)

    @property
    def v_enable(self):
        return self._v_enable

    @v_enable.setter
    def v_enable(self, v_enable):
        self._v_enable = utility_calls.convert_param_to_numpy(
            v_enable, self._n_neurons)

    @property
    def v_reset(self):
        return self._v_reset

    @v_reset.setter
    def v_reset(self, v_reset):
        self._v_reset = utility_calls.convert_param_to_numpy(
            v_reset, self._n_neurons)

    @property
    def tau_refrac(self):
        return self._tau_refrac

    @tau_refrac.setter
    def tau_refrac(self, tau_refrac):
        self._tau_refrac = utility_calls.convert_param_to_numpy(
            tau_refrac, self._n_neurons)

    @overrides(NeuronModelLeakyIntegrate.get_n_neural_parameters)
    def get_n_neural_parameters(self):
        return NeuronModelLeakyIntegrate.get_n_neural_parameters(self) + len(_LIF_TYPES)

    def _tau_refrac_timesteps(self, machine_time_step):
        return numpy.ceil(self._tau_refrac /
                          (machine_time_step / 1000.0))

    @inject_items({"machine_time_step": "MachineTimeStep"})
    def get_neural_parameters(self, machine_time_step):
        params = NeuronModelLeakyIntegrate.get_neural_parameters(self)
        params.extend([

            # count down to end of next refractory period [timesteps]
            # int32_t  refract_timer;
            NeuronParameter(
                self._countdown_to_refactory_period,
                _LIF_TYPES.REFRACT_COUNT.data_type),

            # post-spike reset membrane voltage [mV]
            # REAL     V_reset;
            NeuronParameter(self._v_reset, _LIF_TYPES.V_RESET.data_type),

            # refractory time of neuron [timesteps]
            # int32_t  T_refract;
            NeuronParameter(
                self._tau_refrac_timesteps(machine_time_step),
                _LIF_TYPES.TAU_REFRACT.data_type),

            NeuronParameter(self._v_max, _LIF_TYPES.V_MAX.data_type),
            NeuronParameter(self._v_enable, _LIF_TYPES.V_ENABLE.data_type)

        ])
        return params

    @overrides(NeuronModelLeakyIntegrate.get_neural_parameter_types)
    def get_neural_parameter_types(self):
        if_types = NeuronModelLeakyIntegrate.get_neural_parameter_types(self)
        if_types.extend([item.data_type for item in _LIF_TYPES])
        return if_types

    def get_n_cpu_cycles_per_neuron(self):

        # A guess - 20 for the reset procedure
        return NeuronModelLeakyIntegrate.get_n_cpu_cycles_per_neuron(self) + 20

    @overrides(NeuronModelLeakyIntegrate.get_units)
    def get_units(self, variable):
        if variable in self._my_units:
            return self._my_units[variable]
        else:
            return NeuronModelLeakyIntegrate.get_units(variable)
