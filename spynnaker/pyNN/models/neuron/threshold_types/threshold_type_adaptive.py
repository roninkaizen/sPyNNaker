from pacman.model.decorators import overrides

from spynnaker.pyNN.models.abstract_models import AbstractContainsUnits
from spynnaker.pyNN.utilities import utility_calls
from spynnaker.pyNN.models.neural_properties import NeuronParameter
from .abstract_threshold_type import AbstractThresholdType

from data_specification.enums import DataType

from enum import Enum


class _ADAPTIVE_TYPES(Enum):
    V_THRESH = (1, DataType.S1615)
    MIN_THRESH = (2, DataType.S1615)
    MAX_THRESH = (3, DataType.S1615)
    UP_THRESH = (4, DataType.S1615)
    DOWN_THRESH = (5, DataType.S1615)

    def __new__(cls, value, data_type):
        obj = object.__new__(cls)
        obj._value_ = value
        obj._data_type = data_type
        return obj

    @property
    def data_type(self):
        return self._data_type


class ThresholdTypeAdaptive(AbstractThresholdType, AbstractContainsUnits):

    """ A threshold that is a static value
    """

    def __init__(self, n_neurons, v_thresh, min_thresh, max_thresh,
                 up_thresh, down_thresh):
        AbstractThresholdType.__init__(self)
        AbstractContainsUnits.__init__(self)

        self._units = {'v_thresh': "mV"}

        self._n_neurons = n_neurons
        self._v_thresh = utility_calls.convert_param_to_numpy(
            v_thresh, n_neurons)
        self._min_thresh = utility_calls.convert_param_to_numpy(
            min_thresh, n_neurons)
        self._max_thresh = utility_calls.convert_param_to_numpy(
            max_thresh, n_neurons)
        self._up_thresh = utility_calls.convert_param_to_numpy(
            up_thresh, n_neurons)
        self._down_thresh = utility_calls.convert_param_to_numpy(
            down_thresh, n_neurons)


    @property
    def v_thresh(self):
        return self._v_thresh

    @v_thresh.setter
    def v_thresh(self, v_thresh):
        self._v_thresh = utility_calls.convert_param_to_numpy(
            v_thresh, self._n_neurons)

    @property
    def min_thresh(self):
        return self._min_thresh

    @min_thresh.setter
    def min_thresh(self, min_thresh):
        self._min_thresh = utility_calls.convert_param_to_numpy(
            min_thresh, self._n_neurons)

    @property
    def max_thresh(self):
        return self._max_thresh

    @max_thresh.setter
    def max_thresh(self, max_thresh):
        self._max_thresh = utility_calls.convert_param_to_numpy(
            max_thresh, self._n_neurons)

    @property
    def up_thresh(self):
        return self._up_thresh

    @up_thresh.setter
    def up_thresh(self, up_thresh):
        self._up_thresh = utility_calls.convert_param_to_numpy(
            up_thresh, self._n_neurons)

    @property
    def down_thresh(self):
        return self._down_thresh

    @down_thresh.setter
    def down_thresh(self, down_thresh):
        self._down_thresh = utility_calls.convert_param_to_numpy(
            down_thresh, self._n_neurons)

    @overrides(AbstractThresholdType.get_n_threshold_parameters)
    def get_n_threshold_parameters(self):
        return 5

    @overrides(AbstractThresholdType.get_threshold_parameters)
    def get_threshold_parameters(self):
        return [
            NeuronParameter(self._v_thresh, _ADAPTIVE_TYPES.V_THRESH.data_type),
            NeuronParameter(self._min_thresh, _ADAPTIVE_TYPES.MIN_THRESH.data_type),
            NeuronParameter(self._max_thresh, _ADAPTIVE_TYPES.MAX_THRESH.data_type),
            NeuronParameter(self._up_thresh, _ADAPTIVE_TYPES.UP_THRESH.data_type),
            NeuronParameter(self._down_thresh, _ADAPTIVE_TYPES.DOWN_THRESH.data_type)
        ]

    @overrides(AbstractThresholdType.get_threshold_parameter_types)
    def get_threshold_parameter_types(self):
        return [item.data_type for item in _ADAPTIVE_TYPES]

    def get_n_cpu_cycles_per_neuron(self):
        return 5

    @overrides(AbstractContainsUnits.get_units)
    def get_units(self, variable):
        return self._units[variable]
