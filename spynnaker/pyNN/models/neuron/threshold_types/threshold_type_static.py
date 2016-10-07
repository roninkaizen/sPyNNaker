from spynnaker.pyNN.models.neural_properties.neural_parameter \
    import NeuronParameter
from data_specification.enums.data_type import DataType
from spynnaker.pyNN.models.neuron.threshold_types.abstract_threshold_type \
    import AbstractThresholdType


class ThresholdTypeStatic(AbstractThresholdType):
    """ A threshold that is a static value
    """

    default_parameters = {'v_thresh': -50.0}

    def get_n_threshold_parameters(self):
        return 1

    def get_threshold_parameters(self, neuron_cell):
        return [
            NeuronParameter(neuron_cell.get("v_thresh"), DataType.S1615)
        ]

    def get_n_cpu_cycles_per_neuron(self):

        # Just a comparison, but 2 just in case!
        return 2
