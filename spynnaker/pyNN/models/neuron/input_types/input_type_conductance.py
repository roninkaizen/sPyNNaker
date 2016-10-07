from data_specification.enums.data_type import DataType
from spynnaker.pyNN.models.neural_properties.neural_parameter \
    import NeuronParameter
from spynnaker.pyNN.models.neuron.input_types.abstract_input_type \
    import AbstractInputType


class InputTypeConductance(AbstractInputType):
    """ The conductance input type
    """

    default_parameters = {'e_rev_E': 0.0, 'e_rev_I': -70.0}

    def get_global_weight_scale(self):
        return 1024.0

    def get_n_input_type_parameters(self):
        return 2

    def get_input_type_parameters(self, neuron_cell):
        return [
            NeuronParameter(neuron_cell.get("e_rev_E"), DataType.S1615),
            NeuronParameter(neuron_cell.get("e_rev_I"), DataType.S1615)
        ]

    def get_n_cpu_cycles_per_neuron(self, n_synapse_types):
        return 10
