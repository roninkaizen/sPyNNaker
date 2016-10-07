from pacman.executor.injection_decorator import inject_items
from spynnaker.pyNN.models.neuron.synapse_types.synapse_type_exponential \
    import get_exponential_decay_and_init
from spynnaker.pyNN.models.neural_properties.neural_parameter \
    import NeuronParameter
from spynnaker.pyNN.models.neuron.synapse_types.abstract_synapse_type \
    import AbstractSynapseType

from data_specification.enums.data_type import DataType


class SynapseTypeDualExponential(AbstractSynapseType):

    default_parameters = {
        'tau_syn_E': 5.0, 'tau_syn_E2': 5.0, 'tau_syn_I': 5.0
    }

    def get_n_synapse_types(self):
        return 3

    @classmethod
    def get_synapse_id_by_target(cls, target):
        if target == "excitatory":
            return 0
        elif target == "excitatory2":
            return 1
        elif target == "inhibitory":
            return 2
        return None

    def get_synapse_targets(self):
        return "excitatory", "excitatory2", "inhibitory"

    def get_n_synapse_type_parameters(self):
        return 6

    @inject_items({"machine_time_step": "MachineTimeStep"})
    def get_synapse_type_parameters(self, neuron_cell, machine_time_step):
        e_decay, e_init = get_exponential_decay_and_init(
            neuron_cell.get("tau_syn_E"), machine_time_step)
        e_decay2, e_init2 = get_exponential_decay_and_init(
            neuron_cell.get("tau_syn_E2"), machine_time_step)
        i_decay, i_init = get_exponential_decay_and_init(
            neuron_cell.get("tau_syn_I"), machine_time_step)

        return [
            NeuronParameter(e_decay, DataType.UINT32),
            NeuronParameter(e_init, DataType.UINT32),
            NeuronParameter(e_decay2, DataType.UINT32),
            NeuronParameter(e_init2, DataType.UINT32),
            NeuronParameter(i_decay, DataType.UINT32),
            NeuronParameter(i_init, DataType.UINT32)
        ]

    def get_n_cpu_cycles_per_neuron(self):

        # A guess
        return 100
