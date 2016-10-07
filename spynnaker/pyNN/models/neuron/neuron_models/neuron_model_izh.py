from pacman.executor.injection_decorator import inject_items
from spynnaker.pyNN.models.neural_properties.neural_parameter \
    import NeuronParameter
from spynnaker.pyNN.models.neuron.neuron_models.abstract_neuron_model \
    import AbstractNeuronModel

from data_specification.enums.data_type import DataType

_IZK_THRESHOLD = 30.0


class NeuronModelIzh(AbstractNeuronModel):

    default_parameters = {
        'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 2.0, 'v_init': -70.0,
        'u_init': -14.0, 'i_offset': 0
    }

    fixed_parameters = {'v_thresh': _IZK_THRESHOLD}

    state_variables = ['v', 'u']

    def get_n_neural_parameters(self):
        return 8

    @inject_items({"machine_time_step": "MachineTimeStep"})
    def get_neural_parameters(self, neuron_cell, machine_time_step):
        v_init = neuron_cell.get_state_variable("v")
        if v_init is None:
            v_init = neuron_cell.get("v_init")
        u_init = neuron_cell.get_state_variable("u")
        if u_init is None:
            u_init = neuron_cell.get("u_init")

        return [

            # REAL A
            NeuronParameter(neuron_cell.get('a'), DataType.S1615),

            # REAL B
            NeuronParameter(neuron_cell.get('b'), DataType.S1615),

            # REAL C
            NeuronParameter(neuron_cell.get('c'), DataType.S1615),

            # REAL D
            NeuronParameter(neuron_cell.get('d'), DataType.S1615),

            # REAL V
            NeuronParameter(v_init, DataType.S1615),

            # REAL U
            NeuronParameter(u_init, DataType.S1615),

            # offset current [nA]
            # REAL I_offset;
            NeuronParameter(neuron_cell.get('i_offset'), DataType.S1615),

            # current timestep - simple correction for threshold
            # REAL this_h;
            NeuronParameter(machine_time_step / 1000.0, DataType.S1615)
        ]

    def get_n_global_parameters(self):
        return 1

    @inject_items({"machine_time_step": "MachineTimeStep"})
    def get_global_parameters(self, machine_time_step):
        return [
            NeuronParameter(machine_time_step / 1000.0, DataType.S1615)
        ]

    def get_n_cpu_cycles_per_neuron(self):

        # A bit of a guess
        return 150
