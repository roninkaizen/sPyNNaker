from pacman.executor.injection_decorator import inject_items
from spynnaker.pyNN.models.neural_properties.neural_parameter \
    import NeuronParameter
from spynnaker.pyNN.models.neuron.neuron_models.abstract_neuron_model \
    import AbstractNeuronModel

from data_specification.enums.data_type import DataType

import numpy


class NeuronModelLeakyIntegrate(AbstractNeuronModel):

    default_parameters = {
        'v_init': None, 'v_rest': -65.0, 'tau_m': 20.0, 'cm': 1.0,
        'i_offset': 0
    }

    state_variables = ['v']

    def _r_membrane(self, neuron_cell):
        return (neuron_cell.get("tau_m") / neuron_cell.get('cm'))

    def _exp_tc(self, neuron_cell, machine_time_step):
        return numpy.exp(
            float(-machine_time_step) / (1000.0 * neuron_cell.get("tau_m")))

    def get_n_neural_parameters(self):
        return 5

    @inject_items({"machine_time_step": "MachineTimeStep"})
    def get_neural_parameters(self, neuron_cell, machine_time_step):
        v_init = neuron_cell.get_state_variable("v")
        if v_init is None:
            v_init = neuron_cell.get("v_init")
        if v_init is None:
            v_init = neuron_cell.get("v_rest")

        return [

            # membrane voltage [mV]
            # REAL     V_membrane;
            NeuronParameter(v_init, DataType.S1615),

            # membrane resting voltage [mV]
            # REAL     V_rest;
            NeuronParameter(neuron_cell.get("v_rest"), DataType.S1615),

            # membrane resistance [MOhm]
            # REAL     R_membrane;
            NeuronParameter(self._r_membrane(neuron_cell), DataType.S1615),

            # 'fixed' computation parameter - time constant multiplier for
            # closed-form solution
            # exp( -(machine time step in ms)/(R * C) ) [.]
            # REAL     exp_TC;
            NeuronParameter(
                self._exp_tc(neuron_cell, machine_time_step), DataType.S1615),

            # offset current [nA]
            # REAL     I_offset;
            NeuronParameter(neuron_cell.get("i_offset"), DataType.S1615)
        ]

    def get_n_global_parameters(self):
        return 0

    def get_global_parameters(self):
        return []

    def get_n_cpu_cycles_per_neuron(self):

        # A bit of a guess
        return 80
