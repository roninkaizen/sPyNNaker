from pacman.executor.injection_decorator import inject_items
from spynnaker.pyNN.models.pyNN_model import staticproperty
from spynnaker.pyNN.models.neural_properties.neural_parameter \
    import NeuronParameter
from spynnaker.pyNN.models.neuron.neuron_models.neuron_model_leaky_integrate \
    import NeuronModelLeakyIntegrate

from data_specification.enums.data_type import DataType

import numpy


class NeuronModelLeakyIntegrateAndFire(NeuronModelLeakyIntegrate):

    @staticproperty
    def default_parameters():  # @NoSelf
        parameters = {'v_reset': -65.0, 'tau_refrac': 0.1}
        parameters.update(NeuronModelLeakyIntegrate.default_parameters)
        return parameters

    state_variables = NeuronModelLeakyIntegrate.state_variables

    def get_n_neural_parameters(self):
        return NeuronModelLeakyIntegrate.get_n_neural_parameters(self) + 3

    def _tau_refrac_timesteps(self, tau_refrac, machine_time_step):
        return numpy.ceil(tau_refrac / (machine_time_step / 1000.0))

    @inject_items({"machine_time_step": "MachineTimeStep"})
    def get_neural_parameters(self, neuron_cell, machine_time_step):
        params = NeuronModelLeakyIntegrate.get_neural_parameters(
            self, neuron_cell)
        params.extend([

            # count down to end of next refractory period [timesteps]
            # int32_t  refract_timer;
            NeuronParameter(0, DataType.INT32),

            # post-spike reset membrane voltage [mV]
            # REAL     V_reset;
            NeuronParameter(neuron_cell.get("v_reset"), DataType.S1615),

            # refractory time of neuron [timesteps]
            # int32_t  T_refract;
            NeuronParameter(
                self._tau_refrac_timesteps(
                    neuron_cell.get("tau_refrac"), machine_time_step),
                DataType.INT32)
        ])
        return params

    def get_n_cpu_cycles_per_neuron(self):

        # A guess - 20 for the reset procedure
        return NeuronModelLeakyIntegrate.get_n_cpu_cycles_per_neuron(self) + 20
