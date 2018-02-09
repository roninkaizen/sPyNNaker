from spynnaker.pyNN.models.neuron.neuron_models \
    import NeuronModelLeakyIntegrateAndFire
from spynnaker.pyNN.models.neuron.synapse_types import SynapseTypeExponential
from spynnaker.pyNN.models.neuron.input_types import InputTypeCurrent
from spynnaker.pyNN.models.neuron.threshold_types import ThresholdTypeHTDynamic
from spynnaker.pyNN.models.neuron import AbstractPopulationVertex

# global objects
DEFAULT_MAX_ATOMS_PER_CORE = 255


class HillTononi(AbstractPopulationVertex):
    """
        Hill Tononi Neuron model: leaky integrate and fire neuron, with
        intrinsic HH-like currents, bi-exponential synapses, voltage-dependent
        NMDA, short-term plasticity and dynamic threshold.
    """

    _model_based_max_atoms_per_core = DEFAULT_MAX_ATOMS_PER_CORE

    default_parameters = {
        'tau_m': 20.0, 'cm': 1.0, 'v_rest': -65.0, 'v_reset': -65.0,

        'v_thresh': 0.0, 'v_thresh_resting': -50, 'v_thresh_tau':2,
        'v_thresh_Na_reversal':30,

        'tau_syn_E': 5.0, 'tau_syn_I': 5.0,
        'tau_refrac': 0.1, 'i_offset': 0, 'isyn_exc': 0.0, 'isyn_inh': 0.0}

    none_pynn_default_parameters = {'v_init': None}

    def __init__(
            self, n_neurons, spikes_per_second=AbstractPopulationVertex.
            none_pynn_default_parameters['spikes_per_second'],
            ring_buffer_sigma=AbstractPopulationVertex.
            none_pynn_default_parameters['ring_buffer_sigma'],
            incoming_spike_buffer_size=AbstractPopulationVertex.
            none_pynn_default_parameters['incoming_spike_buffer_size'],
            constraints=AbstractPopulationVertex.none_pynn_default_parameters[
                'constraints'],
            label=AbstractPopulationVertex.none_pynn_default_parameters[
                'label'],

            # Neuron paramters
            tau_m=default_parameters['tau_m'],
            cm=default_parameters['cm'],
            v_rest=default_parameters['v_rest'],
            v_reset=default_parameters['v_reset'],
            i_offset=default_parameters['i_offset'],
            v_init=none_pynn_default_parameters['v_init'],

            # Threshold parameters
            v_thresh=default_parameters['v_thresh'],
            v_thresh_resting=default_parameters['v_thresh_resting'],
            v_thresh_tau=default_parameters['v_thresh_tau'],
            v_thresh_Na_reversal=default_parameters['v_thresh_Na_reversal'],

            # Synapse parameters
            tau_syn_E=default_parameters['tau_syn_E'],
            tau_syn_I=default_parameters['tau_syn_I'],
            tau_refrac=default_parameters['tau_refrac'],
            isyn_exc=default_parameters['isyn_exc'],
            isyn_inh=default_parameters['isyn_inh']):

        neuron_model = NeuronModelLeakyIntegrateAndFire(
            n_neurons, v_init, v_rest, tau_m, cm, i_offset,
            v_reset, tau_refrac)
        synapse_type = SynapseTypeExponential(
            n_neurons, tau_syn_E, tau_syn_I, isyn_exc, isyn_inh)
        input_type = InputTypeCurrent()
        threshold_type = ThresholdTypeHTDynamic(n_neurons, v_thresh,
                                                v_thresh_resting,
                                                v_thresh_tau,
                                                v_thresh_Na_reversal)

        AbstractPopulationVertex.__init__(
            self, n_neurons=n_neurons, binary="Hill_Tononi.aplx", label=label,
            max_atoms_per_core=HillTononi._model_based_max_atoms_per_core,
            spikes_per_second=spikes_per_second,
            ring_buffer_sigma=ring_buffer_sigma,
            incoming_spike_buffer_size=incoming_spike_buffer_size,
            model_name="Hill_Tononi", neuron_model=neuron_model,
            input_type=input_type, synapse_type=synapse_type,
            threshold_type=threshold_type, constraints=constraints)

    @staticmethod
    def set_model_max_atoms_per_core(new_value=DEFAULT_MAX_ATOMS_PER_CORE):
        HillTononi._model_based_max_atoms_per_core = new_value

    @staticmethod
    def get_max_atoms_per_core():
        return HillTononi._model_based_max_atoms_per_core

    @property
    def isyn_exc(self):
        return self.synapse_type.initial_value_exc

    @property
    def isyn_inh(self):
        return self.synapse_type.initial_value_inh

    @isyn_exc.setter
    def isyn_exc(self, new_value):
        self.synapse_type.initial_value_exc = new_value

    @isyn_inh.setter
    def isyn_inh(self, new_value):
        self.synapse_type.initial_value_inh = new_value
