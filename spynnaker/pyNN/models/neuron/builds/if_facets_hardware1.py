from spynnaker.pyNN.models.neuron.builds.abstract_unsupported_neuron_build \
    import AbstractUnsupportedNeuronBuild


class IFFacetsConductancePopulation(AbstractUnsupportedNeuronBuild):
    """ Leaky integrate and fire neuron with conductance-based synapses and\
        fixed threshold as it is resembled by the FACETS Hardware Stage 1
    """

    default_parameters = {
        "g_leak": 40.0, "tau_syn_E": 30.0, "tau_syn_I": 30.0,
        "v_thresh": -55.0, "v_rest": -65.0, "e_rev_I": -80, "v_reset": -80.0,
        "v_init": None
    }
