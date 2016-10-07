from spynnaker.pyNN.models.neuron.builds.abstract_unsupported_neuron_build \
    import AbstractUnsupportedNeuronBuild


class IFCurrAlpha(AbstractUnsupportedNeuronBuild):
    """ Leaky integrate and fire neuron with an alpha-shaped conductance input
    """

    default_parameters = {
        "tau_m": 20, "cm": 1.0, "v_rest": -65.0, "v_reset": -65.0,
        "v_thresh": -50.0, "tau_syn_E": 0.5, "tau_syn_I": 0.5,
        "tau_refrac": 0.1, "i_offset": 0, "v_init": None
    }
