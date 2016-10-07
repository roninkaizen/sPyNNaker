from spynnaker.pyNN.models.neuron.builds.abstract_unsupported_neuron_build \
    import AbstractUnsupportedNeuronBuild


class HHCondExp(AbstractUnsupportedNeuronBuild):
    """ Single-compartment Hodgkin-Huxley model with exponentially decaying \
        current input
    """

    default_parameters = {
        "gbar_K": 6.0, "cm": 0.2, "e_rev_Na": 50.0, "tau_syn_E": 0.2,
        "tau_syn_I": 2.0, "i_offset": 0.0, "g_leak": 0.01, "e_rev_E": 0.0,
        "gbar_Na": 20.0, "e_rev_leak": -65.0, "e_rev_I": -80, "e_rev_K": -90.0,
        "v_offset": -63, "v_init": None
    }
