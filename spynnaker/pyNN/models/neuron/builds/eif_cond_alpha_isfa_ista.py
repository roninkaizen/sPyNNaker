from spynnaker.pyNN.models.neuron.builds.abstract_unsupported_neuron_build \
    import AbstractUnsupportedNeuronBuild


class EIFConductanceAlphaPopulation(AbstractUnsupportedNeuronBuild):
    """ Exponential integrate and fire neuron with spike triggered and \
        sub-threshold adaptation currents (isfa, ista reps.)
    """

    default_parameters = {
        "tau_m": 9.3667, "cm": 0.281, "v_rest": -70.6, "v_reset": -70.6,
        "v_thresh": -50.4, "tau_syn_E": 5.0, "tau_syn_I": 0.5,
        "tau_refrac": 0.1, "i_offset": 0.0, "a": 4.0, "b": 0.0805,
        "v_spike": -40.0, "tau_w": 144.0, "e_rev_E": 0.0, "e_rev_I": -80.0,
        "delta_T": 2.0, "v_init": None
    }
