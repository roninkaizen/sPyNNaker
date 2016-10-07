from spynnaker.pyNN.models.pyNN_model import classproperty
from spynnaker.pyNN.models.neuron_cell import RecordingType
from spynnaker.pyNN import exceptions
from spynnaker.pyNN.models.neuron.population_application_vertex \
    import PopulationApplicationVertex

_REQUIRED_PROPERTIES = [
    "neuron_model",
    "synapse_type",
    "input_type",
    "threshold_type"
]

_OPTIONAL_PROPERTIES = [
    "additional_input"
]

_SUBPROPERTIES = {
    "default_parameters": dict,
    "state_variables": list,
    "fixed_parameters": dict,
    "is_array_parameters": list
}


class AbstractNeuronBuild(object):

    @staticmethod
    def _check_subproperties(cls, value, prop):
        for subprop, subproptype in _SUBPROPERTIES.iteritems():
            if hasattr(value, subprop):
                subvalue = getattr(value, subprop)
                if not isinstance(subvalue, subproptype):
                    raise exceptions.SpynnakerException(
                        "Property {} of component {} specified for {} in "
                        "build {} is a {} not a {}".format(
                            subprop, value, prop, cls, type(subvalue),
                            subproptype))

    @classmethod
    def _check(cls):
        for prop in _REQUIRED_PROPERTIES:
            if not hasattr(cls, prop):
                raise exceptions.SpynnakerException(
                    "Neuron build {} is missing property {}".format(cls, prop))
            value = getattr(cls, prop)
            AbstractNeuronBuild._check_subproperties(cls, value, prop)
        for prop in _OPTIONAL_PROPERTIES:
            if hasattr(cls, prop):
                value = getattr(cls, prop)
                AbstractNeuronBuild._check_subproperties(cls, value, prop)

    @staticmethod
    def _get_values(parameters, cls, item):
        if hasattr(cls, item):
            if isinstance(parameters, dict):
                parameters.update(getattr(cls, item))
            else:
                parameters.extend(getattr(cls, item))

    @staticmethod
    def _get_all_values(cls, parameters, item):
        cls._check()
        cls._get_values(parameters, cls.neuron_model, item)
        cls._get_values(parameters, cls.synapse_type, item)
        cls._get_values(parameters, cls.input_type, item)
        cls._get_values(parameters, cls.threshold_type, item)
        if hasattr(cls, "additional_input"):
            cls._get_values(parameters, cls.additional_input, item)
        return parameters

    @classproperty
    def default_parameters(self):
        return self._get_all_values(self, dict(), "default_parameters")

    @classproperty
    def state_variables(self):
        return self._get_all_values(self, list(), "state_variables")

    @classproperty
    def fixed_parameters(self):
        return self._get_all_values(self, dict(), "fixed_parameters")

    @classproperty
    def is_array_parameters(self):
        return self._get_all_values(self, list(), "is_array_parameters")

    recording_types = [
        RecordingType.SPIKES, RecordingType.V, RecordingType.GSYN
    ]

    population_parameters = ["spikes_per_second", "ring_buffer_sigma"]

    supports_connector = True

    @classmethod
    def create_vertex(
            cls, population_parameters, neuron_cells, label, constraints,
            synapse_dynamics):
        return PopulationApplicationVertex(
            neuron_cells, label, cls,
            population_parameters.get("spikes_per_second", None),
            population_parameters.get("ring_buffer_sigma", None),
            population_parameters.get("incoming_spike_buffer_size"),
            constraints, synapse_dynamics)
