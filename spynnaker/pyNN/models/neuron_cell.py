from spinn_front_end_common.abstract_models.\
    abstract_changable_after_run import \
    AbstractChangableAfterRun
from spinn_front_end_common.utilities import exceptions
from enum.enum import Enum


class RecordingType(Enum):

    SPIKES = 1
    V = 2
    GSYN = 3


class NeuronCell(int):
    """ Stores all data about a cell.
    """

    _next_id = 0

    def __init__(
            self, population, original_class, default_parameters,
            state_variables, fixed_params, recording_types):
        """

        :param population: The population containing the cell
        :param original_class: The class of the model
        :param default_parameters:\
            A dict of default parameters to assign
        :param state_variables: A list of state variables supported by the cell
        :param fixed_params:\
            A dictionary of pre-determined parameter names to values that\
            cannot be changed by the user
        :param recording_types:\
            A list of the recording types supported for this cell
        """

        # Store the id of the cell
        int.__init__(NeuronCell._next_id)
        NeuronCell._next_id += 1

        # Store data about the cell
        self._population = population
        self._original_class = original_class

        # standard parameters
        self._params = dict()
        for key in default_parameters:
            self._params[key] = default_parameters[key]

        self._fixed_params = dict()
        for key in fixed_params:
            self._fixed_params[key] = fixed_params[key]

        # state variables
        self._state_variables = dict()
        for state_variable in state_variables:
            self._state_variables[state_variable] = None

        # A dict of RecordingType to boolean indicating if the recording is
        # to be done for this cell
        self._is_recording = dict()
        for recording_type in recording_types:
            self._is_recording[recording_type] = False

        # A dict of RecordingType to boolean indicating if the recording is
        # to be done to a file
        self._recording_to_file = dict()
        for recording_type in recording_types:
            self._recording_to_file[recording_type] = False

        # the position of the neuron
        self._position = None

        # synaptic link
        self._synapse_dynamics = None

        # change marker (only set to a value if the vertex supports it)
        if issubclass(self._original_class, AbstractChangableAfterRun):
            self._has_change_that_requires_mapping = True
        else:
            self._has_change_that_requires_mapping = None

    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        self.set_param(name, value)

    def set_parameters(self, **parameters):
        """ Set cell parameters given as name=value parameters
        """
        for name, value in parameters.iteritems():
            self.set_param(name, value)

    def get_parameters(self):
        """ Get cell parameters as a dict of names to values
        """
        return dict(self._params)

    @property
    def cell_type(self):
        return self._original_class

    @property
    def position(self):
        """ The position of this cell within its container

        :return: x, y, z coordinates
        """
        return self._position

    @position.setter
    def position(self, position):
        """ Set the position of this cell

        :param new_value: a tuple of x, y, z
        """
        self._position = position

    @property
    def local(self):
        return True

    def inject(self, current_source):
        # TODO: Implement this
        raise NotImplementedError

    def get_initial_value(self, variable):
        return self.get_state_variable(variable)

    def set_initial_value(self, variable, value):
        self.initialize(variable, value)

    def as_view(self):
        # TODO: Implement this
        raise NotImplementedError

    @property
    def state_variables(self):
        """ The state variables of the cell as a dict of name to initial value

        :rtype: dict
        """
        return self._state_variables

    def get_state_variable(self, param):
        """ Get the initial value of a state variable

        :param param: The name of the state variable to get
        :return: The initial value
        """
        if param not in self._state_variables:
            raise exceptions.ConfigurationException(
                "Parameter {} does not exist.".format(param))
        else:
            return self._state_variables[param]

    def initialize(self, key, new_value):
        """ Set the initial value of a state variable

        :param key: the name of the state variable
        :param new_value: the new initial value
        :return:None
        """
        if key in self._state_variables:
            self._has_change_that_requires_mapping = True
            self._state_variables[key] = new_value
        else:
            raise exceptions.ConfigurationException(
                "Trying to set a parameter which does not exist")

    @property
    def requires_mapping(self):
        """ True if a change has been made that requires mapping
        """
        return self._has_change_that_requires_mapping

    def mark_no_changes(self):
        """ Reset so that requires_mapping is False
        """
        self._has_change_that_requires_mapping = False

    def can_record(self, recording_type):
        """ Determine if the cell supports the recording type

        :param recording_type: A RecordingType object
        """
        return recording_type in self._is_recording

    def is_recording(self, recording_type):
        """ Determine if the cell is set to record a certain type
        """
        if recording_type not in self._is_recording:
            raise exceptions.ConfigurationException(
                "This cell cannot record {}".format(recording_type))
        return self._is_recording(recording_type)

    def is_recording_to_file(self, recording_type):
        """ Determine if the cell is set to record to a file
        """
        if recording_type not in self._is_recording:
            raise exceptions.ConfigurationException(
                "This cell cannot record {}".format(recording_type))
        return self._recording_to_file[recording_type]

    def set_recording(self, recording_type, is_recording, to_file):
        """ Set the recording status of a cell
        """
        if recording_type not in self._is_recording:
            raise exceptions.ConfigurationException(
                "This cell cannot record {}".format(recording_type))
        self._is_recording[recording_type] = is_recording
        self._recording_to_file[recording_type] = to_file

    def get(self, key):
        """ Get the value of a neuron parameter

        :param key: the name of the parameter to get
        :return: the parameter value for this neuron cell
        """
        if key in self._params:
            return self._params[key]
        elif key in self._fixed_params:
            return self._fixed_params[key]
        else:
            raise exceptions.ConfigurationException(
                "Parameter {} does not exist".format(key))

    def set_param(self, key, new_value):
        """ Set the value of a parameter

        :param key: the name of the parameter
        :param new_value: the new value of the parameter
        """
        if key in self._params:
            self._has_change_that_requires_mapping = True
            self._params[key] = new_value
        else:
            raise exceptions.ConfigurationException(
                "Trying to set a parameter which does not exist")

    def remove_param(self, key):
        """ Removes a parameter (mainly done to convert from model to state\
            for v_init to v for past support)
        :param key: the name of the parameter to remove
        """
        if key in self._params:
            del self._params[key]

    @property
    def synapse_dynamics(self):
        """ The synapse dynamics of this cell
        """
        return self._synapse_dynamics

    @synapse_dynamics.setter
    def synapse_dynamics(self, new_value):
        """ Set the synapse dynamics of this cell.
            Checks that the new dynamics is compatible with the current one\
            if one exists
        :param new_value: new synapse_dynamics
        """
        if self._synapse_dynamics is None:
            self._synapse_dynamics = new_value
        elif not self._synapse_dynamics.is_same_as(new_value):
            raise exceptions.ConfigurationException(
                "Currently only one type of STDP can be supported per"
                " target cell.")
        self._has_change_that_requires_mapping = True
