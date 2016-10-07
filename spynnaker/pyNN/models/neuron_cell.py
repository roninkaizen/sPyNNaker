from spinn_front_end_common.utilities import exceptions
from enum import Enum


class RecordingType(Enum):

    SPIKES = 1
    V = 2
    GSYN = 3


class NeuronCell(object):
    """ Stores all data about a cell.
    """

    next_id = 0

    def __init__(self, cellclass):

        self._id = NeuronCell.next_id
        NeuronCell.next_id += 1

        # Store data about the cell
        self._cellclass = cellclass

        # standard parameters
        if hasattr(cellclass, "default_parameters"):
            self._params = dict(cellclass.default_parameters)
        else:
            self._params = dict()

        # state variables
        if hasattr(cellclass, "state_variables"):
            self._state_variables = {
                key: None for key in cellclass.state_variables
            }
        else:
            self._state_variables = dict()

        if hasattr(cellclass, "recording_types"):

            # A dict of RecordingType to boolean indicating if the recording is
            # to be done for this cell
            self._is_recording = {
                key: False for key in cellclass.recording_types
            }

            # A dict of RecordingType to boolean indicating if the recording is
            # to be done to a file
            self._recording_to_file = {
                key: False for key in cellclass.recording_types
            }
        else:
            self._is_recording = dict()
            self._recording_to_file = dict()

        # Store any "fixed" parameters for later use (cannot get or set these)
        if hasattr(cellclass, "fixed_parameters"):
            self._fixed_parameters = dict(cellclass.fixed_parameters)

        # the position of the neuron
        self._position = None

        # change marker
        self._has_change_that_requires_mapping = True

        self.__dict__['_initialised'] = True

    def __cmp__(self, other):
        return self._id - other._id

    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        if ("_initialised" not in self.__dict__ or name in self.__dict__ or
                getattr(self.__class__, name, None) is not None):
            object.__setattr__(self, name, value)
        else:
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
        return self._cellclass

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
                "State variable {} does not exist.".format(param))
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
                "Trying to initialise a state variable {} which does not exist"
                .format(key))

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
        elif key in self._fixed_parameters:
            return self._fixed_parameters[key]
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
                "Trying to set a parameter {} which does not exist".format(
                    key))

    def remove_param(self, key):
        """ Removes a parameter (mainly done to convert from model to state\
            for v_init to v for past support)
        :param key: the name of the parameter to remove
        """
        if key in self._params:
            del self._params[key]
