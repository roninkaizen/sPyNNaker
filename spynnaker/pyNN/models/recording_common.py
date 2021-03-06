from spinn_utilities import logger_utils
from spinn_utilities.log import FormatAdapter
from spinn_utilities.timer import Timer
from spinn_front_end_common.utilities.exceptions import ConfigurationException
from spinn_front_end_common.utilities.globals_variables import get_simulator

from spynnaker.pyNN.models.common import AbstractSpikeRecordable
from spynnaker.pyNN.models.common import AbstractNeuronRecordable
from spynnaker.pyNN.models.neuron.input_types import InputTypeConductance

from collections import defaultdict
import numpy
import logging
# pylint: disable=protected-access

logger = FormatAdapter(logging.getLogger(__name__))


class RecordingCommon(object):
    # DO NOT DEFINE SLOTS! Multiple inheritance problems otherwise.
    # __slots__ = [
    #     "_indices_to_record",
    #     "_population",
    #     "_write_to_files_indicators"]

    def __init__(self, population):
        """ object to hold recording behaviour

        :param population: the population to record for
        """

        self._population = population

        # file flags, allows separate files for the recorded variables
        self._write_to_files_indicators = {
            'spikes': None,
            'gsyn_exc': None,
            'gsyn_inh': None,
            'v': None}

        # Create a dict of variable name -> bool array of indices in population
        # that are recorded (initially all False)
        self._indices_to_record = defaultdict(
            lambda: numpy.repeat(False, population.size))

    def _record(self, variable, sampling_interval=None, to_file=None,
                indexes=None):
        """ tells the vertex to record data

        :param variable: the variable to record, valued variables to record
        are: 'gsyn_exc', 'gsyn_inh', 'v', 'spikes'
        :param sampling_interval: the interval to record them
        :param indexes: List of indexes to record or None for all
        :return:  None
        """

        get_simulator().verify_not_running()
        # tell vertex its recording
        if variable == "spikes":
            if not isinstance(self._population._vertex,
                              AbstractSpikeRecordable):
                raise Exception("This population does not support the "
                                "recording of spikes!")
            self._population._vertex.set_recording_spikes(
                sampling_interval=sampling_interval, indexes=indexes)
        elif variable == "all":
            raise Exception("Illegal call with all")
        else:
            if not isinstance(self._population._vertex,
                              AbstractNeuronRecordable):
                raise Exception("This population does not support the "
                                "recording of {}!".format(variable))
            self._population._vertex.set_recording(
                variable, sampling_interval=sampling_interval, indexes=indexes)

        # update file writer
        self._write_to_files_indicators[variable] = to_file

        if variable == "gsyn_exc":
            if not isinstance(self._population._vertex.input_type,
                              InputTypeConductance):
                logger_utils.warn_once(
                    logger, "You are trying to record the excitatory "
                    "conductance from a model which does not use conductance "
                    "input. You will receive current measurements instead.")
        elif variable == "gsyn_inh":
            if not isinstance(self._population._vertex.input_type,
                              InputTypeConductance):
                logger_utils.warn_once(
                    logger, "You are trying to record the inhibtatory "
                    "conductance from a model which does not use conductance "
                    "input. You will receive current measurements instead.")

    def _set_v_recording(self):
        """ Sets the parameters etc that are used by the voltage recording

        :return: None
        """
        self._population._vertex.set_recording("v")

    def _set_spikes_recording(self, sampling_interval, indexes=None):
        """ sets the parameters etc that are used by the spikes recording

        :return: None
        """

    @staticmethod
    def pynn7_format(data, ids, sampling_interval, data2=None):
        n_machine_time_steps = len(data)
        n_neurons = len(ids)
        column_length = n_machine_time_steps * n_neurons
        times = [i * sampling_interval
                 for i in xrange(0, n_machine_time_steps)]
        if data2 is None:
            pynn7 = numpy.column_stack((
                numpy.repeat(ids, n_machine_time_steps, 0),
                numpy.tile(times, n_neurons),
                numpy.transpose(data).reshape(column_length)))
        else:
            pynn7 = numpy.column_stack((
                numpy.repeat(ids, n_machine_time_steps, 0),
                numpy.tile(times, n_neurons),
                numpy.transpose(data).reshape(column_length),
                numpy.transpose(data2).reshape(column_length)))
        return pynn7

    def _get_recorded_pynn7(self, variable):
        if variable == "spikes":
            data = self._get_spikes()

        (data, ids, sampling_interval) = self._get_recorded_matrix(variable)
        return self.pynn7_format(data, ids, sampling_interval)

    def _get_recorded_matrix(self, variable):
        """ method that contains all the safety checks and gets the recorded
            data from the vertex in matrix format

        :param variable: the variable name to read. supported variable names
            are :'gsyn_exc', 'gsyn_inh', 'v'
        :return: the data
        """
        timer = Timer()
        timer.start_timing()
        data = None
        sim = get_simulator()

        get_simulator().verify_not_running()

        # check that we're in a state to get voltages
        if not isinstance(
                self._population._vertex, AbstractNeuronRecordable):
            raise ConfigurationException(
                "This population has not got the capability to record {}"
                .format(variable))

        if not self._population._vertex.is_recording(variable):
            raise ConfigurationException(
                "This population has not been set to record {}"
                .format(variable))

        if not sim.has_ran:
            logger.warning(
                "The simulation has not yet run, therefore {} cannot"
                " be retrieved, hence the list will be empty".format(
                    variable))
            data = numpy.zeros((0, 3))
            indexes = []
            sampling_interval = self._population._vertex.\
                get_neuron_sampling_interval(variable)
        elif sim.use_virtual_board:
            logger.warning(
                "The simulation is using a virtual machine and so has not"
                " truly ran, hence the list will be empty")
            data = numpy.zeros((0, 3))
            indexes = []
            sampling_interval = self._population._vertex.\
                get_neuron_sampling_interval(variable)
        else:
            # assuming we got here, everything is ok, so we should go get the
            # data
            results = self._population._vertex.get_data(
                variable, sim.no_machine_time_steps, sim.placements,
                sim.graph_mapper, sim.buffer_manager, sim.machine_time_step)
            (data, indexes, sampling_interval) = results

        get_simulator().add_extraction_timing(
            timer.take_sample())
        return (data, indexes, sampling_interval)

    def _get_spikes(self):
        """ How to get spikes from a vertex

        :return: the spikes from a vertex
        """

        # check we're in a state where we can get spikes
        if not isinstance(self._population._vertex, AbstractSpikeRecordable):
            raise ConfigurationException(
                "This population has not got the capability to record spikes")
        if not self._population._vertex.is_recording_spikes():
            raise ConfigurationException(
                "This population has not been set to record spikes")

        sim = get_simulator()
        if not sim.has_ran:
            logger.warning(
                "The simulation has not yet run, therefore spikes cannot "
                "be retrieved, hence the list will be empty")
            return numpy.zeros((0, 2))

        if sim.use_virtual_board:
            logger.warning(
                "The simulation is using a virtual machine and so has not "
                "truly ran, hence the list will be empty")
            return numpy.zeros((0, 2))

        # assuming we got here, everything is OK, so we should go get the
        # spikes
        return self._population._vertex.get_spikes(
            sim.placements, sim.graph_mapper, sim.buffer_manager,
            sim.machine_time_step)

    def _turn_off_all_recording(self):
        """ Turns off recording, is used by a pop saying .record()

        :rtype: None
        """

        # check for standard record
        if isinstance(self._population._vertex, AbstractNeuronRecordable):
            variables = self._population._vertex.get_recordable_variables()
            for variable in variables:
                self._population._vertex.set_recording(variable, False)

        # check for spikes
        elif isinstance(self._population._vertex, AbstractSpikeRecordable):
            self._population._vertex.set_recording_spikes(False)
