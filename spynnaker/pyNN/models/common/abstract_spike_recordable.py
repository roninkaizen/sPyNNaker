from six import add_metaclass
from abc import ABCMeta
from abc import abstractmethod


@add_metaclass(ABCMeta)
class AbstractSpikeRecordable(object):
    """ Indicates that spikes can be recorded from this object
    """

    @abstractmethod
    def is_recording_spikes(self):
        """ Determines if spikes are being recorded

        :return: True if spikes are being recorded, False otherwise
        :rtype: bool
        """

    @abstractmethod
    def set_recording_spikes(self, schedule=None):
        """ Sets spikes to being recorded

        :param schedule: a list of tuples of start and end times in ms between\
                which the recording will take place.  The last end time can be\
                None, in which case recording will end at the end of\
                simulation.  If the schedule is empty, recording will be done\
                for the whole simulation.
        """

    @abstractmethod
    def get_spikes(self, placements, graph_mapper, buffer_manager):
        """ Get the recorded spikes from the object
        :param placements: the placements object
        :param graph_mapper: the graph mapper object
        :param buffer_manager: the buffer manager object
        :return: A numpy array of 2-element arrays of (neuron_id, time)\
                ordered by time
        """
