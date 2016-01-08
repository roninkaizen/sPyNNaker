from six import add_metaclass
from abc import ABCMeta
from abc import abstractmethod


@add_metaclass(ABCMeta)
class AbstractVRecordable(object):
    """ Indicates that membrane voltage can be recorded from this object
    """

    @abstractmethod
    def is_recording_v(self):
        """ Determines if v is being recorded

        :return: True if v are being recorded, False otherwise
        :rtype: bool
        """

    @abstractmethod
    def set_recording_v(self, schedule=[]):
        """ Sets v to being recorded

        :param schedule: a list of tuples of start and end times in ms between\
                which the recording will take place.  The last end time can be\
                None, in which case recording will end at the end of\
                simulation.  If the schedule is empty, recording will be done\
                for the whole simulation.
        """

    @abstractmethod
    def get_v(self, n_machine_time_steps, placements, graph_mapper,
              buffer_manager):
        """ Get the recorded v from the object

        :param n_machine_time_steps: the number of timer ticks that will\
                be executed on the machine.
        :param placements: The placements of the graph
        :param graph_mapper: The mapper between subvertices and vertices
        :param buffer_manager: the buffer manager object
        :return: A numpy array of 3-element arrays of (neuron_id, time, v)\
                ordered by time
        """
