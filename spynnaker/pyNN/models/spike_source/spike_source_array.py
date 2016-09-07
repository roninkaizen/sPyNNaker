# spynnaker imports
from spynnaker.pyNN.models.abstract_models.abstract_groupable import \
    AbstractGroupable
from spynnaker.pyNN.utilities import constants
from spynnaker.pyNN.models.neuron_cell import RecordingType
from spinn_front_end_common.abstract_models.abstract_changable_after_run \
    import AbstractChangableAfterRun
from spynnaker.pyNN.models.common.eieio_spike_recorder \
    import EIEIOSpikeRecorder
from spynnaker.pyNN.models.common.abstract_spike_recordable \
    import AbstractSpikeRecordable
from spynnaker.pyNN.utilities.conf import config


# spinn front end common imports
from spinn_front_end_common.abstract_models.\
    abstract_provides_outgoing_partition_constraints import \
    AbstractProvidesOutgoingPartitionConstraints
from spinn_front_end_common.utility_models.reverse_ip_tag_multi_cast_source \
    import ReverseIpTagMultiCastSource
from spinn_front_end_common.utilities import constants as \
    front_end_common_constants
from spinn_front_end_common.utilities import exceptions
from spinn_front_end_common.utility_models\
    .reverse_ip_tag_multicast_source_machine_vertex \
    import ReverseIPTagMulticastSourceMachineVertex

from pacman.model.decorators.overrides import overrides

# general imports
import logging
import sys

logger = logging.getLogger(__name__)


class SpikeSourceArray(
        ReverseIpTagMultiCastSource,
        AbstractSpikeRecordable, AbstractGroupable,
        AbstractChangableAfterRun, AbstractChangableAfterRun):
    """ Model for play back of spikes
    """

    _model_based_max_atoms_per_core = sys.maxint
    SPACE_BEFORE_NOTIFICATION = 640

    population_parameters = {
        'machine_time_step', 'time_scale_factor', 'ip_address', 'port',
        'space_before_notification', 'spike_recorder_buffer_size',
        'max_on_chip_memory_usage_for_spikes_in_bytes',
        'buffer_size_before_receive', 'board_address', 'tag'
    }

    model_name = "SpikeSourceArray"

    @staticmethod
    def default_parameters(_):
        return {'spike_times': None}

    @staticmethod
    def fixed_parameters(_):
        return {}

    @staticmethod
    def state_variables(_):
        return list()

    @staticmethod
    def is_array_parameters(_):
        return {'spike_times'}

    @staticmethod
    def recording_types(_):
        return [RecordingType.SPIKES]

    def __init__(self, bag_of_neurons, label="SpikeSourceArray",
                 constraints=None):

        AbstractGroupable.__init__(self)

        # assume all atoms have the same parameters, so can look at first
        # determine ip address
        ip_address = bag_of_neurons[0].get_population_parameter('ip_address')
        self._ip_address = ip_address
        if ip_address is None:
            self._ip_address = config.get("Buffers", "receive_buffer_host")
            for atom in bag_of_neurons:
                atom.set_population_parameter('ip_address', self._ip_address)

        # determine port
        self._port = bag_of_neurons[0].get_population_parameter('port')
        if self._port is None:
            self._port = config.getint("Buffers", "receive_buffer_port")
            for atom in bag_of_neurons:
                atom.set_population_parameter('port', self._port)

        # determine space_before_notification
        space_before_notification = bag_of_neurons[0].get_population_parameter(
            'space_before_notification')
        if space_before_notification is None:
            space_before_notification = self.SPACE_BEFORE_NOTIFICATION
            for atom in bag_of_neurons:
                atom.set_population_parameter(
                    'space_before_notification', space_before_notification)

        # determine spike_recorder_buffer_size
        spike_recorder_buffer_size = bag_of_neurons[0].\
            get_population_parameter('spike_recorder_buffer_size')
        if spike_recorder_buffer_size is None:
            spike_recorder_buffer_size = \
                (constants.EIEIO_SPIKE_BUFFER_SIZE_BUFFERING_OUT)
            for atom in bag_of_neurons:
                atom.set_population_parameter(
                    'spike_recorder_buffer_size', spike_recorder_buffer_size)

        # determine max_on_chip_memory_usage_for_spikes_in_bytes
        max_on_chip_memory_usage_for_spikes_in_bytes = bag_of_neurons[0].\
            get_population_parameter(
            'max_on_chip_memory_usage_for_spikes_in_bytes')
        if max_on_chip_memory_usage_for_spikes_in_bytes is None:
            max_on_chip_memory_usage_for_spikes_in_bytes = \
                (constants.SPIKE_BUFFER_SIZE_BUFFERING_IN)
            for atom in bag_of_neurons:
                atom.set_population_parameter(
                    'max_on_chip_memory_usage_for_spikes_in_bytes',
                    max_on_chip_memory_usage_for_spikes_in_bytes)

        # determine buffer_size_before_receive
        buffer_size_before_receive = bag_of_neurons[0].\
            get_population_parameter('buffer_size_before_receive')
        if buffer_size_before_receive is None:
            buffer_size_before_receive = \
                (constants.EIEIO_BUFFER_SIZE_BEFORE_RECEIVE)
            for atom in bag_of_neurons:
                atom.set_population_parameter(
                    'buffer_size_before_receive', buffer_size_before_receive)

        # determine board address
        board_address = \
            bag_of_neurons[0].get_population_parameter('board_address')

        # determine tag
        tag = bag_of_neurons[0].get_population_parameter('tag')

        # hard code this for the time being
        spike_times = []

        # store the atoms for future processing
        self._atoms = bag_of_neurons
        self._mapping = None

        # get hard coded values
        self._minimum_sdram_for_buffering = config.getint(
            "Buffers", "minimum_buffer_sdram")
        self._using_auto_pause_and_resume = config.getboolean(
            "Buffers", "use_auto_pause_and_resume")

        ReverseIpTagMultiCastSource.__init__(
            self, n_keys=len(bag_of_neurons), label=label,
            constraints=constraints,
            max_atoms_per_core=(SpikeSourceArray.
                                _model_based_max_atoms_per_core),
            board_address=board_address,
            receive_port=None, receive_sdp_port=None, receive_tag=None,
            virtual_key=None, prefix=None, prefix_type=None, check_keys=False,
            send_buffer_times=spike_times,
            send_buffer_partition_id=constants.SPIKE_PARTITION_ID,
            send_buffer_max_space=max_on_chip_memory_usage_for_spikes_in_bytes,
            send_buffer_space_before_notify=space_before_notification,
            send_buffer_notification_ip_address=self._ip_address,
            send_buffer_notification_port=self._port,
            send_buffer_notification_tag=tag)

        AbstractSpikeRecordable.__init__(self)
        AbstractProvidesOutgoingPartitionConstraints.__init__(self)
        AbstractChangableAfterRun.__init__(self)

        # handle recording
        self._spike_recorder = EIEIOSpikeRecorder()
        self._spike_recorder_buffer_size = spike_recorder_buffer_size
        self._buffer_size_before_receive = buffer_size_before_receive

        # Keep track of any previously generated buffers
        self._send_buffers = dict()
        self._spike_recording_region_size = None
        self._machine_vertices = list()

        # used for reset and rerun
        self._requires_mapping = True
        self._last_runtime_position = 0

        self._max_on_chip_memory_usage_for_spikes = \
            max_on_chip_memory_usage_for_spikes_in_bytes
        self._space_before_notification = space_before_notification
        if self._max_on_chip_memory_usage_for_spikes is None:
            self._max_on_chip_memory_usage_for_spikes = \
                front_end_common_constants.MAX_SIZE_OF_BUFFERED_REGION_ON_CHIP

        # check the values do not conflict with chip memory limit
        if self._max_on_chip_memory_usage_for_spikes < 0:
            raise exceptions.ConfigurationException(
                "The memory usage on chip is either beyond what is supportable"
                " on the spinnaker board being supported or you have requested"
                " a negative value for a memory usage. Please correct and"
                " try again")

        if (self._max_on_chip_memory_usage_for_spikes <
                self._space_before_notification):
            self._space_before_notification =\
                self._max_on_chip_memory_usage_for_spikes

        # check for recording requirements
        is_recording_spikes = False
        for atom in bag_of_neurons:
            if atom.is_recording(RecordingType.SPIKES):
                is_recording_spikes = True
        if is_recording_spikes:
            self.set_recording_spikes()

    @staticmethod
    def create_vertex(bag_of_neurons, population_parameters):
        params = dict(population_parameters)
        params['bag_of_neurons'] = bag_of_neurons
        vertex = SpikeSourceArray(**params)
        return vertex

    def set_mapping(self, mapping):
        total_spikes_times = list()
        mapping = mapping[self]
        for (_, start, end) in mapping:
            for atom_id in range(start, end):
                total_spikes_times.append(
                    self._atoms[atom_id].get("spike_times"))
        self.spike_times = total_spikes_times
        self._mapping = mapping

    @property
    def vertex_to_pop_mapping(self):
        return self._mapping

    @property
    @overrides(AbstractChangableAfterRun.requires_mapping)
    def requires_mapping(self):
        return self._requires_mapping

    @overrides(AbstractChangableAfterRun.mark_no_changes)
    def mark_no_changes(self):
        self._requires_mapping = False

    def requires_remapping_for_change(self, parameter, old_value, new_value):
        if parameter == "spike_times":
            return False
        else:
            return True

    @property
    def spike_times(self):
        """ The spike times of the spike source array
        :return:
        """
        return self.send_buffer_times

    @spike_times.setter
    def spike_times(self, spike_times):
        """ Set the spike source array's spike times. Not an extend, but an\
            actual change
        :param spike_times:
        :return:
        """
        self.send_buffer_times = spike_times

    @overrides(AbstractSpikeRecordable.is_recording_spikes)
    def is_recording_spikes(self):
        return self._spike_recorder.record

    @overrides(AbstractSpikeRecordable.set_recording_spikes)
    def set_recording_spikes(self):
        self.enable_recording(
            self._ip_address, self._port, self._board_address,
            self._send_buffer_notification_tag,
            self._spike_recorder_buffer_size,
            self._buffer_size_before_receive,
            self._minimum_sdram_for_buffering,
            self._using_auto_pause_and_resume)

        self._requires_mapping = not self._spike_recorder.record
        self._spike_recorder.record = True

    @overrides(AbstractSpikeRecordable.get_spikes)
    def get_spikes(
            self, placements, graph_mapper, buffer_manager, machine_time_step):
        return self._spike_recorder.get_spikes(
            self.label, buffer_manager,
            (ReverseIPTagMulticastSourceMachineVertex.
             _REGIONS.RECORDING_BUFFER.value),
            (ReverseIPTagMulticastSourceMachineVertex.
             _REGIONS.RECORDING_BUFFER_STATE.value),
            placements, graph_mapper, self,
            lambda vertex:
                vertex.virtual_key if vertex.virtual_key is not None
                else 0,
            machine_time_step)

    @staticmethod
    def set_model_max_atoms_per_core(new_value):
        SpikeSourceArray._model_based_max_atoms_per_core = new_value

    @staticmethod
    def get_max_atoms_per_core():
        return SpikeSourceArray._model_based_max_atoms_per_core
