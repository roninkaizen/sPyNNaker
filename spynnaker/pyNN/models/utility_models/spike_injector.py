import logging

from spinn_utilities.overrides import overrides
from pacman.model.constraints.key_allocator_constraints \
    import ContiguousKeyRangeContraint

from spinn_front_end_common.abstract_models \
    import AbstractProvidesOutgoingPartitionConstraints
from spinn_front_end_common.utility_models import ReverseIpTagMultiCastSource
from spinn_front_end_common.utilities.globals_variables import get_simulator

from spynnaker.pyNN.models.common \
    import AbstractSpikeRecordable, EIEIOSpikeRecorder, \
    SimplePopulationSettable

logger = logging.getLogger(__name__)


class SpikeInjector(ReverseIpTagMultiCastSource,
                    AbstractProvidesOutgoingPartitionConstraints,
                    AbstractSpikeRecordable, SimplePopulationSettable):
    """ An Injector of Spikes for PyNN populations.  This only allows the user\
        to specify the virtual_key of the population to identify the population
    """
    __slots__ = [
        "_buffer_size_before_receive",
        "_receive_port",
        "_requires_mapping",
        "_spike_buffer_max_size",
        "_spike_recorder",
        "_time_between_requests",
        "_virtual_key"]

    default_parameters = {
        'label': "spikeInjector", 'port': None, 'virtual_key': None}

    SPIKE_RECORDING_REGION_ID = 0

    def __init__(
            self, n_neurons, label=default_parameters['label'],
            port=default_parameters['port'],
            virtual_key=default_parameters['virtual_key'],
            spike_buffer_max_size=None, buffer_size_before_receive=None,
            time_between_requests=None, buffer_notification_ip_address=None,
            buffer_notification_port=None):
        # pylint: disable=too-many-arguments
        config = get_simulator().config
        if buffer_notification_ip_address is None:
            buffer_notification_ip_address = config.get(
                "Buffers", "receive_buffer_host")
        if buffer_notification_port is None:
            buffer_notification_port = config.get_int(
                "Buffers", "receive_buffer_port")

        super(SpikeInjector, self).__init__(
            n_keys=n_neurons, label=label, receive_port=port,
            virtual_key=virtual_key, reserve_reverse_ip_tag=True,
            buffer_notification_ip_address=buffer_notification_ip_address,
            buffer_notification_port=buffer_notification_port)

        # Set up for recording
        self._spike_recorder = EIEIOSpikeRecorder()
        self._spike_buffer_max_size = spike_buffer_max_size
        if spike_buffer_max_size is None:
            self._spike_buffer_max_size = config.getint(
                "Buffers", "spike_buffer_size")
        self._buffer_size_before_receive = buffer_size_before_receive
        if buffer_size_before_receive is None:
            self._buffer_size_before_receive = config.getint(
                "Buffers", "buffer_size_before_receive")
        self._time_between_requests = time_between_requests
        if time_between_requests is None:
            self._time_between_requests = config.getint(
                "Buffers", "time_between_requests")

    @property
    def port(self):
        return self._receive_port

    @port.setter
    def port(self, port):
        self._receive_port = port

    @property
    def virtual_key(self):
        return self._virtual_key

    @virtual_key.setter
    def virtual_key(self, virtual_key):
        self._virtual_key = virtual_key

    @overrides(AbstractSpikeRecordable.is_recording_spikes)
    def is_recording_spikes(self):
        return self._spike_recorder.record

    @overrides(AbstractSpikeRecordable.set_recording_spikes)
    def set_recording_spikes(
            self, new_state=True, sampling_interval=None, indexes=None):
        if sampling_interval is not None:
            logger.warning("Sampling interval currently not supported "
                           "so being ignored")
        if indexes is not None:
            logger.warning("Indexes currently not supported "
                           "so being ignored")
        self.enable_recording(
            self._spike_buffer_max_size, self._buffer_size_before_receive,
            self._time_between_requests)
        self._requires_mapping = not self._spike_recorder.record
        self._spike_recorder.record = new_state

    @overrides(AbstractSpikeRecordable.get_spikes_sampling_interval)
    def get_spikes_sampling_interval(self):
        return get_simulator().machine_time_step

    @overrides(AbstractSpikeRecordable.get_spikes)
    def get_spikes(
            self, placements, graph_mapper, buffer_manager, machine_time_step):
        return self._spike_recorder.get_spikes(
            self.label, buffer_manager,
            SpikeInjector.SPIKE_RECORDING_REGION_ID,
            placements, graph_mapper, self,
            lambda vertex:
                vertex.virtual_key
                if vertex.virtual_key is not None
                else 0,
            machine_time_step)

    @overrides(AbstractSpikeRecordable.clear_spike_recording)
    def clear_spike_recording(self, buffer_manager, placements, graph_mapper):
        machine_vertices = graph_mapper.get_machine_vertices(self)
        for machine_vertex in machine_vertices:
            placement = placements.get_placement_of_vertex(machine_vertex)
            buffer_manager.clear_recorded_data(
                placement.x, placement.y, placement.p,
                SpikeInjector.SPIKE_RECORDING_REGION_ID)

    @overrides(AbstractProvidesOutgoingPartitionConstraints.
               get_outgoing_partition_constraints)
    def get_outgoing_partition_constraints(self, partition):
        constraints = ReverseIpTagMultiCastSource\
            .get_outgoing_partition_constraints(self, partition)
        constraints.append(ContiguousKeyRangeContraint())
        return constraints

    def describe(self):
        """
        Returns a human-readable description of the cell or synapse type.

        The output may be customised by specifying a different template
        together with an associated template engine
        (see ``pyNN.descriptions``).

        If template is None, then a dictionary containing the template context
        will be returned.
        """

        parameters = dict()
        for parameter_name in self.default_parameters:
            parameters[parameter_name] = self.get_value(parameter_name)

        context = {
            "name": "SpikeInjector",
            "default_parameters": self.default_parameters,
            "default_initial_values": self.default_parameters,
            "parameters": parameters,
        }
        return context
