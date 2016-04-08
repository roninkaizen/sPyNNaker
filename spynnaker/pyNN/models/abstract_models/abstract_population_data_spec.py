"""
AbstractPopulationDataSpec
"""

# data specable imports
from data_specification.data_specification_generator import \
    DataSpecificationGenerator
from data_specification import utility_calls as dsg_utility_calls

# spynnaker imports
from spynnaker.pyNN.utilities import constants
from spynnaker.pyNN.models.abstract_models.abstract_synaptic_manager \
    import AbstractSynapticManager
from spynnaker.pyNN.models.abstract_models.\
    abstract_partitionable_population_vertex \
    import AbstractPartitionablePopulationVertex

from spinn_front_end_common.abstract_models\
    .abstract_outgoing_edge_same_contiguous_keys_restrictor\
    import AbstractOutgoingEdgeSameContiguousKeysRestrictor

# general imports
import os
import logging
import numpy
import struct
from abc import ABCMeta
from abc import abstractmethod
from six import add_metaclass

logger = logging.getLogger(__name__)


@add_metaclass(ABCMeta)
class AbstractPopulationDataSpec(
        AbstractSynapticManager, AbstractPartitionablePopulationVertex,
        AbstractOutgoingEdgeSameContiguousKeysRestrictor):
    """
    AbstractPopulationDataSpec: provides functioanlity on how neural models
    generate their data spec files
    """

    def __init__(self, binary, n_neurons, label, constraints,
                 max_atoms_per_core, machine_time_step, timescale_factor,
                 spikes_per_second, ring_buffer_sigma, flush_time,
                 master_pop_algorithm=None):
        AbstractSynapticManager.__init__(self, master_pop_algorithm)
        AbstractPartitionablePopulationVertex.__init__(
            self, n_atoms=n_neurons, label=label,
            machine_time_step=machine_time_step,
            timescale_factor=timescale_factor, constraints=constraints,
            max_atoms_per_core=max_atoms_per_core)
        AbstractOutgoingEdgeSameContiguousKeysRestrictor.__init__(self)
        self._binary = binary
        self._spikes_per_second = spikes_per_second
        self._ring_buffer_sigma = ring_buffer_sigma
        self._flush_time = flush_time

        # By default, profiling is disabled
        self.profiler_num_samples = 0

    def _reserve_population_based_memory_regions(
            self, spec, neuron_params_sz, synapse_params_sz,
            master_pop_table_sz, all_syn_block_sz,
            spike_hist_buff_sz, potential_hist_buff_sz, gsyn_hist_buff_sz,
            synapse_dynamics_params_sz):
        """
        Reserve SDRAM space for memory areas:
        1) Area for information on what data to record
        2) Neuron parameter data (will be copied to DTCM by 'C'
           code at start-up)
        3) synapse parameter data (will be copied to DTCM)
        5) Synaptic block look-up table. Translates the start address
           of each block of synapses (copied to DTCM)
        6) Synaptic row data (lives in SDRAM)
        7) Spike history
        8) Neuron potential history
        9) Gsyn value history
        """

        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserve_memory_region(
            region=constants.POPULATION_BASED_REGIONS.SYSTEM.value,
            size=constants.POPULATION_SYSTEM_REGION_BYTES, label='System')
        spec.reserve_memory_region(
            region=constants.POPULATION_BASED_REGIONS.NEURON_PARAMS.value,
            size=neuron_params_sz, label='NeuronParams')
        spec.reserve_memory_region(
            region=constants.POPULATION_BASED_REGIONS.SYNAPSE_PARAMS.value,
            size=synapse_params_sz, label='SynapseParams')

        if master_pop_table_sz > 0:
            spec.reserve_memory_region(
                region=constants.POPULATION_BASED_REGIONS.POPULATION_TABLE
                                                         .value,
                size=master_pop_table_sz, label='PopTable')

        if all_syn_block_sz > 0:
            spec.reserve_memory_region(
                region=constants.POPULATION_BASED_REGIONS.SYNAPTIC_MATRIX
                                                         .value,
                size=all_syn_block_sz, label='SynBlocks')

        if synapse_dynamics_params_sz != 0:
            spec.reserve_memory_region(
                region=constants.POPULATION_BASED_REGIONS.SYNAPSE_DYNAMICS
                                                         .value,
                size=synapse_dynamics_params_sz, label='synapseDynamicsParams')

        if self._record:
            spec.reserve_memory_region(
                region=constants.POPULATION_BASED_REGIONS.SPIKE_HISTORY.value,
                size=spike_hist_buff_sz, label='spikeHistBuffer',
                empty=True)
        if self._record_v:
            spec.reserve_memory_region(
                region=constants.POPULATION_BASED_REGIONS.POTENTIAL_HISTORY
                                                         .value,
                size=potential_hist_buff_sz, label='potHistBuffer',
                empty=True)
        if self._record_gsyn:
            spec.reserve_memory_region(
                region=constants.POPULATION_BASED_REGIONS.GSYN_HISTORY.value,
                size=gsyn_hist_buff_sz, label='gsynHistBuffer',
                empty=True)

        if self.profiler_num_samples != 0:
            spec.reserve_memory_region(
                region=constants.POPULATION_BASED_REGIONS.PROFILING.value,
                size=(8 + (self.profiler_num_samples * 8)), label="profilerRegion")
        else:
            spec.reserve_memory_region(
                region=constants.POPULATION_BASED_REGIONS.PROFILING.value,
                size=4, label="profilerRegion")

    def _write_setup_info(self, spec, spike_history_region_sz,
                          neuron_potential_region_sz, gsyn_region_sz):
        """
        Write information used to control the simulation and gathering of
        results.Currently, this means the flag word used to signal whether
        information on neuron firing and neuron potential is either stored
        locally in a buffer or passed out of the simulation for storage/display
         as the simulation proceeds.

        The format of the information is as follows:
        Word 0: Flags selecting data to be gathered during simulation.
            Bit 0: Record spike history
            Bit 1: Record neuron potential
            Bit 2: Record gsyn values
            Bit 3: Reserved
            Bit 4: Output spike history on-the-fly
            Bit 5: Output neuron potential
            Bit 6: Output spike rate
        """
        # What recording commands were set for the parent pynn_population.py?
        recording_info = 0
        if spike_history_region_sz > 0 and self._record:
            recording_info |= constants.RECORD_SPIKE_BIT
        if neuron_potential_region_sz > 0 and self._record_v:
            recording_info |= constants.RECORD_STATE_BIT
        if gsyn_region_sz > 0 and self._record_gsyn:
            recording_info |= constants.RECORD_GSYN_BIT
        recording_info |= 0xBEEF0000

        # Write this to the system region (to be picked up by the simulation):
        self._write_basic_setup_info(
            spec, constants.POPULATION_BASED_REGIONS.SYSTEM.value)
        spec.write_value(data=recording_info)
        spec.write_value(data=spike_history_region_sz)
        spec.write_value(data=neuron_potential_region_sz)
        spec.write_value(data=gsyn_region_sz)


    def get_profiling_data(self, txrx, placements, graph_mapper):
        # Define profiler time scale
        MS_SCALE = (1.0 / 200032.4)

        # Create a dictionary to hold each sub-vertex's profiling data
        vertex_profiling_data = {}
        subvertices = graph_mapper.get_subvertices_from_vertex(self)
        for subvertex in subvertices:
            placement = placements.get_placement_of_subvertex(subvertex)
            (x, y, p) = placement.x, placement.y, placement.p
            subvertex_slice = graph_mapper.get_subvertex_slice(subvertex)
            lo_atom = subvertex_slice.lo_atom
            logger.debug("Reading spikes from chip {}, {}, core {}, "
                         "lo_atom {}".format(x, y, p, lo_atom))

            # Get the App Data for the core
            app_data_base_address = \
                txrx.get_cpu_information_from_core(x, y, p).user[0]
            # Get the position of the value buffer
            profiling_region_base_address_offset = \
                dsg_utility_calls.get_region_base_address_offset(app_data_base_address,
                                               constants.POPULATION_BASED_REGIONS.PROFILING.value)
            profiling_region_base_address_buf = buffer(txrx.read_memory(
                x, y, profiling_region_base_address_offset, 4))
            profiling_region_base_address = \
                struct.unpack_from("<I", profiling_region_base_address_buf)[0]
            profiling_region_base_address += app_data_base_address

            # Read the profiling data size
            words_written_data =\
                buffer(txrx.read_memory(
                    x, y, profiling_region_base_address + 4, 4))
            words_written = \
                struct.unpack_from("<I", words_written_data)[0]

            # Read the profiling data
            profiling_data = txrx.read_memory(
                x, y, profiling_region_base_address + 8, words_written * 4)

            # Finally read into numpyi
            profiling_samples = numpy.asarray(profiling_data, dtype="uint8").view(dtype="<u4")

            # If there's no data, continue to next vertex
            if len(profiling_samples) == 0:
                print("No samples recorded")
                continue

            # Slice data to seperate times, tags and flags
            sample_times = profiling_samples[::2]
            sample_tags_and_flags = profiling_samples[1::2]

            # Further split the tags and flags word into seperate arrays of tags and flags
            sample_tags = numpy.bitwise_and(sample_tags_and_flags, 0x7FFFFFFF)
            sample_flags = numpy.right_shift(sample_tags_and_flags, 31)

            # Find indices of samples relating to entries and exits
            sample_entry_indices = numpy.where(sample_flags == 1)
            sample_exit_indices = numpy.where(sample_flags == 0)

            # Convert count-down times to count up times from 1st sample
            sample_times = numpy.subtract(sample_times[0], sample_times)
            sample_times_ms = numpy.multiply(sample_times, MS_SCALE, dtype=numpy.float)

            # Slice tags and times into entry and exits
            entry_tags = sample_tags[sample_entry_indices]
            entry_times_ms = sample_times_ms[sample_entry_indices]
            exit_tags = sample_tags[sample_exit_indices]
            exit_times_ms = sample_times_ms[sample_exit_indices]

            # Loop through unique tags
            tag_dictionary = {}
            unique_tags = numpy.unique(sample_tags)
            for tag in unique_tags:
                # Get indices where these tags occur
                tag_entry_indices = numpy.where(entry_tags == tag)
                tag_exit_indices = numpy.where(exit_tags == tag)

                # Use these to get subset for this tag
                tag_entry_times_ms = entry_times_ms[tag_entry_indices]
                tag_exit_times_ms = exit_times_ms[tag_exit_indices]

                # If the first exit is before the first
                # Entry, add a dummy entry at beginning
                if tag_exit_times_ms[0] < tag_entry_times_ms[0]:
                    print "WARNING: profile starts mid-tag"
                    tag_entry_times_ms = numpy.append(0.0, tag_entry_times_ms)

                if len(tag_entry_times_ms) > len(tag_exit_times_ms):
                    print "WARNING: profile finishes mid-tag"
                    tag_entry_times_ms = tag_entry_times_ms[:len(tag_exit_times_ms)-len(tag_entry_times_ms)]

                # Subtract entry times from exit times to get durations of each call
                tag_durations_ms = numpy.subtract(tag_exit_times_ms, tag_entry_times_ms)

                # Add entry times and durations to dictionary
                tag_dictionary[tag] = (tag_entry_times_ms, tag_durations_ms)

            # Stick tag dictionary in profiling data
            vertex_profiling_data[(subvertex_slice.lo_atom, subvertex_slice.hi_atom)] = tag_dictionary

        return vertex_profiling_data

    @abstractmethod
    def get_parameters(self):
        """ Get any per-neuron parameters for a model
        """

    @abstractmethod
    def get_global_parameters(self):
        """ Get any global parameters for a model
        """

    def _write_neuron_parameters(
            self, spec, key, subvertex, vertex_slice):

        n_atoms = (vertex_slice.hi_atom - vertex_slice.lo_atom) + 1
        spec.comment("\nWriting Neuron Parameters for {} "
                     "Neurons:\n".format(n_atoms))

        # Set the focus to the memory region 2 (neuron parameters):
        spec.switch_write_focus(
            region=constants.POPULATION_BASED_REGIONS.NEURON_PARAMS.value)

        # Write header info to the memory region:

        # Write whether the key is to be used, and then the key, or 0 if it
        # isn't to be used
        if key is None:
            spec.write_value(data=0)
            spec.write_value(data=0)
        else:
            spec.write_value(data=1)
            spec.write_value(data=key)

        # Write the number of neurons in the block:
        spec.write_value(data=n_atoms)

        # Write flush time
        spec.write_value(data=0xFFFFFFFF if self._flush_time is None else self._flush_time)

        # Write the global parameters
        global_params = self.get_global_parameters()
        for param in global_params:
            spec.write_value(data=param.get_value(),
                             data_type=param.get_dataspec_datatype())

        # TODO: NEEDS TO BE LOOKED AT PROPERLY
        # Create loop over number of neurons:
        params = self.get_parameters()
        for atom in range(vertex_slice.lo_atom, vertex_slice.hi_atom + 1):
            # Process the parameters

            # noinspection PyTypeChecker
            for param in params:
                value = param.get_value()
                if hasattr(value, "__len__"):
                    if len(value) > 1:
                        if len(value) <= atom:
                            raise Exception(
                                "Not enough parameters have been specified"
                                " for parameter of population {}".format(
                                    self.label))
                        value = value[atom]
                    else:
                        value = value[0]

                datatype = param.get_dataspec_datatype()

                spec.write_value(data=value, data_type=datatype)
        # End the loop over the neurons:

    def generate_data_spec(
            self, subvertex, placement, subgraph, graph, routing_info,
            hostname, graph_mapper, report_folder, ip_tags, reverse_ip_tags,
            write_text_specs, application_run_time_folder):
        """
        Model-specific construction of the data blocks necessary to
        build a group of IF_curr_exp neurons resident on a single core.
        :param subvertex:
        :param placement:
        :param subgraph:
        :param graph:
        :param routing_info:
        :param hostname:
        :param graph_mapper:
        :param report_folder:
        :param ip_tags:
        :param reverse_ip_tags:
        :param write_text_specs:
        :param application_run_time_folder:
        :return:
        """
        # Create new DataSpec for this processor:
        data_writer, report_writer = \
            self.get_data_spec_file_writers(
                placement.x, placement.y, placement.p, hostname, report_folder,
                write_text_specs, application_run_time_folder)

        spec = DataSpecificationGenerator(data_writer, report_writer)

        spec.comment("\n*** Spec for block of {} neurons ***\n"
                     .format(self.model_name))

        vertex_slice = graph_mapper.get_subvertex_slice(subvertex)

        # Calculate the size of the tables to be reserved in SDRAM:
        vertex_in_edges = graph.incoming_edges_to_vertex(self)
        neuron_params_sz = self.get_neuron_params_size(vertex_slice)
        synapse_params_sz = self.get_synapse_parameter_size(vertex_slice)
        master_pop_table_sz = self.get_population_table_size(vertex_slice,
                                                             vertex_in_edges)

        subvert_in_edges = subgraph.incoming_subedges_from_subvertex(subvertex)
        all_syn_block_sz = self.get_exact_synaptic_block_memory_size(
            graph_mapper, subvert_in_edges)

        spike_hist_buff_sz = self.get_spike_buffer_size(vertex_slice)
        potential_hist_buff_sz = self.get_v_buffer_size(vertex_slice)
        gsyn_hist_buff_sz = self.get_g_syn_buffer_size(vertex_slice)
        synapse_dynamics_region_sz = self.get_synapse_dynamics_parameter_size(
            vertex_in_edges)

        # Declare random number generators and distributions:
        # TODO add random distrubtion stuff
        # self.write_random_distribution_declarations(spec)

        ring_buffer_shifts = self.get_ring_buffer_to_input_left_shifts(
            subvertex, subgraph, graph_mapper, self._spikes_per_second,
            self._machine_time_step, self._ring_buffer_sigma)

        weight_scales = [self.get_weight_scale(r) for r in ring_buffer_shifts]

        if logger.isEnabledFor(logging.DEBUG):
            for t, r, w in zip(self.get_synapse_targets(), ring_buffer_shifts,
                               weight_scales):
                logger.debug(
                    "Synapse type:%s - Ring buffer shift:%d, Max weight:%f"
                    % (t, r, w))

        # update projections for future use
        in_partitioned_edges = \
            subgraph.incoming_subedges_from_subvertex(subvertex)
        for partitioned_edge in in_partitioned_edges:
            partitioned_edge.weight_scales_setter(weight_scales)

        # Construct the data images needed for the Neuron:
        self._reserve_population_based_memory_regions(
            spec, neuron_params_sz, synapse_params_sz,
            master_pop_table_sz, all_syn_block_sz,
            spike_hist_buff_sz, potential_hist_buff_sz, gsyn_hist_buff_sz,
            synapse_dynamics_region_sz)

        # Remove extension to get application name
        self._write_setup_info(spec, spike_hist_buff_sz,
                               potential_hist_buff_sz, gsyn_hist_buff_sz)

        # Write profiler info
        spec.switch_write_focus(region=constants.POPULATION_BASED_REGIONS.PROFILING.value)
        spec.write_value(data=self.profiler_num_samples)

        # Every outgoing edge from this vertex should have the same key
        key = None
        if len(subgraph.outgoing_subedges_from_subvertex(subvertex)) > 0:
            keys_and_masks = routing_info.get_keys_and_masks_from_subedge(
                subgraph.outgoing_subedges_from_subvertex(subvertex)[0])

            # NOTE: using the first key assigned as the key.  Should in future
            # get the list of keys and use one per neuron, to allow arbitrary
            # key and mask assignments
            key = keys_and_masks[0].key

        self._write_neuron_parameters(spec, key, subvertex, vertex_slice)

        self.write_synapse_parameters(spec, subvertex, vertex_slice)
        spec.write_array(ring_buffer_shifts)

        self.write_synaptic_matrix_and_master_population_table(
            spec, subvertex, all_syn_block_sz, weight_scales,
            constants.POPULATION_BASED_REGIONS.POPULATION_TABLE.value,
            constants.POPULATION_BASED_REGIONS.SYNAPTIC_MATRIX.value,
            routing_info, graph_mapper, subgraph)

        self.write_synapse_dynamics_parameters(
            spec, self._machine_time_step,
            constants.POPULATION_BASED_REGIONS.SYNAPSE_DYNAMICS.value,
            weight_scales)

        in_subedges = subgraph.incoming_subedges_from_subvertex(subvertex)
        for subedge in in_subedges:
            subedge.free_sublist()

        # End the writing of this specification:
        spec.end_specification()
        data_writer.close()

    # inherited from data specable vertex
    def get_binary_file_name(self):
        """

        :return:
        """

        # Split binary name into title and extension
        binary_title, binary_extension = os.path.splitext(self._binary)

        # If we have an STDP mechanism, add it's executable suffic to title
        if self._stdp_mechanism is not None:
            binary_title = \
                binary_title + "_" + \
                self._stdp_mechanism.get_vertex_executable_suffix()

        # Reunite title and extension and return
        return binary_title + binary_extension

    def is_data_specable(self):
        """
        helper method for isinstance
        :return:
        """
        return True
