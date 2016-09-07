from spynnaker.pyNN.utilities import utility_calls
from spynnaker.pyNN.models.neuron_cell import RecordingType
from spynnaker.pyNN.models.neuron.input_types.input_type_conductance import \
    InputTypeConductance
from abc import abstractmethod
from ordered_set import OrderedSet
from spynnaker.pyNN.models.common.abstract_v_recordable \
    import AbstractVRecordable
from spynnaker.pyNN.models.common.abstract_spike_recordable \
    import AbstractSpikeRecordable
from spynnaker.pyNN.models.common.abstract_gsyn_recordable \
    import AbstractGSynRecordable

from spinn_front_end_common.utilities import exceptions

from pyNN import random

from six import add_metaclass
from abc import ABCMeta

import numpy
import logging

logger = logging.getLogger(__name__)


@add_metaclass(ABCMeta)
class BasePyNNContainer(object):
    """ A common base container for PyNN cell containers
    """

    def __init__(
            self, spinnaker, populations=None, population_slices=None,
            label=None):
        """

        :param populations:\
            A list of populations contained within the container
        :param population_slices:\
            The slice of each population within this container, in the same\
            order as the list of populations
        """
        self._spinnaker = spinnaker
        self._populations = list()
        self._population_slices = list()
        self._populations_start_and_end = list()
        self._cells = OrderedSet()
        self._label = label
        if populations is not None:
            for population, population_slice in zip(
                    populations, population_slices):
                self._add_container(population, population_slice)

    def _add_container(self, population, population_slice):
        self._populations.append(population)
        self._population_slices.append(population_slice)
        start = len(self._cells)
        for cell in population._cells:
            self._cells.add(cell)
        end = len(self._cells)
        self._populations_start_and_end.append((start, end))

    @abstractmethod
    def __getitem__(self, indices):
        """ Get a single cell or a container of cells
        """

    def __iter__(self):
        """ An iterator over the cells in the container
        """
        return iter(self._cells)

    def __len__(self):
        """ The number of cells in the container
        """
        return len(self._cells)

    @staticmethod
    def _filter_values_by_bool_array(n_cells, vertex_filter, start, values):

        # Filter the values to only contain the relevant cells
        # If the slice is all of the indices, avoid this filter
        values_slice = values
        if not all(vertex_filter):

            # Get the indices of the cells in the slice of the vertex
            vertex_indices = numpy.arange(n_cells)[vertex_filter]

            # numpy.in1d returns an array which is True if the cell index
            # is in vertex_indices, or False if not
            values_slice = values[numpy.in1d(values[:, 0], vertex_indices)]

            # Process the neuron ids to the appropriate positions based on
            # the slice
            # numpy.digitize returns the index of the cell index within
            # vertex_indices; thus the first cell id goes to 0, the second
            # id to 1 etc.
            values_slice[:, 0] = numpy.digitize(
                values_slice[:, 0], vertex_indices, right=True)

        # Add the start index to the cell index if required
        if start != 0:
            values_slice[:, 0] += start

        return values_slice

    @staticmethod
    def _filter_values_by_indices(n_atoms, vertex_indices, start, values):

        # Filter the values to only contain the relevant cells
        # If the slice is all of the indices, avoid this filter
        values_slice = values
        if len(vertex_indices) != n_atoms:

            # numpy.in1d returns an array which is True if the cell index
            # is in vertex_indices, or False if not
            values_slice = values[numpy.in1d(values[:, 0], vertex_indices)]

            # Process the neuron ids to the appropriate positions based on
            # the slice
            # numpy.digitize returns the index of the cell index within
            # vertex_indices; thus the first cell id goes to 0, the second
            # id to 1 etc.
            values_slice[:, 0] = numpy.digitize(
                values_slice[:, 0], vertex_indices, right=True)

        # Add the start index to the cell index if required
        if start != 0:
            values_slice[:, 0] += start

        return values_slice

    @property
    def all_cells(self):
        return self._cells

    def all(self):
        return iter(self._cells)

    def can_record(self, variable):
        """ Determine whether `variable` can be recorded from this population.

        :param variable: the parameter name to check recording for
        """
        return any([cell.can_record(variable) for cell in self._cells])

    @property
    def conductance_based(self):
        return all([
            hasattr(pop.celltype, "input_type") and
            issubclass(pop.celltype.input_type, InputTypeConductance)
            for pop in self._populations
        ])

    def get(self, parameter_name, gather=False):
        """ Get the value of a parameter for every cell
        """
        return [cell[parameter_name] for cell in self._cells]

    def _get_recorded_values(
            self, name, n_elements, vertex_type, recording_type, get_data):
        """ Get the values of a recorded parameter for the cells in this\
            container

        :param name: The name of the data being read
        :param n_elements: The number of elements in each value of the data
        :param vertex_type: The type of the vertex for this recorded data
        :param recording_type: The cell recording type to look for
        :param get_data: The function that actually gets the data
        """

        if not self._spinnaker.has_ran:
            logger.warn(
                "The simulation has not yet run, therefore {} cannot"
                " be retrieved, hence the list will be empty".format(name))
            return numpy.zeros((0, n_elements))

        if self._spinnaker.use_virtual_board:
            logger.warn(
                "The simulation is using a virtual machine and so has not"
                " truly ran, hence the list will be empty")
            return numpy.zeros((0, n_elements))

        # Keep track of data already received for a given vertex
        data_for_vertex = dict()

        # Go through the populations and get data for each
        all_data = list()
        for population, population_slice, (start, _) in zip(
                self._populations, self._population_slices,
                self._populations_start_and_end):

            # Get the part of the population that is relevant
            indices_in_slice = numpy.arange(len(population))[
                population_slice]

            # Go through the vertices for the population
            for (vertex, vertex_start, _, population_start,
                    population_end) in population._vertices:

                if isinstance(vertex, vertex_type):

                    # find cells that are recording this population within
                    # the vertex
                    population_indices_in_vertex = indices_in_slice[
                        (indices_in_slice >= population_start) and
                        (indices_in_slice <= population_end)]
                    population_indices_in_vertex = [
                        index for index in population_indices_in_vertex
                        if population._cells[index].is_recording(
                            recording_type)
                    ]
                    vertex_indices = (
                        population_indices_in_vertex +
                        (vertex_start - population_start)
                    )

                    # Only get the data if one of the atoms is recording
                    if len(vertex_indices) > 0:

                        # extract data from the vertex
                        vertex_data = data_for_vertex.get(vertex, None)
                        if vertex_data is None:
                            vertex_data = get_data(vertex)

                        all_data.append(self._filter_values_by_indices(
                            vertex.n_atoms, vertex_indices, start,
                            vertex_data))

        if len(all_data) == 0:
            logger.warn(
                "Recording of {} was not enabled for any cells in this"
                " container".format(name))
            return numpy.zeros((0, n_elements))

        return numpy.vstack(all_data)

    def get_gsyn(self, gather=True, compatible_output=True):
        """ Get the gsyn for all cells in this container

        :param gather:\
            not used - inserted to match PyNN specs
        :type gather: bool
        :param compatible_output:\
            not used - inserted to match PyNN specs
        :type compatible_output: bool
        :return: 4 dimensional numpy array
        """

        return self._get_recorded_values(
            "conductance", 4, AbstractGSynRecordable, RecordingType.GSYN,
            lambda vertex: vertex.get_gsyn(
                self._spinnaker.no_machine_time_steps,
                self._spinnaker.placements, self._spinnaker.graph_mapper,
                self._spinnaker.buffer_manager,
                self._spinnaker.machine_time_step))

    def getSpikes(self, gather=True, compatible_output=True):
        """ Get the spikes for all cells in the container

        :param gather:\
            not used - inserted to match PyNN specs
        :type gather: bool
        :param compatible_output:\
            not used - inserted to match PyNN specs
        :type compatible_output: bool
        :return: returns 2-column numpy array
        """
        return self._get_recorded_values(
            "spikes", 2, AbstractSpikeRecordable, RecordingType.SPIKES,
            lambda vertex: vertex.get_spikes(
                self._spinnaker.placements, self._spinnaker.graph_mapper,
                self._spinnaker.buffer_manager,
                self._spinnaker.machine_time_step))

    def get_spike_counts(self, gather=True):
        """ Get the spike counts of all the cells in this container

        :param gather: gather means nothing to Spinnaker
        """
        spike_count_total = dict()
        for container, (start, _) in zip(
                self._containers, self._start_and_end):
            spike_count = container.get_spike_counts(gather)

            # Put the corrected value in the dict
            for n_index, count in spike_count.iteritems():
                spike_count_total[n_index + start] = count

        return spike_count_total

    def get_v(self, gather=True, compatible_output=True):
        """ Get v for all cells in this container

        :param gather:
            not used - inserted to match PyNN specs
        :type gather: bool
        :param compatible_output:
            not used - inserted to match PyNN specs
        :type compatible_output: bool
        :return: 3 dimensional numpy array
        """

        return self._get_recorded_values(
            "membrane voltages", 3, AbstractVRecordable, RecordingType.V,
            lambda vertex: vertex.get_v(
                self._spinnaker.no_machine_time_steps,
                self._spinnaker.placements, self._spinnaker.graph_mapper,
                self._spinnaker.buffer_manager,
                self._spinnaker.machine_time_step))

    @property
    def first_id(self):
        return numpy.min(self._cells)

    @property
    def last_id(self):
        return numpy.max(self._cells)

    def id_to_index(self, id):  # @ReservedAssignment
        """ Get the index of the given cell in the container

        :param id:  the neuron cell object to find the index of
        :return: index
        :rtype: int
        """
        if not numpy.iterable(id):
            return self._cells.index(id)

        return [self._cells[item_id] for item_id in id]

    def is_local(self, id):  # @ReservedAssignment
        """ All cells are local on SpiNNaker
        """
        return True

    def initialize(self, variable, value):
        """ Initialise the given variable for all cells in this container

        :param variable: the variable to set
        :param value: the value to use
        :return: None
        """

        # expand variables
        values_for_atoms = utility_calls.convert_param_to_numpy(
            value, len(self._cells))

        # Set the value
        for cell, value in zip(self._cells, values_for_atoms):
            cell.set_initial_value(variable, value)

    def inject(self, current_source):
        """ Add a current source to each cell

        :param current_source:
        :return:
        """

        # TODO: To be implemented
        raise NotImplementedError

    @property
    def label(self):
        return self._label

    @property
    def local_cells(self):
        return self._cells

    @property
    def local_size(self):
        return self.size

    def meanSpikeCount(self, gather=True):
        """ The mean spike count for the cells in the container

        :param gather:
        """
        spike_counts = self.get_spike_counts(gather)
        total_spikes = sum(spike_counts.values())
        return total_spikes / self._size

    @property
    def positions(self):
        return numpy.vstack([cell.position for cell in self._cells])

    @positions.setter
    def positions(self, positions):
        for position, cell in zip(positions, self._cells):
            cell.position = position

    @property
    def position_generator(self):
        def gen(i):
            return self.positions[:, i]
        return gen

    def printSpikes(
            self, file,  # @ReservedAssignment
            gather=True, compatible_output=True):
        """ Write spike time information to a given file.

        :param file: The file path to write to
        :param gather: Supported from the PyNN language, but ignored here
        """
        spikes = self.getSpikes(gather, compatible_output)
        if spikes is not None:
            first_id = 0
            num_neurons = len(self._cells)
            dimensions = len(self._cells)
            last_id = len(self._cells) - 1
            utility_calls.check_directory_exists_and_create_if_not(file)
            spike_file = open(file, "w")
            spike_file.write("# first_id = {}\n".format(first_id))
            spike_file.write("# n = {}\n".format(num_neurons))
            spike_file.write("# dimensions = [{}]\n".format(dimensions))
            spike_file.write("# last_id = {}\n".format(last_id))
            for (neuronId, time) in spikes:
                spike_file.write("{}\t{}\n".format(time, neuronId))
            spike_file.close()

    def print_gsyn(
            self, file,  # @ReservedAssignment
            gather=True, compatible_output=True):
        """ Write gsyn information to a given file.

        :param file: the file path to write to
        :param gather: Supported from the PyNN language, but ignored here
        """
        time_step = (self._spinnaker.machine_time_step * 1.0) / 1000.0
        gsyn = self.get_gsyn(gather, compatible_output)
        first_id = 0
        num_neurons = len(self._cells)
        dimensions = len(self._cells)
        utility_calls.check_directory_exists_and_create_if_not(file)
        file_handle = open(file, "w")
        file_handle.write("# first_id = {}\n".format(first_id))
        file_handle.write("# n = {}\n".format(num_neurons))
        file_handle.write("# dt = {}\n".format(time_step))
        file_handle.write("# dimensions = [{}]\n".format(dimensions))
        file_handle.write("# last_id = {{}}\n".format(num_neurons - 1))
        file_handle = open(file, "w")
        for (neuronId, time, value_e, value_i) in gsyn:
            file_handle.write("{}\t{}\t{}\t{}\n".format(
                time, neuronId, value_e, value_i))
        file_handle.close()

    def print_v(
            self, file,  # @ReservedAssignment
            gather=True, compatible_output=True):
        """ Write membrane voltage to a file

        :param file: the file path to write to
        :param gather: Supported from the PyNN language, but ignored here
        """
        time_step = (self._spinnaker.machine_time_step * 1.0) / 1000.0
        v = self.get_v(gather, compatible_output)
        utility_calls.check_directory_exists_and_create_if_not(file)
        file_handle = open(file, "w")
        first_id = 0
        num_neurons = len(self._cells)
        dimensions = len(self._cells)
        file_handle.write("# first_id = {}\n".format(first_id))
        file_handle.write("# n = {}\n".format(num_neurons))
        file_handle.write("# dt = {}\n".format(time_step))
        file_handle.write("# dimensions = [{}]\n".format(dimensions))
        file_handle.write("# last_id = {}\n".format(num_neurons - 1))
        for (neuronId, time, value) in v:
            file_handle.write("{}\t{}\t{}\n".format(time, neuronId, value))
        file_handle.close()

    # noinspection PyPep8Naming
    def randomInit(self, distribution):
        """ Set initial membrane potentials for all the cells in the\
            population to random values.

        :param `pyNN.random.RandomDistribution` distribution:
            the distribution used to draw random values.

        """
        self.initialize('v', distribution)

    def record(self, to_file=None):
        """ Record the spikes for the cells in this collection

        :param to_file: True if the neurons should be recorded to a file
        :return: None
        """

        # set all the atoms directly
        recording_enabled = False
        for atom in self._atoms:
            if atom.can_record(RecordingType.SPIKES):
                atom.set_recording(RecordingType.SPIKES, True, to_file)
                recording_enabled = True

        if not recording_enabled:
            logger.warn(
                "No cells in this collection can record conductance")

        else:

            # locate vertices to set to recording status
            for population in self._populations:
                for vertex, _, _, _, _ in population._vertices:
                    if isinstance(vertex, AbstractSpikeRecordable):
                        vertex.set_recording_spikes()

    def record_gsyn(self, to_file=None):
        """ Record conductance for all the cells in this collection

        :param to_file: True if the data should be recorded to a file
        """

        # set all the atoms directly
        recording_enabled = False
        for atom in self._atoms:
            if atom.can_record(RecordingType.GSYN):
                atom.set_recording(RecordingType.GSYN, True, to_file)
                recording_enabled = True

        if not recording_enabled:
            logger.warn(
                "No cells in this collection can record conductance")

        else:

            # locate vertices to set to recording status
            for population in self._populations:
                for vertex, _, _, _, _ in population._vertices:
                    if isinstance(vertex, AbstractGSynRecordable):
                        vertex.set_recording_gsyn()

    def record_v(self, to_file=None):
        """ Record membrane voltage for all cells in this collection

        :param to_file: True if the data should be recorded to a file
        """

        # set all the atoms directly
        recording_enabled = False
        for atom in self._atoms:
            if atom.can_record(RecordingType.V):
                atom.set_recording(RecordingType.V, True, to_file)
                recording_enabled = True

        if not recording_enabled:
            logger.warn(
                "No cells in this collection can record membrane voltage")

        else:

            # locate vertices to set to recording status
            for population in self._populations:
                for vertex, _, _, _, _ in population._vertices:
                    if isinstance(vertex, AbstractVRecordable):
                        vertex.set_recording_v()

    def rset(self, parametername, rand_distr):
        """ Set a parameter to a random distribution for all cells in this\
            container

        :param parametername: parameter to set
        :param rand_distr: the random distribution
        """
        values_for_atoms = utility_calls.convert_param_to_numpy(
            rand_distr, len(self._cells))
        for cell, value in zip(self._cells, values_for_atoms):
            cell[parametername] = value

    def sample(self, n, rng=None):
        """ Return a random selection of neurons from a population in the form\
            of a population view

        :param n: the number of neurons to sample
        :param rng: the random number generator to use.
        """
        if n > self._size:
            raise exceptions.ConfigurationException(
                "n must be less than or equal to the size of the population")
        if rng is None:
            rng = random.NumpyRNG()
        indices = rng.permutation(numpy.arange(len(self)))[0:n]
        return self[indices]

    def size(self):
        return len(self._cells)

    def save_positions(self, file):  # @ReservedAssignment
        """ Write the positions of the atoms to a file

        :param file: The name of the file to write to
        :return:
        """
        file_handle = open(file, "w")
        for cell in self._cells:
            x, y, z = cell.position
            file_handle.write("{}\t{}\t{}\t{}\n".format(cell, x, y, z))
        file_handle.close()

    def tset(self, parametername, value_array):
        """ 'Topographic' set. Set the value of parametername to the values in\
            value_array, which must have the same length as this container.
        :param parametername: the name of the parameter
        :param value_array: the array of values which must have the correct\
                number of elements.
        """
        values_for_atoms = utility_calls.convert_param_to_numpy(
            value_array, len(self._cells))
        for cell, value in zip(self._cells, values_for_atoms):
            cell[parametername] = value

    def set(self, param, val=None):
        """ Set one or more parameters for every cell in the population view.

        param can be a dict, in which case value should not be supplied, or a
        string giving the parameter name, in which case value is the parameter
        value. value can be a numeric value, or list of such
        (e.g. for setting spike times)::

          p.set("tau_m", 20.0).
          p.set({'tau_m':20, 'v_rest':-65})
        :param param: the parameter to set
        :param val: the value of the parameter to set.
        """
        if isinstance(param, dict):
            for name, value in param.iteritems():
                for cell in self._cells:
                    cell[name] = value
        else:
            for cell in self._cells:
                cell[param] = val
