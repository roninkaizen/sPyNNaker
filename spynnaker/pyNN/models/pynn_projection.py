from spynnaker.pyNN.models.neuron.connection_holder import ConnectionHolder
from spynnaker.pyNN.models.neural_projections.synapse_retreval_information \
    import SynapseRetrevalInformation
from spynnaker.pyNN.models.neural_projections.synapse_information \
    import SynapseInformation
from spynnaker.pyNN.models.neuron.synapse_dynamics.synapse_dynamics_static \
    import SynapseDynamicsStatic

from spinn_front_end_common.abstract_models.abstract_changable_after_run \
    import AbstractChangableAfterRun
from spinn_front_end_common.utilities import exceptions

from spinn_machine.utilities.progress_bar import ProgressBar

import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


# noinspection PyProtectedMember
class Projection(object):
    """ A container for all the connections of a given type (same synapse type\
        and plasticity mechanisms) between two populations, together with\
        methods to set parameters of those connections, including of\
        plasticity mechanisms.
    """

    # partition id used by all edges of projections
    EDGE_PARTITION_ID = "SPIKE"

    # noinspection PyUnusedLocal
    def __init__(
            self, presynaptic_population, postsynaptic_population, label,
            connector, spinnaker_control, machine_time_step, timescale_factor,
            source=None, target='excitatory', synapse_dynamics=None, rng=None):
        self._spinnaker = spinnaker_control
        self._presynaptic_population = presynaptic_population
        self._postsynaptic_population = postsynaptic_population
        self._connector = connector
        self._target = target
        self._rng = rng
        self._virtual_connection_list = None
        self._synapse_information = list()
        self._synapse_retrieval_information = list()
        self._synapse_information_by_edge = dict()

        self._n_pre_vertex_slices = 0
        self._n_post_vertex_slices = 0
        self._pre_vertex_slices = OrderedDict()
        self._post_vertex_slices = OrderedDict()
        self._pre_slice_indices = dict()
        self._post_slice_indices = dict()

        if source is not None:
            logger.warn(
                "source currently means nothing to the SpiNNaker"
                " implementation of the PyNN projection, therefore it will be"
                " ignored")

        # Check projection is to a vertex which can handle spikes reception
        if not postsynaptic_population.celltype.supports_connector:
            raise exceptions.ConfigurationException(
                "postsynaptic population is not designed to receive"
                " synaptic projections")

        # Check projection synapse target is supported by the postsynaptic
        # population
        synapse_type = postsynaptic_population.celltype.synapse_type
        self._synapse_id = synapse_type.get_synapse_id_by_target(target)
        if self._synapse_id is None:
            raise exceptions.ConfigurationException(
                "The synapse type {} of the post synaptic cell type {} does"
                " not support the target {}".format(
                    synapse_type.__name__,
                    postsynaptic_population.cell_type.__name__, target))

        # Set the synapse dynamics in the post-population
        self._synapse_dynamics = synapse_dynamics
        if synapse_dynamics is None:
            self._synapse_dynamics = SynapseDynamicsStatic()
        postsynaptic_population.set_synapse_dynamics(self._synapse_dynamics)

        # check that the projection edges label is not none, and give an
        # auto generated label if set to None
        if label is None:
            self._label = "Projection {}".format(
                spinnaker_control.none_labelled_edge_count)
            spinnaker_control.increment_none_labelled_edge_count()
        else:
            self._label = label

        spinnaker_control._add_projection(self)

    @property
    def synapse_information(self):
        return self._synapse_information

    @property
    def connector(self):
        return self._connector

    @property
    def synapse_id(self):
        return self._synapse_id

    def add_synaptic_edge(
            self, application_edge, pre_vertex_start, pre_vertex_end,
            pre_population_start, pre_population_end,
            post_vertex_start, post_vertex_end,
            post_population_start, post_population_end):
        """ Add an application edge that goes between a pre-vertex of the\
            pre-population and post-vertex of the post-population of this\
            projection
        """

        synapse_info = SynapseInformation(
            application_edge, pre_vertex_start, pre_vertex_end,
            pre_population_start, pre_population_end,
            post_vertex_start, post_vertex_end,
            post_population_start, post_population_end)
        self._synapse_information.append(synapse_info)
        self._synapse_information_by_edge[application_edge] = synapse_info

    def add_synaptic_machine_edge(
            self, application_edge, machine_edge,
            pre_vertex_slice, post_vertex_slice):
        """ Add a machine edge that goes between a machine vertex of an\
            application vertex of the pre-population and a machine vertex of\
            an application vertex of the post-population of this vertex.\
            Note that this might not be relevant to this projection, and so\
            it might not actually get added.
        """
        synapse_information = self._synapse_information_by_edge.get(
            application_edge)
        if (synapse_information is not None and
                pre_vertex_slice.lo_atom <=
                synapse_information.pre_vertex_end and
                pre_vertex_slice.hi_atom >=
                synapse_information.pre_vertex_start):
            pre_slice_index = self._n_pre_vertex_slices
            self._n_pre_vertex_slices += 1
            self._pre_slice_indices[
                machine_edge.pre_vertex] = pre_slice_index
            self._pre_vertex_slices[
                machine_edge.pre_vertex] = pre_vertex_slice
            post_slice_index = self._n_post_vertex_slices
            self._n_post_vertex_slices += 1
            self._post_slice_indices[
                machine_edge.post_vertex] = post_slice_index
            self._post_vertex_slices[
                machine_edge.post_vertex] = post_vertex_slice

    @property
    def pre_vertex_slices(self):
        """ Pre-machine vertices and slices of the pre-vertex that are\
            relevant to this projection

        :return: list of slice
        """
        return self._pre_vertex_slices.values()

    def get_pre_vertex_slice(self, pre_vertex):
        """ Get the pre-vertex slice of the given pre-vertex
        """
        return self._pre_vertex_slices[pre_vertex]

    def get_pre_vertex_slice_index(self, pre_vertex):
        """ Get the index of the pre-vertex in the list of slices
        """
        return self._pre_slice_indices[pre_vertex]

    @property
    def post_vertex_slices(self):
        """ Post-machine vertices and slices of the post-vertex that are\
            relevant to this projection

        :return: list of slice
        """
        return self._post_vertex_slices.values()

    def get_post_vertex_slice(self, post_vertex):
        """ Get the post-vertex slice of the given post-vertex
        """
        return self._post_vertex_slices[post_vertex]

    def get_post_vertex_slice_index(self, post_vertex):
        """ Get the index of the post-vertex in the list of slices
        """
        return self._post_slice_indices[post_vertex]

    def add_synapse_retreval_information(
            self, post_vertex, machine_edge, index, delay_index,
            synapse_information):
        """ Add information to enable the retrieval of the synaptic\
            information of this projection
        """
        self._synapse_retrieval_information.append(
            SynapseRetrevalInformation(
                post_vertex, machine_edge, index, delay_index,
                synapse_information))

    @property
    def label(self):
        return self._label

    @property
    def target(self):
        return self._target

    @property
    def requires_mapping(self):
        for info in self._synapse_information:
            if (isinstance(
                    info.application_edge, AbstractChangableAfterRun) and
                    info.application_edge.requires_mapping):
                return True
        return False

    def mark_no_changes(self):
        for info in self._synapse_information:
            if isinstance(info.application_edge, AbstractChangableAfterRun):
                info.application_edge.mark_no_changes()

    def describe(self, template='projection_default.txt', engine='default'):
        """ Return a human-readable description of the projection.

        The output may be customised by specifying a different template
        together with an associated template engine (see ``pyNN.descriptions``)

        If template is None, then a dictionary containing the template context
        will be returned.
        """
        # TODO
        raise NotImplementedError

    def __getitem__(self, i):
        """Return the `i`th connection within the Projection."""
        # TODO: Need to work out what is being returned
        raise NotImplementedError

    # noinspection PyPep8Naming
    def getSynapseDynamics(self, parameter_name, list_format='list',
                           gather=True):
        """ Get parameters of the dynamic synapses for all connections in this\
            Projection.
        :param parameter_name:
        :param list_format:
        :param gather:
        """
        # TODO: Need to work out what is to be returned
        raise NotImplementedError

    def _get_synaptic_data(self, as_list, data_to_get):

        # If in virtual board mode, the connection data should be set
        if self._virtual_connection_list is not None:
            return ConnectionHolder(
                data_to_get, as_list, self._presynaptic_population.size,
                self._postsynaptic_population.size,
                self._virtual_connection_list)

        connection_holder = ConnectionHolder(
            data_to_get, as_list, self._presynaptic_population.size,
            self._postsynaptic_population.size)

        # If we haven't run, add the holder to get connections, and return it
        if not self._spinnaker.has_ran:
            for synapse_info in self._synapse_information:
                synapse_info.application_edge.post_vertex\
                    .add_pre_run_connection_holder(
                        connection_holder, synapse_info)
            return connection_holder

        # Otherwise, get the connections now
        graph_mapper = self._spinnaker.graph_mapper
        placements = self._spinnaker.placements
        transceiver = self._spinnaker.transceiver
        routing_infos = self._spinnaker.routing_infos
        machine_time_step = self._spinnaker.machine_time_step
        progress = ProgressBar(
            len(self._synapse_retrieval_information),
            "Getting {}s for projection between {} and {}".format(
                data_to_get, self._pre_synaptic_population.label,
                self._post_synaptic_population.label))
        for info in self._synapse_retrieval_information:
            placement = placements.get_placement_of_vertex(info.post_vertex)
            connections = info.post_vertex.get_connections_from_machine(
                transceiver, placement, info.machine_edge, graph_mapper,
                routing_infos, self._synapse_dynamics, self._synapse_id,
                info.synapse_information, info.index, info.delay_index,
                machine_time_step)
            if connections is not None:
                connection_holder.add_connections(connections)
            progress.update()
        progress.end()
        connection_holder.finish()
        return connection_holder

    # noinspection PyPep8Naming
    def getWeights(self, format='list', gather=True):  # @ReservedAssignment
        """
        Get synaptic weights for all connections in this Projection.

        Possible formats are: a list of length equal to the number of
        connections in the projection, a 2D weight array (with NaN for
        non-existent connections). Note that for the array format, if there is
        more than connection between two cells, the summed weight will be
        given.
        :param format: the type of format to be returned (only support "list")
        :param gather: gather the weights from stuff. currently has no meaning\
                in spinnaker when set to false. Therefore is always true
        """
        if not gather:
            exceptions.ConfigurationException(
                "the gather param has no meaning for spinnaker when set to "
                "false")

        return self._get_synaptic_data(format == 'list', "weight")

    # noinspection PyPep8Naming
    def getDelays(self, format='list', gather=True):  # @ReservedAssignment
        """
        Get synaptic delays for all connections in this Projection.

        Possible formats are: a list of length equal to the number of
        connections in the projection, a 2D delay array (with NaN for
        non-existent connections).
        """
        if not gather:
            exceptions.ConfigurationException(
                "the gather param has no meaning for spinnaker when set to "
                "false")

        return self._get_synaptic_data(format == 'list', "delay")

    def __len__(self):
        """ Return the total number of local connections.
        """

        # TODO: Need to work out what this means
        raise NotImplementedError

    # noinspection PyPep8Naming
    def printDelays(self, file_name, list_format='list', gather=True):
        """ Print synaptic weights to file. In the array format, zeros are\
            printed for non-existent connections.
        """
        # TODO:
        raise NotImplementedError

    # noinspection PyPep8Naming
    def printWeights(self, file_name, list_format='list', gather=True):
        """ Print synaptic weights to file. In the array format, zeros are\
            printed for non-existent connections.
        """
        # TODO:
        raise NotImplementedError

    # noinspection PyPep8Naming
    def randomizeWeights(self, rand_distr):
        """ Set weights to random values taken from rand_distr.
        """
        # TODO: Requires that the synapse list is not created proactively
        raise NotImplementedError

    # noinspection PyPep8Naming
    def randomizeDelays(self, rand_distr):
        """ Set delays to random values taken from rand_distr.
        """
        # TODO: Requires that the synapse list is not created proactively
        raise NotImplementedError

    # noinspection PyPep8Naming
    def randomizeSynapseDynamics(self, param, rand_distr):
        """ Set parameters of the synapse dynamics to values taken from\
            rand_distr
        """
        # TODO: Look at what this is randomising
        raise NotImplementedError

    def __repr__(self):
        return self._label

    # noinspection PyPep8Naming
    def saveConnections(self, file_name, gather=True, compatible_output=True):
        """ Save connections to file in a format suitable for reading in with\
            a FromFileConnector.
        """
        # TODO
        raise NotImplementedError

    def size(self, gather=True):
        """ Return the total number of connections.
         - only local connections, if gather is False,
         - all connections, if gather is True (default)
        """
        # TODO
        raise NotImplementedError

    # noinspection PyPep8Naming
    def setDelays(self, d):
        """ Set the delays

        d can be a single number, in which case all delays are set to this\
        value, or a list/1D array of length equal to the number of connections\
        in the projection, or a 2D array with the same dimensions as the\
        connectivity matrix (as returned by `getDelays(format='array')`).
        """
        # TODO: Requires that the synapse list is not created proactively
        raise NotImplementedError

    # noinspection PyPep8Naming
    def setSynapseDynamics(self, param, value):
        """ Set parameters of the dynamic synapses for all connections in this\
            projection.
        """
        # TODO: Need to set this in the edge
        raise NotImplementedError

    # noinspection PyPep8Naming
    def setWeights(self, w):
        """ Set the weights

        w can be a single number, in which case all weights are set to this\
        value, or a list/1D array of length equal to the number of connections\
        in the projection, or a 2D array with the same dimensions as the\
        connectivity matrix (as returned by `getWeights(format='array')`).\
        Weights should be in nA for current-based and uS for conductance-based\
        synapses.
        """

        # TODO: Requires that the synapse list is not created proactively
        raise NotImplementedError

    # noinspection PyPep8Naming
    def weightHistogram(self, min_weight=None, max_weight=None, nbins=10):
        """ Return a histogram of synaptic weights.

        If min and max are not given, the minimum and maximum weights are\
        calculated automatically.
        """
        # TODO
        raise NotImplementedError
