from pacman.model.constraints.partitioner_constraints.\
    partitioner_same_size_as_vertex_constraint \
    import PartitionerSameSizeAsVertexConstraint

from spinn_front_end_common.utilities import exceptions
from spinn_machine.utilities.progress_bar import ProgressBar

from spynnaker.pyNN import DelayExtensionVertex
from spynnaker.pyNN.models.neural_projections.projection_application_edge \
    import ProjectionApplicationEdge
from spynnaker.pyNN.models.neural_projections.delay_afferent_application_edge \
    import DelayAfferentApplicationEdge
from spynnaker.pyNN.models.neural_projections.delayed_application_edge \
    import DelayedApplicationEdge
from spynnaker.pyNN.models.neural_projections.synapse_information \
    import SynapseInformation
from spynnaker.pyNN.models.neuron.connection_holder import ConnectionHolder
from spynnaker.pyNN.utilities import constants

import logging
import math
logger = logging.getLogger(__name__)


class AbstractGrouper(object):
    """ Provides basic functionality for grouping algorithms
    """

    def handle_projections(
            self, projections, population_to_vertices,
            user_max_delay, graph, using_virtual_board, machine_time_step):
        """ Handle the addition of projections

        :param projections: the list of projections from the pynn level
        :param population_atom_mapping: the mapping from pops and model types
        :param pop_to_vertex_mapping: the mapping of pop views and atoms
        :param user_max_delay: the end users max delay
        :param partitionable_graph: the partitionable graph to add edges into
        :param using_virtual_board: if the end user is using a virtual board.
        :return: None
        """

        # hold a vertex to delay vertex map for tracking which vertices
        # have delays already
        vertex_to_delay_vertex = dict()

        # Hold a pre-vertex, post-vertex to edge map to ensure that only
        # one edge exists
        edges_by_vertices = dict()

        progress_bar = ProgressBar(
            len(projections), "Creating Edges")

        # iterate through projections and create edges
        for projection in projections:

            # Get the presynaptic and postsynaptic vertices
            pre_vertices = population_to_vertices[
                projection._presynaptic_population]
            post_vertices = population_to_vertices[
                projection._postsynaptic_population]

            # For each pair of vertices, add edges
            for pre_vertex, pre_start_atom, pre_end_atom in pre_vertices:
                for (post_vertex, post_start_atom,
                     post_end_atom) in post_vertices:

                    # check if all delays requested can fit into the natively
                    # supported delays in the models
                    max_delay = projection._synapse_dynamics.get_delay_maximum(
                        projection._connector)
                    if max_delay is None:
                        max_delay = user_max_delay
                    delay_extension_max_supported_delay = (
                        constants.MAX_DELAY_BLOCKS *
                        constants.MAX_TIMER_TICS_SUPPORTED_PER_BLOCK)

                    # get post vertex max delay
                    post_vertex_max_supported_delay_ms = \
                        post_vertex.maximum_delay_supported_in_ms

                    # verify that the max delay is less than the max supported
                    # by the implementation of delays
                    if max_delay > (post_vertex_max_supported_delay_ms +
                                    delay_extension_max_supported_delay):
                        raise exceptions.ConfigurationException(
                            "The maximum delay {} for projection {} is not"
                            " supported".format(max_delay, projection))

                    # verify max delay is less than the max delay entered by
                    # the user during setup.
                    if max_delay > (user_max_delay /
                                    (machine_time_step / 1000.0)):
                        logger.warn(
                            "Delay on projection {} of {} is greater than the"
                            " specified maximum {}".format(
                                projection, max_delay, user_max_delay))

                    # Find out if there is an existing edge between the
                    # vertices
                    edge_to_merge = edges_by_vertices.get(
                        (pre_vertex, post_vertex), None)

                    if edge_to_merge is not None:

                        # If there is an existing edge, add the synapse info
                        edge_to_merge.add_synapse_information(
                            synapse_information)
                        projection_edge = edge_to_merge
                    else:

                        # If there isn't an existing edge, create a new one
                        projection_edge = ProjectionApplicationEdge(
                            pre_vertex, post_vertex, synapse_information,
                            projection, label=projection.label)

                        # add to graph
                        graph.add_edge(
                            projection_edge, projection.EDGE_PARTITION_ID)
                        edges_by_vertices[(pre_vertex, post_vertex)] = \
                            projection_edge

                    # update projection
                    projection._projection_edge = projection_edge

                    # If the delay exceeds the post vertex delay, add a delay
                    # extension
                    if max_delay > post_vertex_max_supported_delay_ms:

                        delay_vertex = vertex_to_delay_vertex.get(pre_vertex)

                        if delay_vertex is None:

                            # build a delay vertex
                            delay_name = "{}_delayed".format(pre_vertex.label)
                            delay_vertex = DelayExtensionVertex(
                                pre_vertex.n_atoms,
                                post_vertex_max_supported_delay_ms, pre_vertex,
                                label=delay_name)

                            # store in map for other projections
                            vertex_to_delay_vertex[pre_vertex] = delay_vertex

                            # add partitioner constraint to the pre pop vertex
                            pre_vertex.add_constraint(
                                PartitionerSameSizeAsVertexConstraint(
                                    delay_vertex))
                            graph.add_vertex(delay_vertex)

                            # Add the edge
                            delay_afferent_edge = DelayAfferentApplicationEdge(
                                pre_vertex, delay_vertex,
                                label="{}_to_DelayExtension".format(
                                    pre_vertex.label))
                            graph.add_edge(
                                delay_afferent_edge,
                                projection.EDGE_PARTITION_ID)

                        # Ensure that the delay extension knows how many states
                        # it will support
                        n_stages = int(math.ceil(
                            float(max_delay -
                                  post_vertex_max_supported_delay_ms) /
                            float(post_vertex_max_supported_delay_ms)))
                        if n_stages > delay_vertex.n_delay_stages:
                            delay_vertex.n_delay_stages = n_stages

                        # Create the delay edge if there isn't one already
                        delay_edge = edges_by_vertices.get(
                            (delay_vertex, post_vertex), None)
                        if delay_edge is None:
                            delay_edge = DelayedApplicationEdge(
                                delay_vertex, post_vertex, synapse_information,
                                label="{}_delayed_to_{}".format(
                                    pre_vertex.label, post_vertex.label))
                            graph.add_edge(
                                delay_edge, projection.EDGE_PARTITION_ID)
                            edges_by_vertices[(delay_vertex, post_vertex)] = \
                                delay_edge

                        projection_edge.delay_edge = delay_edge

                    # If there is a virtual board, we need to hold the data in
                    # case the user asks for it
                    if using_virtual_board:
                        virtual_connection_list = list()
                        connection_holder = ConnectionHolder(
                            None, False, pre_vertex.n_atoms,
                            post_vertex.n_atoms,
                            virtual_connection_list)

                        post_vertex.add_pre_run_connection_holder(
                            connection_holder, projection_edge,
                            synapse_information)

            projection._virtual_connection_list = virtual_connection_list
            progress_bar.update()
        progress_bar.end()
