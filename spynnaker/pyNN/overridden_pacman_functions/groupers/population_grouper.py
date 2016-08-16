from collections import OrderedDict
from pacman.model.partitionable_graph.partitionable_graph import \
    PartitionableGraph

from spinn_machine.utilities.progress_bar import ProgressBar
from spynnaker.pyNN.models.neuron.bag_of_neurons_vertex import \
    BagOfNeuronsVertex
from spynnaker.pyNN.overridden_pacman_functions.groupers.\
    abstract_grouper import AbstractGrouper


import logging
logger = logging.getLogger(__name__)


class Grouper(AbstractGrouper):
    """ Takes a bag of neurons and maps them into vertices
    """

    def __init__(self):
        AbstractGrouper.__init__(self)

    def __call__(
            self, populations, projections, user_max_delay,
            using_virtual_board):

        # build a partitionable graph
        graph = PartitionableGraph("grouped_application_graph")
        pop_to_vertex_mapping = dict()

        progress_bar = ProgressBar(len(populations), "Creating Vertices")

        # Build a vertex for each population
        for population in populations:
            vertex = population.create_vertex()
            graph.add_vertex(vertex)
            pop_to_vertex_mapping[population] = vertex
            progress_bar.update()
        progress_bar.end()

        # handle projections
        self.handle_projections(
            projections, population_atom_mapping, pop_to_vertex_mapping,
            user_max_delay, partitionable_graph, using_virtual_board)

        return {
            'partitionable_graph': graph,
                'pop_to_vertex_mapping': pop_to_vertex_mapping
        }
