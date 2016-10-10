from spinn_machine.utilities.progress_bar import ProgressBar
from spynnaker.pyNN.overridden_pacman_functions.groupers.\
    abstract_grouper import AbstractGrouper
from pacman.model.graphs.application.impl.application_graph \
    import ApplicationGraph

from collections import defaultdict
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
        graph = ApplicationGraph("grouped_application_graph")
        pop_to_vertex_mapping = dict()

        progress_bar = ProgressBar(len(populations), "Creating Vertices")

        # Track which vertices contain which populations
        population_to_vertices = defaultdict(list)

        # Build a vertex for each population
        for population in populations:
            vertex = population.cell_type.create_vertex(
                population._population_parameters, population._cells,
                population._label, population._constraints,
                population._synapse_dynamics)
            population.add_vertex(
                vertex, 0, population.size, 0, population.size)
            population_to_vertices[population].append(vertex)
            progress_bar.update()
        progress_bar.end()

        return graph
