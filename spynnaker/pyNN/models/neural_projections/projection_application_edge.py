from pacman.model.decorators.overrides import overrides
from pacman.model.graphs.application.impl.application_edge import \
    ApplicationEdge

from spynnaker.pyNN.models.neural_projections.projection_machine_edge \
    import ProjectionMachineEdge

import logging

logger = logging.getLogger(__name__)


class ProjectionApplicationEdge(ApplicationEdge):
    """ An edge which terminates on an AbstractPopulationVertex
    """

    def __init__(
            self, pre_vertex, post_vertex, projection, label=None):
        ApplicationEdge.__init__(
            self, pre_vertex, post_vertex, label=label)

        # The edge from the delay extension of the pre_vertex to the
        # post_vertex - this might be None if no long delays are present
        self._delay_edge = None

        # the projection that this edge is associated with
        self._projections = [projection]

    def add_projection(self, projection):
        self._projections.append(projection)

    @property
    def projections(self):
        return self._projections

    @property
    def delay_edge(self):
        return self._delay_edge

    @delay_edge.setter
    def delay_edge(self, delay_edge):
        self._delay_edge = delay_edge

    @property
    def n_delay_stages(self):
        if self._delay_edge is None:
            return 0
        return self._delay_edge.pre_vertex.n_delay_stages

    @overrides(ApplicationEdge.create_machine_edge)
    def create_machine_edge(
            self, pre_vertex, pre_vertex_slice,
            post_vertex, post_vertex_slice, label):
        machine_edge = ProjectionMachineEdge(
            self._projections, pre_vertex, post_vertex, label)
        for projection in self._projections:
            projection.add_synaptic_machine_edge(
                self, machine_edge, pre_vertex_slice, post_vertex_slice)
        return machine_edge
