from pacman.model.decorators.overrides import overrides
from spynnaker.pyNN.models.neural_projections.delayed_machine_edge \
    import DelayedMachineEdge
from pacman.model.graphs.application.impl.application_edge import \
    ApplicationEdge


class DelayedApplicationEdge(ApplicationEdge):

    def __init__(
            self, pre_vertex, post_vertex, projection, label=None):
        ApplicationEdge.__init__(
            self, pre_vertex, post_vertex, label=label)
        self._projections = [projection]

    def add_projection(self, projection):
        self._projections.append(projection)

    @overrides(ApplicationEdge.create_machine_edge)
    def create_machine_edge(
            self, pre_vertex, pre_vertex_slice,
            post_vertex, post_vertex_slice, label):
        return DelayedMachineEdge(
            self._projections, pre_vertex, post_vertex, label)
