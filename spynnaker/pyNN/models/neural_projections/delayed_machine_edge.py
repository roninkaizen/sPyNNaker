from pacman.model.decorators.overrides import overrides
from pacman.model.graphs.machine.impl.machine_edge import MachineEdge
from pacman.executor.injection_decorator import inject_items
from spynnaker.pyNN.models.abstract_models.abstract_filterable_edge \
    import AbstractFilterableEdge


class DelayedMachineEdge(MachineEdge, AbstractFilterableEdge):

    def __init__(
            self, projections, pre_vertex, post_vertex,
            label=None, weight=1):
        MachineEdge.__init__(
            self, pre_vertex, post_vertex, label=label, traffic_weight=weight)
        AbstractFilterableEdge.__init__(self)
        self._projections = projections

    @inject_items({
        "machine_graph": "MemoryMachineGraph"
    })
    @overrides(AbstractFilterableEdge.filter_edge)
    def filter_edge(self, graph_mapper, machine_graph):



        # If here, all n_connections were 0
        return True
