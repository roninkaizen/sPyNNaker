from pyNN.random import RandomDistribution

from pacman.model.decorators.overrides import overrides
from spynnaker.pyNN.utilities import utility_calls
from spinn_front_end_common.interface.provenance\
    .abstract_provides_local_provenance_data \
    import AbstractProvidesLocalProvenanceData
from spynnaker.pyNN.models.abstract_models.abstract_weight_updatable \
    import AbstractWeightUpdatable
from pacman.model.graphs.machine.impl.machine_edge import MachineEdge
from spynnaker.pyNN.models.abstract_models.abstract_filterable_edge \
    import AbstractFilterableEdge


class ProjectionMachineEdge(
        MachineEdge, AbstractFilterableEdge,
        AbstractWeightUpdatable, AbstractProvidesLocalProvenanceData):

    def __init__(
            self, projections, pre_vertex, post_vertex,
            label=None, traffic_weight=1):
        MachineEdge.__init__(
            self, pre_vertex, post_vertex, label=label,
            traffic_weight=traffic_weight)
        AbstractFilterableEdge.__init__(self)
        AbstractWeightUpdatable.__init__(self)

        self._projections = projections

    @overrides(AbstractFilterableEdge.filter_edge)
    def filter_edge(self, graph_mapper):

        for projection in self._projections:
            pre_slices = projection.pre_vertex_slices
            post_slices = projection.post_vertex_slices
            pre_slice = projection.get_pre_vertex_slice(self._pre_vertex)
            post_slice = projection.get_post_vertex_slice(self._post_vertex)
            pre_slice_index = projection.get_pre_vertex_slice_index(
                self._pre_vertex)
            post_slice_index = projection.get_post_vertex_slice_index(
                self._post_vertex)
            n_connections = projection.connector.\
                get_n_connections_to_post_vertex_maximum(
                    pre_slices, pre_slice_index, post_slices,
                    post_slice_index, pre_slice, post_slice)
            if n_connections > 0:
                return False

        # If here, all n_connections were 0
        return True

    @overrides(AbstractWeightUpdatable.update_weight)
    def update_weight(self, graph_mapper):
        pre_vertex = graph_mapper.get_application_vertex(
            self.pre_vertex)
        pre_slice_index = graph_mapper.get_machine_vertex_index(
            self.pre_vertex)
        pre_vertex_slice = graph_mapper.get_slice(
            self.pre_vertex)
        pre_slices = graph_mapper.get_slices(pre_vertex)
        post_vertex = graph_mapper.get_application_vertex(
            self.post_vertex)
        post_slice_index = graph_mapper.get_machine_vertex_index(
            self.post_vertex)
        post_vertex_slice = graph_mapper.get_slice(
            self.post_vertex)
        post_slices = graph_mapper.get_slices(post_vertex)

        weight = 0
        for projection in self._projections:
            for synapse_info in projection.synapse_information:
                new_weight = synapse_info.connector.\
                    get_n_connections_to_post_vertex_maximum(
                        pre_slices, pre_slice_index, post_slices,
                        post_slice_index, pre_vertex_slice, post_vertex_slice)
                new_weight *= pre_vertex_slice.n_atoms
                if hasattr(pre_vertex, "rate"):
                    rate = pre_vertex.rate
                    if hasattr(rate, "__getitem__"):
                        rate = max(rate)
                    elif isinstance(rate, RandomDistribution):
                        rate = utility_calls.get_maximum_probable_value(
                            rate, pre_vertex_slice.n_atoms)
                    new_weight *= rate
                elif hasattr(pre_vertex, "spikes_per_second"):
                    new_weight *= pre_vertex.spikes_per_second
                weight += new_weight

            self._traffic_weight = weight

    @overrides(AbstractProvidesLocalProvenanceData.get_local_provenance_data)
    def get_local_provenance_data(self):
        prov_items = list()
        for projection in self._projections:
            for synapse_info in projection.synapse_information:
                prov_items.extend(
                    synapse_info.connector.get_provenance_data())
                prov_items.extend(
                    synapse_info.synapse_dynamics.get_provenance_data(
                        self.pre_vertex.label, self.post_vertex.label))
        return prov_items

    def __repr__(self):
        return "{}:{}".format(self.pre_vertex, self.post_vertex)
