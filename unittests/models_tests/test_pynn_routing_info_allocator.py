import unittest
from pacman.model.partitioned_graph.partitioned_graph import PartitionedGraph
from pacman.model.partitioned_graph.partitioned_edge import PartitionedEdge
from pacman.model.partitioned_graph.partitioned_vertex import PartitionedVertex

from pacman.model.graph_mapper.graph_mapper \
    import GraphMapper
from spynnaker.pyNN.overridden_pacman_functions.pynn_routing_info_allocator \
    import PyNNRoutingInfoAllocator
from pacman.model.placements.placement import Placement
from pacman.model.placements.placements import Placements


class TestPyNNRoutingInfoAllocator(unittest.TestCase):
    def test_key_mask_combo(self):
        ria = PyNNRoutingInfoAllocator()
        self.assertEqual(ria.get_key_mask_combo(0xf1f2, 0x00ff), 0xf2)
        self.assertEqual(ria.get_key_mask_combo(0xf1f2, 0x0f0f), 0x0102)
        self.assertEqual(ria.get_key_mask_combo(0xf1f2, 0xf00f), 0xf002)
        self.assertEqual(ria.get_key_mask_combo(0xf1f2, 0xf0f0), 0xf0f0)
        self.assertEqual(ria.get_key_mask_combo(0xf1f2, 0xffff), 0xf1f2)

    @unittest.skip("Test out of date")
    def test_add_subgraph_and_placement(self):
        gsm = GraphMapper()
        ria = PyNNRoutingInfoAllocator()
        subvertices = list()
        subedges = list()
        for i in range(10):
            subvertices.append(PartitionedVertex(None, ""))
        for i in range(5):
            subedges.append(
                PartitionedEdge(subvertices[0], subvertices[(i + 1)]))
        for i in range(5, 10):
            subedges.append(
                PartitionedEdge(subvertices[5], subvertices[(i + 1) % 10]))
        subgraph = PartitionedGraph("", subvertices, subedges)

        subv = PartitionedVertex(None, "")
        pl = Placement(subv, 0, 0, 1)
        pls = Placements([pl])
        ria.allocate_routing_info(subgraph, pls)

    @unittest.skip("Test out of date")
    def test_add_subgraph_and_placement_with_proper_graph_subgraph_mapper(self):

        subvertices = list()
        subedges = list()
        gsm = GraphMapper()
        for i in range(10):
            subvertices.append(PartitionedVertex(None, ""))
            gsm.add_subvertex(subvertices[0], i * 10, (i + 1) * 10 - 1, None)
        for i in range(5):
            subedges.append(
                PartitionedEdge(subvertices[0], subvertices[(i + 1)]))
        for i in range(5, 10):
            subedges.append(
                PartitionedEdge(subvertices[5], subvertices[(i + 1) % 10]))
        subgraph = PartitionedGraph("", subvertices, subedges)
        ria = PyNNRoutingInfoAllocator()
        subv = PartitionedVertex(None, "")
        pl = Placement(subv, 0, 0, 1)
        pls = Placements([pl])
        ria.allocate_routing_info(subgraph, pls)

    def test_check_masks(self):
        pass


if __name__ == '__main__':
    unittest.main()
