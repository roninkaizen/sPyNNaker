class SynapseRetrevalInformation(object):
    """ Information to make it possible to retrieve synaptic data
    """

    def __init__(
            self, post_vertex, machine_edge, index, delay_index,
            synapse_information):
        self._post_vertex = post_vertex
        self._machine_edge = machine_edge
        self._index = index
        self._delay_index = delay_index
        self._synapse_information = synapse_information

    @property
    def post_vertex(self):
        return self._post_vertex

    @property
    def machine_edge(self):
        return self._machine_edge

    @property
    def index(self):
        return self._index

    @property
    def delay_index(self):
        return self._delay_index

    @property
    def synapse_information(self):
        return self._synapse_information
