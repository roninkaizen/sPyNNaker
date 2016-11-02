
class SynapseInformation(object):
    """ Contains the synapse information including the connector, synapse type\
        and synapse dynamics
    """

    def __init__(
            self, application_edge, pre_vertex_start, pre_vertex_end,
            pre_population_start, pre_population_end,
            post_vertex_start, post_vertex_end,
            post_population_start, post_population_end):
        self._application_edge = application_edge
        self._pre_vertex_start = pre_vertex_start
        self._pre_vertex_end = pre_vertex_end
        self._pre_population_start = pre_population_start
        self._pre_population_end = pre_population_end
        self._post_vertex_start = post_vertex_start
        self._post_vertex_end = post_vertex_end
        self._post_population_start = post_population_start
        self._post_population_end = post_population_end

    @property
    def application_edge(self):
        return self._application_edge

    @property
    def pre_vertex_start(self):
        return self._pre_vertex_start

    @property
    def pre_vertex_end(self):
        return self._pre_vertex_end

    @property
    def pre_population_start(self):
        return self._pre_population_start

    @property
    def pre_population_end(self):
        return self._pre_population_end

    @property
    def post_vertex_start(self):
        return self._post_vertex_start

    @property
    def post_vertex_end(self):
        return self._post_vertex_end

    @property
    def post_population_start(self):
        return self._post_population_start

    @property
    def post_population_end(self):
        return self._post_population_end
