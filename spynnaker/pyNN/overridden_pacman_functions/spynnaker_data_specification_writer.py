from spinn_utilities.progress_bar import ProgressBar

from spinn_front_end_common.interface.interface_functions import \
    GraphDataSpecificationWriter

from spynnaker.pyNN.models.utility_models import DelayExtensionVertex


class SpynnakerDataSpecificationWriter(GraphDataSpecificationWriter):
    """ Executes data specification generation for sPyNNaker. The main \
    difference from the standard one is that this puts delay extensions \
    after all other vertices.
    """

    __slots__ = ()

    def __init__(self):
        GraphDataSpecificationWriter.__init__(self)

    def __call__(
            self, placements, graph, hostname,
            report_default_directory, write_text_specs,
            app_data_runtime_folder, machine, graph_mapper=None):
        # Sort the delay extensions to the end, looking up the application
        # vertex for each placement in the process.
        pvlist = self._sort_delays_to_end(placements.placements, graph_mapper)

        # create a progress bar for end users
        progress = ProgressBar(
            pvlist, "Generating sPyNNaker data specifications")

        # Keep the results in a dictionary
        dsg_targets = dict()

        # Generate the data for each placed vertex
        for placement, vertex in progress.over(pvlist):
            self._generate_data_spec_for_vertices(
                placement, vertex, dsg_targets, hostname,
                report_default_directory, write_text_specs,
                app_data_runtime_folder, machine)

        return dsg_targets

    @staticmethod
    def _sort_delays_to_end(plist, graph_mapper):
        placements = list()
        delay_extension_placements = list()
        for placement in plist:
            vertex = graph_mapper.get_application_vertex(placement.vertex)
            if isinstance(vertex, DelayExtensionVertex):
                delay_extension_placements.append((placement, vertex))
            else:
                placements.append((placement, vertex))
        placements.extend(delay_extension_placements)
        return placements
