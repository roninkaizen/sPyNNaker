# front end common imports
from spinn_front_end_common.utilities import exceptions
from spinn_front_end_common.abstract_models.abstract_changable_after_run \
    import AbstractChangableAfterRun

# pynn imports
from pyNN import descriptions
from pyNN import space

# spynnaker imports
from spynnaker.pyNN.models.neuron_cell import \
    NeuronCell
from spynnaker.pyNN.models.base_pynn_container import BasePyNNContainer
from spynnaker.pyNN.models.neuron_cell import RecordingType
from spynnaker.pyNN.models.neuron.synapse_dynamics.synapse_dynamics_static \
    import SynapseDynamicsStatic
from spynnaker.pyNN.models.neuron.builds.abstract_unsupported_neuron_build \
    import AbstractUnsupportedNeuronBuild

# pacman imports
from pacman.model.constraints.placer_constraints\
    .placer_chip_and_core_constraint import PlacerChipAndCoreConstraint
from pacman.model.constraints.partitioner_constraints\
    .partitioner_maximum_size_constraint \
    import PartitionerMaximumSizeConstraint
from pacman.model.constraints.abstract_constraint import AbstractConstraint
from pacman.model.decorators.overrides import overrides

# general imports
import logging
import numpy

logger = logging.getLogger(__name__)

# Lines for objects
# Assembly 42
# PopView 648
# Population 1142


class Assembly(BasePyNNContainer):
    """ A view of a collection of populations/ population views / assembles
    """

    count = 0

    def __init__(self, populations, spinnaker, label=None):

        BasePyNNContainer.__init__(self, spinnaker, label)
        if label is None:
            self._label = "assembly{}".format(Assembly.count)
        Assembly.count += 1

        for container in populations:
            for population, population_slice in zip(
                    container._populations, container._population_slices):
                self._add_container(population, population_slice)

    def __getitem__(self, index):
        """ An assembly containing a subset of cells

        :param index: the index or slice of the cells
        :return: An Assembly
        """

        # If just a single cell, return the cell
        if isinstance(index, int):
            return self._cells[index]

        # Get a boolean array for indicating which cells are in the index
        neuron_filter = numpy.zeros(len(self._cells), dtype="bool")
        neuron_filter[index] = True

        populations = list()
        for population, population_slice, (start, end) in zip(
                self._populations, self._population_slices,
                self._populations_start_and_end):

            # Work out where the slice of the population starts in the
            # population by creating an index of cells in the population and
            # using the slice against it, and getting the first index
            slice_start = numpy.arange(len(population))[population_slice][0]

            # Get the indices of the neurons to be included relative to the
            # population base index
            neuron_indices = (
                numpy.where(neuron_filter[start:end])[0] +
                (start - slice_start)
            )

            # Create a new global filter for the population using the indices
            population_filter = numpy.zeros(len(population), dtype="bool")
            population_filter[neuron_indices] = True

            # Make a new population view containing the indices
            populations.append(PopulationView(
                population, population_filter, self.spinnaker))

        return Assembly(populations, self._spinnaker)

    def __add__(self, other):
        """ Join a cell container to this and return the resulting assembly

        :param other: other population, population view or assembly.
        :return:
        """

        # Create a list of all the existing populations and add them, with
        # other to a new assembly
        # Note that even if other == self, this should work
        new_list = list(self._containers)
        new_list.append(other)
        return Assembly(new_list, self._spinnaker)

    def __iadd__(self, other):
        """ Add a Population or PopulationView to this Assembly

        :param other:  pop or pop view or assembly
        :return: None
        """
        for population, population_slice in zip(
                other._populations, other._population_slices):
            self._add_container(population, population_slice)

    def describe(self, template='assembly_default.txt', engine='default'):
        """ Get a human readable description of the assembly.

        The output may be customised by specifying a different template
        together with an associated template engine (see ``pyNN.descriptions``)

        If template is None, then a dictionary containing the template context
        will be returned.

        :param template: the different format to write
        :param engine: the writer for the template.
        """
        context = {
            "label": self._label,
            "populations":
                [p.describe(template=None)
                 for p in self._populations]}
        return descriptions.render(engine, template, context)

    def get_population(self, label):
        """ Get a population by label

        :param label: the label of the population/population_view to find
        :return: the population or population_view
        :raises: KeyError if no population exists
        """
        for pop in self._populations:
            if pop.label == label:
                return label
        raise KeyError(
            "Population / Population view does not exist in this Assembly")

    def populations(self):
        return self._populations


class PopulationView(BasePyNNContainer):
    """ A view of a subset of cells from a population
    """

    def __init__(
            self, parent, selector, spinnaker, label=None):
        """
        :param parent: The parent population object
        :param selector: The filter for the neurons
        :type selector:\
            iterable of booleans, iterable of ints, or a slice
        :param spinnaker:
        :param label:
        """
        BasePyNNContainer.__init__(
            self, spinnaker, [parent], [selector], label)

        self._parent = parent
        self._mask = selector

        # update label accordingly
        if label is None:
            self._label = "view of {} with mask {}".format(
                parent.label, selector)
            spinnaker.increment_none_labelled_pop_view_count()

    def __add__(self, other):
        """ Add another container to this one and return an Assembly

        :param other: Other container to add
        :return:
        """

        # build assembly
        return Assembly(populations=[self, other], spinnaker=self._spinnaker)

    def __getitem__(self, index):
        """ Get a single cell or population view of the subset of this view

        :param index: either a index or a slice.
        :return: a NeuronCell object or a PopulationView object
        """

        # If just a single item, return it
        if isinstance(index, int):
            return self._cells[index]

        # Get a boolean array for indicating which cells are in the index
        neuron_filter = numpy.zeros(len(self._cells), dtype="bool")
        neuron_filter[index] = True

        # Work out where the slice of the population starts in the
        # population by creating an index of cells in the population and
        # using the slice against it, and getting the first index
        slice_start = numpy.arange(len(self._parent))[self._mask][0]

        # Get the indices of the neurons to be included relative to the
        # population base index
        neuron_indices = (
            numpy.where(neuron_filter)[0] + slice_start
        )

        # Create a new global filter for the population using the indices
        population_filter = numpy.zeros(len(self._cells), dtype="bool")
        population_filter[neuron_indices] = True

        # Make a new population view containing the indices
        return PopulationView(
            self._parent, population_filter, self.spinnaker)

    def describe(
            self, template='populationview_default.txt', engine='default'):
        context = {"label": self.label,
                   "parent": self._parent.label,
                   "size": self.size,
                   "mask": self.mask}

        return descriptions.render(engine, template, context)

    @property
    def grandparent(self):
        return self._parent

    def nearest(self, position):
        """ Get the cell nearest to the given position

        :param position: the space position
        :return:
        """

        # return closest position
        return self._parent._nearest(position, self.positions)

    @property
    def mask(self):
        return self._mask

    @property
    def parent(self):
        return self._parent

    def randomInit(self, rand_distr):
        """ Randomise the membrane voltage of the cells

        :param rand_distr: the random distribution used for initialising v
        :return: None
        """
        self.initialize("v", rand_distr)

    @property
    def structure(self):
        return self._parent.structure


class Population(BasePyNNContainer):
    """ A collection neuron of the same types. It encapsulates a type of\
        vertex used with Spiking Neural Networks, comprising n cells (atoms)\
        of the same model type.
    """

    nPop = 0

    def __init__(
            self, size, cellclass, cellparams, spinnaker, label=None,
            structure=None, max_neurons_per_core=None):
        """

        :param int size:\
            size (number of cells) of the Population.
        :param cellclass:\
            specifies the neural model to use for the Population
        :param dict cellparams:\
            a dictionary containing model specific parameters and values
        :param structure:\
            a spatial structure
        :param string label:\
            a label identifying the Population
        :returns a list of vertexes and edges
        """
        if (size is not None and size <= 0) or size is None:
            raise exceptions.ConfigurationException(
                "A population cannot have a negative, None or zero size.")

        BasePyNNContainer.__init__(self, spinnaker, label=label)

        # Check for unimplemented models
        if isinstance(cellclass, AbstractUnsupportedNeuronBuild):
            raise exceptions.ConfigurationException(
                "The cell class {} is currently unimplemented in sPyNNaker"
                .format(cellclass))

        # Check for unchecked PyNN models
        # (those without the @pyNN_model decorator - this decorator checks that
        # the model is compatible with the following code)
        if not hasattr(cellclass, "__is_pyNN_model__"):
            raise exceptions.ConfigurationException(
                "The cell class {} does not appear to be a PyNN model."
                " Please decorate it with the"
                " spynnaker.pyNN.models.pyNN_model.pyNNModel decorator.")

        # Set the label
        if label is None:
            self._label = "Population {}".format(Population.nPop)
        Population.nPop += 1

        # Set up the internal objects
        self._populations.append(self)
        self._population_slices.append(slice(None))
        self._populations_start_and_end.append((0, size))
        self._class = cellclass
        self._structure = structure or space.Line()
        self._requires_remapping = True

        # Keep track of vertices containing cells of the population
        # This is a list of:
        #  (vertex, vertex_start, vertex_end, population_start, population_end)
        # where:
        #  vertex - the vertex containing some cells from this population
        #  vertex_start - the first atom of the vertex which is a cell from
        #                 this population
        #  vertex_end - the last atom of the vertex which is a cell from this
        #               population
        #  population_start - the index of the cell at vertex_start in the
        #                     population
        #  population_end - the index of the cell at vertex_end in the
        #                   population
        self._vertices = list()

        # Keep track of constraints on the populations to be added to the
        # vertices
        self._constraints = list()
        if max_neurons_per_core is not None:
            self._constraints.append(PartitionerMaximumSizeConstraint(
                max_neurons_per_core))

        # Store "population parameter" settings
        if hasattr(cellclass, "population_parameters"):
            self._population_parameters = {
                param: cellparams.get(param)
                for param in cellclass.population_parameters
            }
        else:
            self._population_parameters = dict()

        # Create the cells, minus the population parameters
        final_cellparams = {
            key: value for key, value in cellparams.iteritems()
            if key not in self._population_parameters
        }
        self._cells = [NeuronCell(cellclass) for _ in range(size)]
        self.set(final_cellparams)

        # Generate the positions an set them
        self.positions = self._structure.generate_positions(size)

        # Keep the synapse dynamics required to be processed by the
        # population later
        self._synapse_dynamics = SynapseDynamicsStatic()

    @property
    def celltype(self):
        return self._class

    def add_vertex(
            self, vertex, vertex_start, vertex_end,
            population_start, population_end):
        self._vertices.append(
            (vertex, vertex_start, vertex_end, population_start,
             population_end)
        )

    @property
    def constraints(self):
        return self._constraints

    @property
    def requires_mapping(self):
        """ Determine if any cell requires re-mapping

        :return: boolean
        """
        if self._requires_remapping:
            return True

        is_changable_after_run = False
        for (vertex, _, _, _, _) in self._vertices:
            if isinstance(vertex, AbstractChangableAfterRun):
                is_changable_after_run = True
                if vertex.requires_mapping:
                    return True

        if not is_changable_after_run:
            return any([cell.requires_mapping for cell in self._cells])

    def mark_no_changes(self):
        """ Inform all cells to start re tracking changes from now on
        """
        self._requires_remapping = False
        is_changable_after_run = False
        for (vertex, _, _, _, _) in self._vertices:
            if isinstance(vertex, AbstractChangableAfterRun):
                vertex.mark_no_changes()
                is_changable_after_run = True

        if not is_changable_after_run:
            for cell in self._cells:
                cell.mark_no_changes()

    def __add__(self, other):
        """ Get an assembly of this Population and another Population,\
            PopulationView or Assembly
        """
        return Assembly([self, other], self._spinnaker)

    def describe(self, template='population_default.txt', engine='default'):
        context = {
            "label": self._pop_label,
            "celltype": self._class.model_name(),
            "structure": None,
            "size": self.size,
            "first_id": 0,
            "last_id": self.size - 1,
        }

        if self.structure:
            context["structure"] = self.structure.describe(template=None)
        return descriptions.render(engine, template, context)

    def __getitem__(self, index):
        """ Get a single cell or a PopulationView of cells

        :param index: the slice or index
        :return: a cell or a pop view
        """
        # If just a single item, return it
        if isinstance(index, int):
            return self._cells[index]
        return PopulationView(self, index, self._spinnaker)

    def nearest(self, position):
        """ Return the neuron closest to the specified position

        :param position: space position
        """
        return self._nearest(position, self.positions)

    def _nearest(self, position, positions):
        """ Return the neuron closest to the specified position

        :param position: space position
        """
        # doesn't always work correctly if a position is equidistant between
        # two neurons, i.e. 0.5 should be rounded up, but it isn't always.
        # also doesn't take account of periodic boundary conditions
        pos = numpy.array([position] * positions.shape[1]).transpose()
        dist_arr = (positions - pos) ** 2
        distances = dist_arr.sum(axis=0)
        nearest = distances.argmin()
        return self[nearest]

    @staticmethod
    def _generate_positions_for_atoms(atoms):
        positions = None
        used_structure = None
        for atom_index in range(0, len(atoms)):
            atom = atoms[atom_index]
            if atom.position is None:
                if atom.structure is None:
                    raise ValueError("attempted to retrieve positions "
                                     "for an unstructured population")

                # get positions as needed
                if atom_index == 0:
                    positions = atom.structure.generate_positions(len(atoms))
                    used_structure = atom.structure
                elif atom.structure != used_structure:
                    raise exceptions.ConfigurationException(
                        "Atoms in the population have different "
                        "structures, this is considered an error here.")

                # update atom with position
                atom.position = positions[atom_index]
        return positions

    @property
    def structure(self):
        """ Return the structure for the population.
        """
        return self._structure

    # NONE PYNN API CALL
    def set_constraint(self, constraint):
        """ Apply a constraint to a population that restricts the processor\
            onto which its sub-populations will be placed.
        """
        if isinstance(constraint, AbstractConstraint):
            self._constraints.append(constraint)
        else:
            raise exceptions.ConfigurationException(
                "the constraint entered is not a recognised constraint")
        self._requires_remapping = True

    # NONE PYNN API CALL
    def add_placement_constraint(self, x, y, p=None):
        """ Add a placement constraint

        :param x: The x-coordinate of the placement constraint
        :type x: int
        :param y: The y-coordinate of the placement constraint
        :type y: int
        :param p: The processor id of the placement constraint (optional)
        :type p: int
        """
        self._constraints.append(PlacerChipAndCoreConstraint(x, y, p))
        self._requires_remapping = True

    # NONE PYNN API CALL
    def set_mapping_constraint(self, constraint_dict):
        """ Add a placement constraint - for backwards compatibility

        :param constraint_dict: A dictionary containing "x", "y" and\
                    optionally "p" as keys, and ints as values
        :type constraint_dict: dict of str->int
        """
        self.add_placement_constraint(**constraint_dict)
        self._requires_remapping = True

    # NONE PYNN API CALL
    def set_number_of_neurons_per_core(self, new_value):
        """ Supports the setting of the population maximum neurons per core

        :param new_value: the new value for the max atoms per core.
        """
        self._constraints.add(PartitionerMaximumSizeConstraint(new_value))
        self._requires_remapping = True

    def _end(self):
        """ Do final steps at the end of the simulation
        """
        if any([cell.is_recording_to_file(RecordingType.SPIKES)
                for cell in self._cells]):
            self.printSpikes("spikes")
        if any([cell.is_recording_to_file(RecordingType.V)
                for cell in self._cells]):
            self.print_v("v")
        if any([cell.is_recording_to_file(RecordingType.GSYN)
                for cell in self._cells]):
            self.print_gsyn("gsyn")

    @overrides(BasePyNNContainer.set)
    def set(self, param, val=None):
        if isinstance(param, dict):
            for name, value in param.iteritems():
                if name in self._population_parameters:
                    self._population_parameters[name] = value
                else:
                    BasePyNNContainer.set(self, name, value)
        else:
            if param in self._population_parameters:
                self._population_parameters[param] = val
            else:
                BasePyNNContainer.set(self, param, val)

    @overrides(BasePyNNContainer.get)
    def get(self, parameter_name, gather=False):
        if parameter_name in self._population_parameters:
            return self._population_parameters[parameter_name]
        return BasePyNNContainer.get(parameter_name, gather)

    def set_synapse_dynamics(self, synapse_dynamics):
        """ Set the synapse dynamics of this population
            Checks that the new dynamics is compatible with the current one\
            if one exists
        :param synapse_dynamics: new synapse_dynamics
        """

        # We can always override static dynamics or None
        if isinstance(self._synapse_dynamics, SynapseDynamicsStatic):
            self._synapse_dynamics = synapse_dynamics

        # We can ignore a static dynamics trying to overwrite a plastic one
        elif isinstance(synapse_dynamics, SynapseDynamicsStatic):
            pass

        # Otherwise, the dynamics must be equal
        elif not synapse_dynamics.is_same_as(self._synapse_dynamics):
            raise exceptions.ConfigurationException(
                "Synapse dynamics must match exactly when using multiple edges"
                "to the same population")
