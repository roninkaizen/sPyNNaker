from spinn_utilities.overrides import overrides
from spynnaker.pyNN.models.neuron.plasticity.stdp.common.plasticity_helpers import STDP_FIXED_POINT_ONE
from __builtin__ import property
from spynnaker.pyNN.models.neuron.plasticity.stdp.common \
    import plasticity_helpers
from .abstract_timing_dependence import AbstractTimingDependence
from spynnaker.pyNN.models.neuron.plasticity.stdp.synapse_structure\
    import SynapseStructureWeightOnly
from data_specification.enums import DataType

import numpy
import logging
logger = logging.getLogger(__name__)

# LOOKUP_TAU_PLUS_SIZE = 256
# LOOKUP_TAU_PLUS_SHIFT = 0
# LOOKUP_TAU_MINUS_SIZE = 256
# LOOKUP_TAU_MINUS_SHIFT = 0
LOOKUP_TAU_P_SIZE = 256
LOOKUP_TAU_P_SHIFT = 0


class TimingDependenceAbbotSTP(AbstractTimingDependence):

    def __init__(self, STP_type, f, P_baseline, tau_P,
                # unused parameters, but required due to using
                # existing STDP framework
                tau_plus=20.0, tau_minus=20.0):
        AbstractTimingDependence.__init__(self)
#         self._tau_plus = tau_plus
#         self._tau_minus = tau_minus

        self._STP_type = STP_type
        self._f = f
        self._P_baseline = P_baseline
        self._tau_P = tau_P

        self._synapse_structure = SynapseStructureWeightOnly()

        # provenance data
        self._tau_P_last_entry = None # To check transition back to baseline
#         self._tau_plus_last_entry = None
#         self._tau_minus_last_entry = None


    @property
    def STP_type(self):
        return self._STP_type

    @property
    def f(self):
        return self._f

    @property
    def P_baseline(self):
        return self._P_baseline

    @property
    def tau_P(self):
        return self._tau_P

#     @property
#     def tau_plus(self):
#         return self._tau_plus
#
#     @property
#     def tau_minus(self):
#         return self._tau_minus

    @overrides(AbstractTimingDependence.is_same_as)
    def is_same_as(self, timing_dependence):
        if not isinstance(timing_dependence, TimingDependenceSpikePair):
            return False
        return ((self.tau_plus == timing_dependence.tau_plus) and
                (self.tau_minus == timing_dependence.tau_minus))

    @property
    def vertex_executable_suffix(self):
        return "abbot_stp"

    @property
    def pre_trace_n_bytes(self):
        # Organised as array of 32-bit datastructures
        # [0] = [16 bit STDP pre_trace, 16-bit STP P_baseline]
        # [1] = [16-bit STP_trace, 16-bit empty]

        # note that a third entry will be added by synapse_dynamics_stdp_mad
        # [2] = [32-bit time stamp]

        # here we need only account for the first two entries = 4 * 16-bits
        return 8

    @overrides(AbstractTimingDependence.get_parameters_sdram_usage_in_bytes)
    def get_parameters_sdram_usage_in_bytes(self):
        size = 0
        size += 2 * LOOKUP_TAU_P_SIZE # two bytes per lookup table entry
        size += 1 * 4 # 1 parameters at 4 bytes
        return size

    @property
    def n_weight_terms(self):
        return 1

    def write_parameters(self, spec, machine_time_step, weight_scales):

        # Check timestep is valid
        if machine_time_step != 1000:
            raise NotImplementedError(
                "STP LUT generation currently only supports 1ms timesteps")

        # Write STP lookup table
        self._tau_P_last_entry = plasticity_helpers.write_exp_lut(
            spec, self._tau_P, LOOKUP_TAU_P_SIZE,
            LOOKUP_TAU_P_SHIFT)

        # Write rule parameters
        # f
        fixed_point_f = plasticity_helpers.float_to_fixed(
            self._f, plasticity_helpers.STDP_FIXED_POINT_ONE)
        spec.write_value(data=fixed_point_f,
                         data_type=DataType.INT32)


    @property
    def synaptic_structure(self):
        return self._synapse_structure

    def get_provenance_data(self, pre_population_label, post_population_label):
        prov_data = list()
        prov_data.append(plasticity_helpers.get_lut_provenance(
            pre_population_label, post_population_label, "STP_Abbot_Rule",
            "tau_P_last_entry", "tau_P", self._tau_P_last_entry))
        return prov_data

    @overrides(AbstractTimingDependence.get_parameter_names)
    def get_parameter_names(self):
        return ['STP_type', 'f', 'P_baseline','tau_P']

    @overrides(AbstractTimingDependence.initialise_row_headers)
    def initialise_row_headers(self, n_rows, n_header_bytes):
        # note that data is written here as 16-bit quantities, but converted
        # back to 8-bit qunatities for consistency with synaptic row generation
        # code

        # Initialise header structure
        header = numpy.zeros(
            (n_rows, (n_header_bytes/2)), dtype="uint16")

        # Initialise header parameters
        # header[0,0] = int(0.6 * STDP_FIXED_POINT_ONE) # STDP pre_trace
        header[0,1] = int(self._P_baseline * STDP_FIXED_POINT_ONE) # P_Baseline
        header[0,2] = int(self._P_baseline * STDP_FIXED_POINT_ONE) # STP trace
        # header[0,3] = 0 # empty (unused) - only here due to 32-bit packing
        # header[0,4-5] = 32-bit timestamp

        # re-cast as array of 8-bit quantities to facilitate row generation
        return header.view(dtype="uint8")