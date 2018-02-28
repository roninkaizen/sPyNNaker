#include "timing_abbot_impl.h"

//---------------------------------------
// Globals
//---------------------------------------
// Exponential lookup-tables
//int16_t tau_plus_lookup[TAU_PLUS_SIZE];
//int16_t tau_minus_lookup[TAU_MINUS_SIZE];
int16_t tau_P_lookup[TAU_P_SIZE];
stp_params_t STP_params;

//---------------------------------------
// Functions
//---------------------------------------
address_t timing_initialise(address_t address) {

    log_info("timing_initialise: starting");
    log_info("\tAbbot STP rule");
    // **TODO** assert number of neurons is less than max

    // Copy LUTs from following memory
    address_t next_param_address = maths_copy_int16_lut(&address[0], TAU_P_SIZE,
                                                 &tau_P_lookup[0]);

    // Copy parameters
    STP_params.f = (int32_t) next_param_address[0];

    log_info("Parameters: \n \t f = %k", STP_params.f << 4);
    log_info("STP memory initialisation completed successfully");

    return next_param_address[1];
}
