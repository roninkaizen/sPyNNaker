#ifndef _TIMING_PAIR_IMPL_H_
#define _TIMING_PAIR_IMPL_H_

//---------------------------------------
// Typedefines
//---------------------------------------
typedef int16_t post_trace_t;
typedef int16_t pre_trace_t;

#include "../synapse_structure/synapse_structure_weight_impl.h"
#include "timing.h"
#include "../weight_dependence/weight_one_term.h"

// Include debug header for log_info etc
#include <debug.h>

// Include generic plasticity maths functions
#include "../../common/maths.h"
#include "../../common/stdp_typedefs.h"

//---------------------------------------
// Macros
//---------------------------------------
// Exponential decay lookup parameters
#define TAU_PLUS_TIME_SHIFT 0
#define TAU_PLUS_SIZE 256

#define TAU_MINUS_TIME_SHIFT 0
#define TAU_MINUS_SIZE 256

#define TAU_P_TIME_SHIFT 0
#define TAU_P_SIZE 256

// Helper macros for looking up decays
//#define DECAY_LOOKUP_TAU_PLUS(time) \
//    maths_lut_exponential_decay( \
//        time, TAU_PLUS_TIME_SHIFT, TAU_PLUS_SIZE, tau_plus_lookup)
//#define DECAY_LOOKUP_TAU_MINUS(time) \
//    maths_lut_exponential_decay( \
//        time, TAU_MINUS_TIME_SHIFT, TAU_MINUS_SIZE, tau_minus_lookup)
#define DECAY_LOOKUP_TAU_P(time) \
    maths_lut_exponential_decay( \
        time, TAU_P_TIME_SHIFT, TAU_P_SIZE, tau_P_lookup)
//---------------------------------------
// Structures
//---------------------------------------
typedef struct {
    int32_t f;
} stp_params_t;


//---------------------------------------
// Externals
//---------------------------------------
//extern int16_t tau_plus_lookup[TAU_PLUS_SIZE];
//extern int16_t tau_minus_lookup[TAU_MINUS_SIZE];
extern int16_t tau_P_lookup[TAU_P_SIZE];
extern stp_params_t STP_params;

//---------------------------------------
// STP Inline functions
//---------------------------------------
static inline pre_trace_t timing_apply_stp(
        uint32_t time, uint32_t last_time, stp_trace_t last_stp_trace) {

	// This function is called once per synaptic row, so update multiplier
	// for entire row here - using time since last pre spike

	// Get time since last spike
	uint32_t delta_time = time - last_time;

	// Decay previous stp trace
	int32_t decayed_one = STDP_FIXED_MUL_16X16(last_stp_trace,
	            DECAY_LOOKUP_TAU_P(delta_time));

	log_info("\n old STP trace: %k \n delta_t: %u \n decayed STP trace: %k",
			last_stp_trace << 4,
			delta_time,
			decayed_one << 4);

	// Now add one - if trace was decayed to zero, this will scale the weight by 1
	return STDP_FIXED_POINT_ONE + decayed_one;
}


//---------------------------------------
// Timing dependence inline functions
//---------------------------------------
static inline post_trace_t timing_get_initial_post_trace() {
    return 0;
}

//---------------------------------------
static inline post_trace_t timing_add_post_spike(
        uint32_t time, uint32_t last_time, post_trace_t last_trace) {
	// No post-synaptic dependence in STP

	use(&time);
	use(&last_time);
    return last_trace;
}

//---------------------------------------
static inline pre_trace_t timing_add_pre_spike(
        uint32_t time, uint32_t last_time, pre_trace_t last_trace) {
	use(&time);
	use(&last_time);

	// do nothing as no STDP
    return last_trace;
}

//---------------------------------------
static inline update_state_t timing_apply_pre_spike(
        uint32_t time, pre_trace_t trace, uint32_t last_pre_time,
        pre_trace_t last_pre_trace, uint32_t last_post_time,
        post_trace_t last_post_trace, update_state_t previous_state) {
    use(&time);
    use(&trace);
    use(last_pre_time);
    use(&last_pre_trace);
    use(&last_post_time);
    use(&last_post_trace);

    return previous_state;

}

//---------------------------------------
static inline update_state_t timing_apply_post_spike(
        uint32_t time, post_trace_t trace, uint32_t last_pre_time,
        pre_trace_t last_pre_trace, uint32_t last_post_time,
        post_trace_t last_post_trace, update_state_t previous_state) {
    use(&time);
    use(&trace);
    use(&last_pre_time);
    use(&last_pre_trace);
    use(&last_post_time);
    use(&last_post_trace);
    // No post-synaptic dependence in STP
        return previous_state;
}

#endif // _TIMING_PAIR_IMPL_H_
