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
#define DECAY_LOOKUP_TAU_P(time) \
    maths_lut_exponential_decay( \
        time, TAU_P_TIME_SHIFT, TAU_P_SIZE, tau_P_lookup)
//---------------------------------------
// Structures
//---------------------------------------
typedef struct {
	int32_t stp_type;
    int32_t f;
} stp_params_t;


//---------------------------------------
// Externals
//---------------------------------------
extern int16_t tau_P_lookup[TAU_P_SIZE];
extern stp_params_t STP_params;

//---------------------------------------
// STP Inline functions
//---------------------------------------

static inline stp_trace_t timing_decay_stp_trace(
        uint32_t time, uint32_t last_time, stp_trace_t last_stp_trace, uint16_t P_Baseline) {

	// This function is called once per synaptic row, so update multiplier
	// for entire row here - using time since last pre spike

	// Get time since last spike
	uint32_t delta_time = time - last_time;

	int32_t decayed_one;
	if (STP_params.stp_type==0){ // todo: mask compare to only look at first bit
		// Decay previous stp trace UP to baseline
		decayed_one = P_Baseline - STDP_FIXED_MUL_16X16(P_Baseline - last_stp_trace,
	            DECAY_LOOKUP_TAU_P(delta_time));
	} else {
		// Decay previous stp trace DOWN to baseline
		decayed_one = P_Baseline + STDP_FIXED_MUL_16X16(last_stp_trace - P_Baseline,
	            DECAY_LOOKUP_TAU_P(delta_time));
	} // note that two functions are required to swap update to ensure integers don't wrap

//	log_info("Decaying STP trace: "
//			"\n old STP trace: %k "
//			"\n time: %u "
//			"\n delta_t: %u "
//			"\n decayed STP trace: %k",
//			last_stp_trace << 4,
//			time,
//			delta_time,
//			decayed_one << 4);


	// Now add one - if trace was decayed to zero, this will scale the weight by 1
	return decayed_one;
}

static inline stp_trace_t timing_apply_stp_spike(
        uint32_t time, uint32_t last_time, stp_trace_t last_stp_trace, uint16_t P_Baseline) {
	use(time);
	use(last_time);
	use(P_Baseline);

	if (STP_params.stp_type == 0){ // todo: mask compare to only look at first bit
		// depress
		log_info("depressing");
		return last_stp_trace - STDP_FIXED_MUL_16X16(
			STP_params.f, last_stp_trace);
	} else { // STP_params.stp_type = 1
		// Potentiate
		log_info("potentiating");
		return last_stp_trace + STDP_FIXED_MUL_16X16(
			STP_params.f, (STDP_FIXED_POINT_ONE - last_stp_trace));
	}
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
