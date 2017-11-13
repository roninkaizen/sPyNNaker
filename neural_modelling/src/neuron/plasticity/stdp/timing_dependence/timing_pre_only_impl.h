#ifndef _TIMING_PRE_ONLY_IMPL_H_
#define _TIMING_PRE_ONLY_IMPL_H_

//---------------------------------------
// Typedefines
//---------------------------------------
typedef int16_t post_trace_t;
typedef int16_t pre_trace_t;

#include "../synapse_structure/synapse_structure_weight_impl.h"
#include "timing.h"
#include "../weight_dependence/weight_one_term.h"
#include "../../../additional_inputs/additional_input_ca2_concentration_impl.h"

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

// Helper macros for looking up decays
//#define DECAY_LOOKUP_TAU_PLUS(time) \
//    maths_lut_exponential_decay( \
//        time, TAU_PLUS_TIME_SHIFT, TAU_PLUS_SIZE, tau_plus_lookup)
//#define DECAY_LOOKUP_TAU_MINUS(time) \
//    maths_lut_exponential_decay( \
//        time, TAU_MINUS_TIME_SHIFT, TAU_MINUS_SIZE, tau_minus_lookup)

//---------------------------------------
// Externals
//---------------------------------------
//extern int16_t tau_plus_lookup[TAU_PLUS_SIZE];
//extern int16_t tau_minus_lookup[TAU_MINUS_SIZE];
REAL th_v_mem;
// Ca thresholds
REAL th_ca_up_l;
REAL th_ca_up_h;
REAL th_ca_dn_l;
REAL th_ca_dn_h;
//---------------------------------------
// Timing dependence inline functions
//---------------------------------------
static inline post_trace_t timing_get_initial_post_trace() {
    return 0;
}

//---------------------------------------
// unused / empty ?
//----------------------------------------
static inline post_trace_t timing_add_post_spike(
        uint32_t time, uint32_t last_time, post_trace_t last_trace) {
//
//    // Get time since last spike
//    uint32_t delta_time = time - last_time;
//
//    // Decay previous o1 and o2 traces
//    int32_t decayed_o1_trace = STDP_FIXED_MUL_16X16(last_trace,
//            DECAY_LOOKUP_TAU_MINUS(delta_time));
//
//    // Add energy caused by new spike to trace
//    // **NOTE** o2 trace is pre-multiplied by a3_plus
//    int32_t new_o1_trace = decayed_o1_trace + STDP_FIXED_POINT_ONE;
//
//    log_debug("\tdelta_time=%d, o1=%d\n", delta_time, new_o1_trace);

    // Return new pre- synaptic event with decayed trace values with energy
    // for new spike added
    return (post_trace_t) last_trace;
}

//---------------------------------------
// also not needed?
//--------------------------------------
static inline pre_trace_t timing_add_pre_spike(
        uint32_t time, uint32_t last_time, pre_trace_t last_trace) {

//    // Get time since last spike
//    uint32_t delta_time = time - last_time;
//
//    // Decay previous r1 and r2 traces
//    int32_t decayed_r1_trace = STDP_FIXED_MUL_16X16(
//        last_trace, DECAY_LOOKUP_TAU_PLUS(delta_time));
//
//    // Add energy caused by new spike to trace
//    int32_t new_r1_trace = decayed_r1_trace + STDP_FIXED_POINT_ONE;
//
//    log_debug("\tdelta_time=%u, r1=%d\n", delta_time, new_r1_trace);
//
//    // Return new pre-synaptic event with decayed trace values with energy
//    // for new spike added
    return (pre_trace_t) last_trace;
}

//---------------------------------------
static inline update_state_t timing_apply_pre_spike(
        uint32_t time, pre_trace_t trace, uint32_t last_pre_time,
        pre_trace_t last_pre_trace, uint32_t last_post_time,
        post_trace_t last_post_trace, update_state_t previous_state,
		neuron_pointer_t post_synaptic_neuron,
		additional_input_pointer_t post_synaptic_additional_input) {

    use(&trace);
    use(last_pre_time);
    use(&last_pre_trace);

    int dt = time - last_pre_time;
    int32_t w =  previous_state.initial_weight;
    int32_t th_w = previous_state.weight_region->th_weight;
    int32_t w_drift = previous_state.weight_region->weight_drift;

    log_info("Ca concentration: %12.6k", post_synaptic_additional_input->I_Ca2);
    log_info("Ca alpha: %12.6k", post_synaptic_additional_input->I_alpha);
    log_info("Ca tau multiplier: %u", post_synaptic_additional_input->exp_TauCa);

    if(w>th_w){
    	log_info("drifting up w_drift: %d, dt: %d", w_drift, dt);
    	w += w_drift * dt;
    }else{
    	log_info("drifting down w_drift: %d, dt: %d", w_drift, dt);
    	w -= w_drift * dt;
    }

    previous_state.initial_weight = w;

    REAL I_Ca2 = post_synaptic_additional_input->I_Ca2;

    log_info("Ca: %12.6k, dn_l:  %12.6k, dn_h:  %12.6k", I_Ca2, th_ca_dn_l, th_ca_dn_h);

    if (neuron_model_get_membrane_voltage(post_synaptic_neuron) > th_v_mem && I_Ca2 > th_ca_up_l && I_Ca2 < th_ca_up_h ){
    	log_info("above threshold, in ca range");
    	return weight_one_term_apply_potentiation(previous_state, last_pre_trace);

    } else if (neuron_model_get_membrane_voltage(post_synaptic_neuron) <= th_v_mem && I_Ca2 > th_ca_dn_l && I_Ca2 < th_ca_dn_h ){
    	log_info("below threshold, in ca range");
    	return weight_one_term_apply_depression(previous_state, last_pre_trace);

    } else {
    	log_info("not in the ca range");

    }
    // get V_post(t_pre)
    // get Ca_post(t_pre)
    // get A+, A-
    // thresholds can be global values, but potentially may have to be per synapse or neuron
    // integrate drifts
    // if V>th_V and Ca in up range : apply potentiation
    // if V<=th_V and Ca in down range : apply depression

    // missing params: th_v, up range, down range, th_w (drifts), 2 drift rates

    return previous_state;
}

//---------------------------------------
// unused / empty ?
//-------------------------------------
static inline update_state_t timing_apply_post_spike(
        uint32_t time, post_trace_t trace, uint32_t last_pre_time,
        pre_trace_t last_pre_trace, uint32_t last_post_time,
        post_trace_t last_post_trace, update_state_t previous_state, neuron_pointer_t post_synaptic_neuron) {
    use(&trace);
    use(last_post_time);
    use(&last_post_trace);
//
//    // Get time of event relative to last pre-synaptic event
//    uint32_t time_since_last_pre = time - last_pre_time;
//    if (time_since_last_pre > 0) {
//        int32_t decayed_r1 = STDP_FIXED_MUL_16X16(
//            last_pre_trace, DECAY_LOOKUP_TAU_PLUS(time_since_last_pre));
//
//        log_debug("\t\t\ttime_since_last_pre_event=%u, decayed_r1=%d\n",
//                  time_since_last_pre, decayed_r1);
//
//        // Apply potentiation to state (which is a weight_state)
//        return weight_one_term_apply_potentiation(previous_state, decayed_r1);
//    } else {
        return previous_state;
//    }
}

#endif // _TIMING_PAIR_IMPL_H_