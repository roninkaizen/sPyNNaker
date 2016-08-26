#ifndef _TIMING_PFISTER_TRIPLET_IMPL_H_
#define _TIMING_PFISTER_TRIPLET_IMPL_H_

//---------------------------------------
// Structures
//---------------------------------------
typedef int32_t post_trace_t;

typedef int32_t pre_trace_t;

#include "timing.h"
#include "../weight_dependence/weight_two_term.h"

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

#define TAU_X_TIME_SHIFT 2
#define TAU_X_SIZE 256

#define TAU_Y_TIME_SHIFT 2
#define TAU_Y_SIZE 256

// Helper macros for looking up decays
#define DECAY_LOOKUP_TAU_PLUS(time) \
    maths_lut_exponential_decay( \
        time, TAU_PLUS_TIME_SHIFT, TAU_PLUS_SIZE, tau_plus_lookup)
#define DECAY_LOOKUP_TAU_MINUS(time) \
    maths_lut_exponential_decay( \
        time, TAU_MINUS_TIME_SHIFT, TAU_MINUS_SIZE, tau_minus_lookup)

#define DECAY_LOOKUP_TAU_X(time) \
    maths_lut_exponential_decay( \
        time, TAU_X_TIME_SHIFT, TAU_X_SIZE, tau_x_lookup)
#define DECAY_LOOKUP_TAU_Y(time) \
    maths_lut_exponential_decay( \
        time, TAU_Y_TIME_SHIFT, TAU_Y_SIZE, tau_y_lookup)

//---------------------------------------
// Externals
//---------------------------------------
extern int16_t tau_plus_lookup[TAU_PLUS_SIZE];
extern int16_t tau_minus_lookup[TAU_MINUS_SIZE];
extern int16_t tau_x_lookup[TAU_X_SIZE];
extern int16_t tau_y_lookup[TAU_Y_SIZE];

//---------------------------------------
// Triplet helper functions
//---------------------------------------
static inline int32_t triplet_trace_get_trace1(int32_t trace)
{
  return (trace & 0xFFFF);
}
//---------------------------------------
static inline int32_t triplet_trace_get_trace2(int32_t trace)
{
  return (trace >> 16);
}
//---------------------------------------
static inline int32_t triplet_trace_build(int32_t trace1, int32_t trace2)
{
  return (trace1 | (trace2 << 16));
}

//---------------------------------------
// Timing dependence inline functions
//---------------------------------------
static inline post_trace_t timing_get_initial_post_trace() {
  return (post_trace_t) 0;
}

//---------------------------------------
static inline post_trace_t timing_add_post_spike(
        uint32_t time, uint32_t last_time, post_trace_t last_trace) {

    // Get time since last spike
    const uint32_t delta_time = time - last_time;

    // Decay previous o1 trace and add energy caused by new spike
    const int32_t decayed_o1 = __smulbb(last_trace,
            DECAY_LOOKUP_TAU_MINUS(delta_time)) >> STDP_FIXED_POINT;
    const int32_t new_o1 = decayed_o1 + STDP_FIXED_POINT_ONE;

    // If this is the 1st post-synaptic event, o2 trace is zero
    // (as it's sampled BEFORE the spike),
    int32_t new_o2;
    if(last_time == 0)
    {
      new_o2 = 0;
    }
    // Otherwise
    else
    {
      // Decay last value
      const int32_t decay = DECAY_LOOKUP_TAU_Y(delta_time);

      // new_o2 = (last_trace.o2 + 1) * decay
      // => new_o2 = (last_trace.o2 * decay) + decay
      new_o2 = __smulbt(decay, last_trace) >> STDP_FIXED_POINT;
      new_o2 += decay;
    }

    log_debug("\tdelta_time=%d, o1=%d, o2=%d\n", delta_time, new_o1, new_o2);

    // Return new pre- synaptic event with decayed trace values with energy
    // for new spike added
    return (post_trace_t)triplet_trace_build(new_o1, new_o2);
}

//---------------------------------------
static inline pre_trace_t timing_add_pre_spike(
        uint32_t time, uint32_t last_time, pre_trace_t last_trace, bool flush) {

    // Get time since last spike
    const uint32_t delta_time = time - last_time;

    // Decay previous r1 trace and add energy caused by new spike
    int32_t new_r1 = __smulbb(last_trace,
            DECAY_LOOKUP_TAU_PLUS(delta_time)) >> STDP_FIXED_POINT;
    if(!flush)
    {
      new_r1 += STDP_FIXED_POINT_ONE;
    }

    // If this is the 1st pre-synaptic event, r2 trace is zero
    // (as it's sampled BEFORE the spike),
    int32_t new_r2;
    if(last_time == 0)
    {
      new_r2 = 0;
    }
    // Otherwise
    else
    {
      // Decay last value
      const int32_t decay = DECAY_LOOKUP_TAU_X(delta_time);

      // new_r2 = (last_trace.r2 + 1) * decay
      // => new_r2 = (last_trace.r2 * decay) + decay
      new_r2 = __smulbt(decay, last_trace) >> STDP_FIXED_POINT;
      new_r2 += decay;
    }

    log_debug("\tdelta_time=%u, r1=%d, r2=%d\n", delta_time, new_r1, new_r2);

    // Return new pre-synaptic event with decayed trace values with energy
    // for new spike added
    return (pre_trace_t) triplet_trace_build(new_r1, new_r2);
}

//---------------------------------------
static inline update_state_t timing_apply_pre_spike(
        uint32_t time, pre_trace_t trace, uint32_t last_pre_time,
        pre_trace_t last_pre_trace, uint32_t last_post_time,
        post_trace_t last_post_trace, update_state_t previous_state) {
    use(last_pre_time);
    use(&last_pre_trace);

    // Get time of event relative to last post-synaptic event
    const uint32_t time_since_last_post = time - last_post_time;
    if (time_since_last_post > 0) {
        const int32_t decayed_o1 = __smulbb(last_post_trace,
            DECAY_LOOKUP_TAU_MINUS(time_since_last_post)) >> STDP_FIXED_POINT;

        // Calculate triplet term
        const int32_t decayed_o1_r2 = __smulbt(decayed_o1, trace) >> STDP_FIXED_POINT;

        log_debug("\t\t\ttime_since_last_post_event=%u, decayed_o1=%d, r2=%d,"
                  "decayed_o1_r2=%d\n", time_since_last_post, decayed_o1,
                                        triplet_trace_get_trace2(trace),
                                        decayed_o1_r2);

        // Apply depression to state (which is a weight_state)
        return weight_two_term_apply_depression(previous_state, decayed_o1,
                                                decayed_o1_r2);
    } else {
        return previous_state;
    }
}

//---------------------------------------
static inline update_state_t timing_apply_post_spike(
        uint32_t time, post_trace_t trace, uint32_t last_pre_time,
        pre_trace_t last_pre_trace, uint32_t last_post_time,
        post_trace_t last_post_trace, update_state_t previous_state) {
    use(last_post_time);
    use(&last_post_trace);

    // Get time of event relative to last pre-synaptic event
    const uint32_t time_since_last_pre = time - last_pre_time;
    if (time_since_last_pre > 0) {
        const int32_t decayed_r1 = __smulbb(last_pre_trace,
            DECAY_LOOKUP_TAU_PLUS(time_since_last_pre)) >> STDP_FIXED_POINT;

        // Calculate triplet term
        const int32_t decayed_r1_o2 = __smulbt(decayed_r1, trace) >> STDP_FIXED_POINT;

        log_debug("\t\t\ttime_since_last_pre_event=%u, decayed_r1=%d, o2=%d,"
                  "decayed_r1_o2=%d\n", time_since_last_pre, decayed_r1,
                                        triplet_trace_get_trace2(trace),
                                        decayed_r1_o2);

        // Apply potentiation to state (which is a weight_state)
        return weight_two_term_apply_potentiation(previous_state, decayed_r1,
                                                  decayed_r1_o2);
    } else {
        return previous_state;
    }
}

#endif	// PFISTER_TRIPLET_IMPL_H
