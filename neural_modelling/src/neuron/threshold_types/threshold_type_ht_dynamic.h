#ifndef _THRESHOLD_TYPE_DYNAMIC_H_
#define _THRESHOLD_TYPE_DYNAMIC_H_

#include "threshold_type.h"

typedef struct threshold_type_t {

    // The instantaneous value of the static threshold
    REAL threshold_value; // -50

    // Resting threshold to decay back to
    REAL threshold_resting; // -50

    // Decay factor
    REAL threshold_decay; // exp(0.1/2)

    // Na reversal potential to set threshold_value to on spike
    REAL threhsold_Na_reversal; //(30)
    //
} threshold_type_t;

static inline bool threshold_type_is_above_threshold(state_t value,
                        threshold_type_pointer_t threshold_type) {

	// If neuron has spiked
	if REAL_COMPARE((value - threshold_type->threshold_resting),
			>=, (threshold_type->threshold_value
			)) {

		// Set threshold level to sodium reversal potential + threshold, so
		// we can decay threshold value back to zero
		threshold_type->threshold_value = threshold_type->threhsold_Na_reversal -
				threshold_type->threshold_resting;

		// Return true as neuron spiked
		return true;

	} else {
		// Decay threshold value back towards resting threshold
		threshold_type->threshold_value *=
				threshold_type->threshold_decay;

		// Return false 'has not spiked' to neuron.c
		return false;
	}

}

#endif // _THRESHOLD_TYPE_HT_DYNAMIC_H_
