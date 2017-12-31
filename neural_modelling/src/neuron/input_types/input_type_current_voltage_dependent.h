#ifndef _INPUT_TYPE_CURRENT_H_
#define _INPUT_TYPE_CURRENT_H_

#include "input_type.h"

typedef struct input_type_t {
} input_type_t;

static inline uint_ulr_t _evaluate_v_effect(state_t v){
	v = v >> 7;
	return  (0.783385 + v*(1.11276 +
					v*(-1.83231 + v*(-1.76801 + v*(4.52296 + 4.45553*v)))) ) << 7;
}

static inline input_t input_type_get_input_value(
        input_t value, input_type_pointer_t input_type) {
    use(input_type);
    return value;
}

static inline input_t input_type_convert_excitatory_input_to_current(
        input_t exc_input, input_type_pointer_t input_type,
        state_t membrane_voltage) {
    use(input_type);
    return exc_input * _evaluate_v_effect(membrane_volatge);
}

static inline input_t input_type_convert_inhibitory_input_to_current(
        input_t inh_input, input_type_pointer_t input_type,
        state_t membrane_voltage) {
    use(input_type);
    return inh_input * _evaluate_v_effect(membrane_voltage);
}

#endif // _INPUT_TYPE_CURRENT_H_
