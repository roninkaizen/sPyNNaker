#ifndef _INPUT_TYPE_CURRENT_H_
#define _INPUT_TYPE_CURRENT_H_

#include "input_type.h"

typedef struct input_type_t {
} input_type_t;

static inline s1615 _evaluate_v_effect(state_t v){
	v = v / 128.0k;
	s1615 v_dep = 0.783385k +  v * (1.42433k + v * (-3.00206k
			+ v * (-3.70779k + v * (12.1412k + 15.3091k * v))));
	log_info("v before: %k, v_dep: %k", v, v_dep);
	return v_dep;
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
    s1615 v_dep = _evaluate_v_effect(membrane_voltage);
    log_info("Exc: V: %k, V_dep: %k", membrane_voltage, v_dep);
    return exc_input * v_dep;
}

static inline input_t input_type_convert_inhibitory_input_to_current(
        input_t inh_input, input_type_pointer_t input_type,
        state_t membrane_voltage) {
    use(input_type);
    s1615 v_dep = _evaluate_v_effect(membrane_voltage);
    log_info("Inh: V: %k, V_dep: %k", membrane_voltage, v_dep);
    return inh_input * v_dep;
}

#endif // _INPUT_TYPE_CURRENT_H_
