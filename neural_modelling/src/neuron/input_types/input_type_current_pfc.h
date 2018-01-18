#ifndef _INPUT_TYPE_CURRENT_PFC_H_
#define _INPUT_TYPE_CURRENT_PFC_H_

#ifndef NUM_EXCITATORY_RECEPTORS
#define NUM_EXCITATORY_RECEPTORS 1
#error NUM_EXCITATORY_RECEPTORS was undefined.  It should be defined by a synapse\
       shaping include
#endif

#ifndef NUM_INHIBITORY_RECEPTORS
#define NUM_INHIBITORY_RECEPTORS 1
#error NUM_INHIBITORY_RECEPTORS was undefined.  It should be defined by a synapse\
       shaping include
#endif

#ifndef NUM_NEUROMODULATORS
#define NUM_NEUROMODULATORS 0
#error NUM_NEUROMODULATORS was undefined.  It should be defined by a synapse\
       shaping include
#endif

#include "input_type.h"

typedef struct input_type_t {
} input_type_t;

uint16_t excitatory_shifts[NUM_EXCITATORY_RECEPTORS] = {3, 3, 3, 3};
uint16_t inhibitory_shifts[NUM_INHIBITORY_RECEPTORS] = {3, 3, 3, 3};

// Todo - write the above data structure from python
//excitatory_shifts[] = {0, 0, 0};
//inhibitory_shifts[] = {0, 0, 0};

static inline s1615 _evaluate_v_effect(state_t v){
	s1615 v_dep = 0;
	v = v >> 7;
	if (v > -0.625k){
		if (v <= 0.3125k) {
			v_dep = 0.783385k +  v * (1.42433k + v * (-3.00206k
					+ v * (-3.70779k + v * (12.1412k + 15.3091k * v))));
		} else {
			v_dep = 1.0k;
		}
	} else {
		v_dep = 0.0k;
	}

	// log_info("v before: %k, v_dep: %k", v, v_dep);
	return v_dep;
}

// makes sense to remove this function, as any type-dependent scaling could be applied in the two functions below.
// Further complicated by fact that you have to use this function of both excitatory and inhibitory synapses
static inline input_t input_type_get_input_value(
        input_t value, input_type_pointer_t input_type) {
    use(input_type);
    return value;
}

static inline input_t* input_type_convert_excitatory_input_to_current(
        input_t* exc_input, input_type_pointer_t input_type,
        state_t membrane_voltage) {
    use(input_type);

    for(int i=0; i < NUM_EXCITATORY_RECEPTORS; i++){

    	exc_input[i] = exc_input[i] >> //input_type->
    			excitatory_shifts[i];

    	if (i==3){ // NMDA synapse
    	    exc_input[i] = exc_input[i] *
    	    		_evaluate_v_effect(membrane_voltage);
//    	    log_info("Exc: V: %k, V_dep: %k", membrane_voltage, v_dep);
    	}
    }

    return &exc_input[0];
}

static inline input_t* input_type_convert_inhibitory_input_to_current(
        input_t* inh_input, input_type_pointer_t input_type,
        state_t membrane_voltage) {
    use(input_type);
    use(membrane_voltage);

    // Currently no voltage-dependent inhibitory synapses so return scaled input
    for(int i=0; i < NUM_INHIBITORY_RECEPTORS; i++){
    	    inh_input[i] = inh_input[i] >> //input_type->
    	    		inhibitory_shifts[i];
    }

    return &inh_input[0];
}

#endif // _INPUT_TYPE_CURRENT_PFC_H_
