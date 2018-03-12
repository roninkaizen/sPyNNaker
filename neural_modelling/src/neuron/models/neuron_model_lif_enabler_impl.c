#include "neuron_model_lif_enabler_impl.h"

#include <debug.h>

// simple Leaky I&F ODE
static inline void _lif_neuron_closed_form(
        neuron_pointer_t neuron, REAL V_prev, input_t input_this_timestep) {

    REAL alpha = input_this_timestep * neuron->R_membrane + neuron->V_rest;

    // update membrane voltage
    neuron->V_membrane = alpha - (neuron->exp_TC * (alpha - V_prev));
    if(neuron->V_membrane > neuron->V_max){
        neuron->V_membrane = neuron->V_max;
    }
}

void neuron_model_set_global_neuron_params(
        global_neuron_params_pointer_t params) {
    use(params);

    // Does Nothing - no params
}

state_t neuron_model_state_update(
        input_t exc_input, input_t inh_input, input_t external_bias,
        neuron_pointer_t neuron) {
    input_t input_this_timestep = 0;
    // If outside of the refractory period
    if (neuron->refract_timer <= 0) {

        // Get the input in nA
//        input_t input_this_timestep =
//            exc_input + external_bias + neuron->I_offset;
        input_this_timestep =
            exc_input*inh_input + external_bias + neuron->I_offset;

        _lif_neuron_closed_form(
            neuron, neuron->V_membrane, input_this_timestep);
    } else {

        // countdown refractory timer
        neuron->refract_timer -= 1;
    }


//    if(inh_input > neuron->V_enable){
//        inh_input = neuron->V_enable;
//    }
//    REAL v_out = neuron->V_membrane * inh_input;
    REAL v_out = neuron->V_membrane;

//    log_debug("state update, v = %11.4k, inh = %11.4k > %11.4k",
//                neuron->V_membrane, inh_input, neuron->V_enable);
    log_debug(
    "state update, v = %11.4k, fast = %11.4k, slow = %11.4k, "
    "input = %11.4k, v_out = %11.4k",
    neuron->V_membrane, exc_input, inh_input, input_this_timestep, v_out);

//    if(inh_input > neuron->V_enable){
//        log_debug("Try to spike");
//        return neuron->V_membrane;
//    }else{
//        log_debug("Don't spike!");
//        return neuron->V_rest;
//    }
    return v_out;
}

void neuron_model_has_spiked(neuron_pointer_t neuron) {

    // reset membrane voltage
    neuron->V_membrane = neuron->V_reset;

    // reset refractory timer
    neuron->refract_timer  = neuron->T_refract;
}

state_t neuron_model_get_membrane_voltage(neuron_pointer_t neuron) {
    return neuron->V_membrane;
}

void neuron_model_print_state_variables(restrict neuron_pointer_t neuron) {
    log_debug("V membrane    = %11.4k mv", neuron->V_membrane);
}

void neuron_model_print_parameters(restrict neuron_pointer_t neuron) {
    log_debug("V reset       = %11.4k mv", neuron->V_reset);
    log_debug("V rest        = %11.4k mv", neuron->V_rest);

    log_debug("I offset      = %11.4k nA", neuron->I_offset);
    log_debug("R membrane    = %11.4k Mohm", neuron->R_membrane);

    log_debug("exp(-ms/(RC)) = %11.4k [.]", neuron->exp_TC);

    log_debug("T refract     = %10u timesteps", neuron->T_refract);
    log_debug("V_max         = %11.4k mV", neuron->V_max);
    log_debug("V_enable      = %11.4k mV", neuron->V_enable);
}
