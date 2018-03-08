#ifndef _ADDITIONAL_INPUT_PACEMAKER_H_
#define _ADDITIONAL_INPUT_PACEMAKER_H_

#include "additional_input.h"

//----------------------------------------------------------------------------
// Model from Liu, Y. H., & Wang, X. J. (2001). Spike-frequency adaptation of
// a generalized leaky integrate-and-fire model neuron. Journal of
// Computational Neuroscience, 10(1), 25-45. doi:10.1023/A:1008916026143
//----------------------------------------------------------------------------

typedef struct additional_input_t {

    // Pacemaker Current
    accum    I_H;
    accum    m;
    accum    m_inf;
    accum    e_to_t_on_tau_m_approx;
    accum    g_H; // max pacemaker conductance
    accum    E_H; // reversal potential
    accum    dt;

} additional_input_t;

static input_t additional_input_get_input_value_as_current(
        additional_input_pointer_t additional_input,
        state_t membrane_voltage) {

	// Update m_inf (Substitute polynomial approximation)
	additional_input->m_inf = 0.783385k +  membrane_voltage * (1.42433k + membrane_voltage * (-3.00206k
			+ membrane_voltage * (-3.70779k + membrane_voltage * (12.1412k + 15.3091k * membrane_voltage))));

    // Update exp(t/tau_m) (Substitute polynomial approximation)
	additional_input->e_to_t_on_tau_m_approx = 0.783385k +  membrane_voltage * (1.42433k + membrane_voltage * (-3.00206k
			+ membrane_voltage * (-3.70779k + membrane_voltage * (12.1412k + 15.3091k * membrane_voltage))));

	// Update m
	additional_input->m = additional_input->m_inf +
			(additional_input->m - additional_input->m_inf) *
			additional_input->e_to_t_on_tau_m_approx;
			// this last exponential is hard to avoid

	// H is 1 and constant, so ignore
	additional_input->I_H = additional_input->g_H *
			additional_input->m *
			(membrane_voltage - additional_input->E_H);


    return additional_input->I_H;
}

static void additional_input_has_spiked(
        additional_input_pointer_t additional_input) {
	// no action to be taken on spiking
}

#endif // _ADDITIONAL_INPUT_PACEMAKER_H_
