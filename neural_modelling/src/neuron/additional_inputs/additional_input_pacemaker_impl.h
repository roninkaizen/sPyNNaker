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
    REAL    I_H;
	REAL 	m;
    REAL    m_inf;
    REAL    tau_m_inf;
    REAL	g_H; // max pacemaker conductance
    REAL 	E_H; // reversal potential
    REAL	dt;

} additional_input_t;

static input_t additional_input_get_input_value_as_current(
        additional_input_pointer_t additional_input,
        state_t membrane_voltage) {
//
//	// Update m_inf
//	additional_input->m_inf = 1 / (1 + expk((membrane_voltage - -75)/5.5));
//		// this could be approximated with polynomial
//
//	// Update tau_m_inf
//	additional_input->tau_m_inf = 1 / (
//			expk(-14.59 - (0.086 * membrane_voltage)) +
//			expk(-1.87 + (0.0701*membrane_voltage))
//			); // this could be approximated with polynomial

	// Update m
	additional_input->m = additional_input->m_inf +
			(additional_input->m - additional_input->m_inf) *
			expk(-additional_input->dt/additional_input->tau_m_inf);
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
