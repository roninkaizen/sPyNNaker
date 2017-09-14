#ifndef _SYNAPSE_STRUCUTRE_WEIGHT_IMPL_H_
#define _SYNAPSE_STRUCUTRE_WEIGHT_IMPL_H_

//---------------------------------------
// Structures
//---------------------------------------
// Plastic synapse types have weights and eligibility traces
#if SYNAPSE_TYPE_COUNT > SYNAPSE_INPUT_TYPE_COUNT
typedef struct {
    int16_t eligibility_trace;
    weight_t weight;
} plastic_synapse_t;
#else
typedef weight_t plastic_synapse_t;
#endif

// The update state is purely a weight state
typedef weight_state_t update_state_t;

// The final state is just a weight as this is
// Both the weight and the synaptic word
typedef weight_t final_state_t;

//---------------------------------------
// Synapse interface functions
//---------------------------------------
static inline update_state_t synapse_structure_get_update_state(
        plastic_synapse_t synaptic_word, index_t synapse_type) {
#if SYNAPSE_TYPE_COUNT > SYNAPSE_INPUT_TYPE_COUNT
    return weight_get_initial(synaptic_word.weight, synapse_type);
#else
    return weight_get_initial(synaptic_word, synapse_type);
#endif
}

//---------------------------------------
static inline final_state_t synapse_structure_get_final_state(
        update_state_t state) {
    return weight_get_final(state);
}

//---------------------------------------
static inline weight_t synapse_structure_get_final_weight(
        final_state_t final_state) {
    return final_state;
}


//---------------------------------------

static inline plastic_synapse_t synapse_structure_get_final_synaptic_word(
        final_state_t final_state) {
#if SYNAPSE_TYPE_COUNT == SYNAPSE_INPUT_TYPE_COUNT
    return final_state;
#else
    use(final_state);
    return (plastic_synapse_t){0,0};
#endif
}


#endif  // _SYNAPSE_STRUCUTRE_WEIGHT_IMPL_H_
