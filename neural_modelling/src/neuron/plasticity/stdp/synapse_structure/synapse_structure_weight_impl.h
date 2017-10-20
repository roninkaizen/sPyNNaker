#ifndef _SYNAPSE_STRUCUTRE_WEIGHT_IMPL_H_
#define _SYNAPSE_STRUCUTRE_WEIGHT_IMPL_H_

//---------------------------------------
// Structures
//---------------------------------------
// Plastic synapse types have weights and eligibility traces
#ifdef _SYNAPSE_TYPES_EXP_SUPERVISION_IMPL_H
typedef uint32_t plastic_synapse_t;
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

static inline int32_t synapse_structure_get_weight(plastic_synapse_t state) {
    return (int16_t)(state >> 16);
}

static inline int32_t synapse_structure_get_eligibility_trace(
                                                    plastic_synapse_t state) {
    return (int16_t)(state & 0xFFFF);
}

static inline plastic_synapse_t synapse_structure_update_state(
                                                    int32_t trace, int32_t weight) {
    return (plastic_synapse_t)( (weight << 16) | ((uint16_t)(trace)) );
}

static inline update_state_t synapse_structure_get_update_state(
        plastic_synapse_t synaptic_word, index_t synapse_type) {

#ifdef _SYNAPSE_TYPES_EXP_SUPERVISION_IMPL_H
    return weight_get_initial(synapse_structure_get_weight(synaptic_word),
                              synapse_type);
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
#ifndef _SYNAPSE_TYPES_EXP_SUPERVISION_IMPL_H
static inline plastic_synapse_t synapse_structure_get_final_synaptic_word(
        final_state_t final_state) {
    return final_state;
}
#endif

#endif  // _SYNAPSE_STRUCUTRE_WEIGHT_IMPL_H_
