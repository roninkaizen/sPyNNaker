#include <stdbool.h>
#include <debug.h>
#include "pde_utils.h"
#include "delay.h"

// The number of post vertices
static uint32_t n_post_vertices;

// The number of post vertices that have completed
static uint32_t n_post_vertices_finished = 0;

// The list of post vertices that have completed to check for duplicates
static uint32_t *post_vertices_finished;

static void read_pde(pde_spec pde_spec) {
    n_post_vertices = pde_utils_read_next_word(pde_spec);
    post_vertices_finished = spin1_malloc(n_post_vertices * sizeof(uint32_t));
    sark_word_set(
        post_vertices_finished, 0, n_post_vertices * sizeof(uint32_t));
    log_debug("%u post vertices", n_post_vertices);

    // TODO: Read the rest of PDE spec
}

// Sends an acknowledgement response to an SDP
static void send_ack_response(sdp_msg_t *msg) {
    msg->cmd_rc = RC_OK;
    msg->length = 12;
    uint dest_port = msg->dest_port;
    uint dest_addr = msg->dest_addr;
    msg->dest_port = msg->srce_port;
    msg->srce_port = dest_port;
    msg->dest_addr = msg->srce_addr;
    msg->srce_addr = dest_addr;
    spin1_send_sdp_msg(msg, 10);
}

// Handle an incoming SDP message
void handle_sdp_message(uint mailbox, uint port) {

    // Read the message
    sdp_msg_t *msg = (sdp_msg_t *) mailbox;
    uint16_t *data = &(msg->cmd_rc);
    uint16_t n_delays = data[0];

    // If the number of delays is 0, this is a finish message
    if (n_delays == 0) {

        // Send a response to say the message was received
        send_ack_response(msg);

        // Free the message as no longer needed
        spin1_msg_free(msg);

        // Check if the source has been seen before
        uint32_t source = (msg->srce_addr << 16) | (msg->srce_port & 0x1F);
        bool seen = false;
        for (uint32_t i = 0; i < n_post_vertices_finished; i++) {
            if (source == post_vertices_finished[i]) {
                seen = true;
                break;
            }
        }

        // If the source hasn't been seen, mark it as finished
        if (!seen) {
            post_vertices_finished[n_post_vertices_finished] = source;
            n_post_vertices_finished += 1;
            log_debug(
                "%u of %u post vertices complete",
                n_post_vertices_finished, n_post_vertices);
            if (n_post_vertices_finished == n_post_vertices) {
                log_info("All post vertices complete: exiting");
                sark_cpu_state(CPU_STATE_EXIT);
            }
        }
        return;
    }

    // Otherwise, continue reading
    log_debug("Reading %u delays", n_delays);

    delay_t *delays = (delay_t *) &(data[1]);
    for (uint32_t i = 0; i < n_delays; i++) {
        log_debug(
            "Delay %u, source neuron id = %u, delay stage = %u",
            i, delays[i].source_neuron_id, delays[i].delay_stage);

        // TODO: Process the data
    }

    // Send the acknowledgement
    send_ack_response(msg);

    // Free the message
    spin1_msg_free(msg);
}

void c_main() {

    // Get the PDE in DTCM
    pde_spec pde_spec = pde_utils_get_pde();

    // Process the PDE
    read_pde(pde_spec);

    // Wait for SDP messages
    spin1_callback_on(SDP_PACKET_RX, handle_sdp_message, 1);
    spin1_start(SYNC_WAIT);
}
