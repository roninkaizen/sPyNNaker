#include <stdbool.h>
#include <debug.h>
#include "pde_utils.h"
#include "delay.h"

#define SLEEP_TIME 10000

// A message to send to delay extensions with their data
static sdp_msg_t delay_message;

// The number of subedges incoming to this vertex
static uint32_t n_subedges;

// The placement of the delays (an array of n_subedges placements)
// Used to send shutdown messages at the end of execution
static uint32_t *delay_placements;

// Indicates whether a response has been received to the last message
// sent to a delay extension (true initially to bypass waiting)
static bool delay_response_received = true;

void sdp_callback(uint mailbox, uint sdp_port) {
    log_debug("Received delay response");
    delay_response_received = true;
}

static void wait_for_delay_response() {

    // Wait until the response to the last message has been received
    while (!delay_response_received) {

        // Wait for a time for a response
        log_debug("Waiting for response from last delay message");
        spin1_delay_us(SLEEP_TIME);

        // Re-send the message
        if (!delay_response_received) {
            spin1_send_sdp_msg(&delay_message, 1);
        }
    }
}

// Sends delays to the delay core
static void send_delays(uint32_t edge, uint32_t n_delays, delay_t *delays) {

    wait_for_delay_response();
    delay_response_received = false;

    // If there is no delay on this edge, ignore it
    if (delay_placements[edge] == 0) {
        return;
    }

    uint16_t delay_chip = (delay_placements[edge] >> 16) & 0xFFFF;
    uint8_t delay_core = delay_placements[edge] & 0xFF;

    // initialise SDP header
    delay_message.tag = DELAY_SDP_TAG;
    delay_message.flags = 0x07;
    delay_message.dest_addr = delay_chip;
    delay_message.dest_port = (DELAY_SDP_PORT << PORT_SHIFT) | delay_core;
    delay_message.srce_addr = spin1_get_chip_id();
    delay_message.srce_port = (3 << PORT_SHIFT) | spin1_get_core_id();

    // If the number of delays is 0, this is an end message
    if (n_delays == 0) {
        log_debug(
            "Sending end message to 0x%04x, %u", delay_chip, delay_core);
        delay_message.cmd_rc = 0;
        delay_message.length = sizeof(sdp_hdr_t) + sizeof(uint32_t);
        spin1_send_sdp_msg(&delay_message, 1);
        return;
    }

    uint32_t n_delays_to_send = n_delays;
    uint32_t offset = 0;
    while (n_delays_to_send > 0) {

        uint32_t n_delays_in_packet = n_delays_to_send;
        if (n_delays_in_packet > MAX_N_DELAYS_PER_PACKET) {
            n_delays_in_packet = MAX_N_DELAYS_PER_PACKET;
        }

        n_delays_to_send -= n_delays_in_packet;

        delay_message.length =
            sizeof(sdp_hdr_t) + sizeof(uint32_t) +
            (sizeof(delay_t) * n_delays_in_packet);

        uint16_t *data = &delay_message.cmd_rc;
        data[0] = (uint16_t) n_delays_in_packet;
        spin1_memcpy(
            &(data[1]), &(delays[offset]),
            sizeof(delay_t) * n_delays_in_packet);
        spin1_send_sdp_msg(&delay_message, 1);

        log_debug(
            "Sending %u of %u delays to 0x%04x, %u",
            n_delays_in_packet, n_delays, delay_chip, delay_core);
    }

}

// Reads the data for an edge
static void read_edge_data(uint32_t edge, pde_spec pde_spec) {
    uint32_t edge_key = pde_utils_read_next_word(pde_spec);
    uint32_t edge_mask = pde_utils_read_next_word(pde_spec);
    uint32_t edge_delay_key = pde_utils_read_next_word(pde_spec);
    uint32_t edge_delay_mask = pde_utils_read_next_word(pde_spec);
    uint8_t n_delay_stages = pde_utils_read_next_byte(pde_spec);
    delay_placements[edge] = pde_utils_read_next_word(pde_spec);

    // TODO: Read the rest of the edge data

    // TODO: Remove this - this is an example!
    delay_t delay_example[1];
    delay_example[0] = ((delay_t) {.delay_stage = 1, .source_neuron_id = 0});
    send_delays(edge, 1, delay_example);
}

// Read and process the PDE spec
static void read_pde_spec(pde_spec pde_spec) {

    // Read the initial data
    uint32_t dse_id = pde_utils_read_next_word(pde_spec);
    uint32_t app_id = pde_utils_read_next_word(pde_spec);
    uint32_t spec_version = pde_utils_read_next_word(pde_spec);
    uint32_t memsize[11];
    for (uint32_t i = 0; i < 11; i++) {
        memsize[i] = pde_utils_read_next_word(pde_spec);
    }

    // TODO: Read the rest of the data

    // Read the edge data
    n_subedges = pde_utils_read_next_word(pde_spec);
    delay_placements = spin1_malloc(n_subedges * sizeof(uint32_t));
    for (uint i = 0; i < n_subedges; i++) {
        read_edge_data(i, pde_spec);
    }

    // TODO: Read the rest of the data

}

void timer_callback(uint time, uint unused) {
    spin1_exit(0);
}

void c_main() {

    // Synchronise with the delay PDEs
    log_debug("Synchronising with delay extension");
    spin1_callback_on(SDP_PACKET_RX, sdp_callback, -1);
    spin1_callback_on(TIMER_TICK, timer_callback, 1);
    spin1_set_timer_tick(1000);
    spin1_start(SYNC_WAIT);

    // Process the PDE
    pde_spec pde_spec = pde_utils_get_pde();
    read_pde_spec(pde_spec);

    // Send the exit messages to the delay extensions
    for (uint32_t i = 0; i < n_subedges; i++) {
        send_delays(i, 0, NULL);
    }
    wait_for_delay_response();
}
