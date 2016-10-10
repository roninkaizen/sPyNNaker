#include <spin1_api.h>
#include <stdint.h>
#include <stdbool.h>
#include <debug.h>
#include "pde_utils.h"

struct pde_spec {

    // The current position in the data
    uint16_t *data;

    // true if the next byte is aligned, false otherwise
    bool is_aligned;
};


// Structure for write pointer and functions to write words, half-words and bytes:
struct mem_write_ptr {

    // The current position in the data
    uint16_t * address;

    // true if the next byte is aligned, false otherwise
    bool is_aligned;
};

static inline void _read_print(char *type, uint32_t value) {
    //log_debug("Read %s %u (0x%08x)", type, value, value);
}

pde_spec pde_utils_get_pde() {

    pde_spec pde_spec_obj = (pde_spec) spin1_malloc(sizeof(struct pde_spec));
    if (pde_spec_obj == NULL) {
        log_error("Not enough space for PDE struct");
        rt_error(RTE_SWERR);
    }
    pde_spec_obj->is_aligned = true;

    // Get pointer to 1st virtual processor info struct in SRAM
    vcpu_t *sark_virtual_processor_info = (vcpu_t*) SV_VCPU;

    // The address of the data is in user0
    uint16_t *data = (uint16_t *)
        sark_virtual_processor_info[spin1_get_core_id()].user0;

    // The number of bytes in the data is in user1
    uint32_t n_bytes =
        sark_virtual_processor_info[spin1_get_core_id()].user1;

    // Allocate memory to copy the data into
    log_debug(
        "Allocating %u bytes for spec which starts at 0x%08x", n_bytes, data);
    pde_spec_obj->data = (uint16_t *) spin1_malloc(n_bytes);
    if (pde_spec_obj->data == NULL) {
        log_error("Not enough space for PDE of %u bytes", n_bytes);
        rt_error(RTE_SWERR);
    }

    // Copy the data into place
    log_debug("Copying %u bytes of spec", n_bytes);
    spin1_memcpy(pde_spec_obj->data, data, n_bytes);

    // Free the SDRAM ready for allocation
    sark_free(data);

    return pde_spec_obj;
}

uint8_t pde_utils_read_next_byte(pde_spec spec) {

    if (spec->is_aligned) {

        // If the read is aligned, read the lower byte
        spec->is_aligned = false;
        _read_print("aligned byte", spec->data[0] & 0xFF);
        return spec->data[0] & 0xFF;
    } else {

        // If the read is not aligned, read the upper byte
        uint8_t value = (spec->data[0] >> 8) & 0xFF;
        spec->data = &(spec->data[1]);
        spec->is_aligned = true;
        _read_print("unaligned byte", value);
        return value;
    }
}

uint16_t pde_utils_read_next_half_word(pde_spec spec) {
    if (spec->is_aligned) {

        // If the half-word is aligned, read it
        uint16_t value = spec->data[0];
        spec->data = &(spec->data[1]);
        _read_print("aligned half word", value);
        return value;
    } else {

        // If the half-word is not aligned, mask it in
        uint16_t value =
            ((spec->data[0] & 0xFF00) >> 8) |
            ((spec->data[1] & 0xFF) << 8);
        spec->data = &(spec->data[1]);
        _read_print("unaligned half word", value);
        return value;
    }
}

uint32_t pde_utils_read_next_word(pde_spec spec) {
    if (spec->is_aligned) {

        // If the word is aligned, read it
        uint32_t value = (spec->data[1] << 16) | spec->data[0];
        spec->data = &(spec->data[2]);
        _read_print("aligned word", value);
        return value;
    } else {

        // If the word is not aligned, mask it in
        uint32_t value =
            ((spec->data[0] >> 8) & 0xFF) |
            (spec->data[1] << 8) |
            ((spec->data[2] & 0xFF) << 24);
        spec->data = &(spec->data[2]);
        _read_print("unaligned word", value);
        return value;
    }
}

static inline void _write_print(char *type, uint16_t * addr, uint32_t value) {
    //log_debug("Write %s %u  @ addr 0x%x", type, value, addr);
}

mem_write_ptr pde_utils_create_write_ptr(uint32_t start_addr) {
    mem_write_ptr ptr = (mem_write_ptr) spin1_malloc(
        sizeof(struct mem_write_ptr));

    ptr->address = (uint16_t *) start_addr;
    ptr->is_aligned =  true;
    return ptr;
}

// Assign new value to write pointer:
void pde_utils_set_write_ptr(mem_write_ptr ptr, uint32_t new_addr) {
    ptr->address = (uint16_t *) new_addr;
}

// Get current value of the write pointer
uint32_t pde_utilsget_write_ptr(mem_write_ptr ptr) {
    return (uint32_t) ptr->address;
}

// Write a byte to the current write pointer position:
void pde_utils_write_byte(mem_write_ptr ptr, uint8_t data) {
    if (ptr->is_aligned) {

        uint16_t orig_data       = (uint16_t) *(ptr->address);
        uint16_t bottom_byte     = (uint16_t)data & 0xFF;
        uint16_t new_halfword    = (orig_data & 0xFF00) | (bottom_byte);

        _write_print("aligned byte", ptr->address, data);

        *(ptr->address) = (uint16_t) new_halfword;

        // Now unaligned:
        ptr->is_aligned = false;

    } else {
        // If the word is not aligned, qrit to upper half of current halfword:
        uint16_t orig_data         = (uint16_t) *(ptr->address);
        uint16_t top_byte          = (uint16_t) data & 0xFF;
        uint16_t new_halfword      = (orig_data & 0xFF) | (top_byte << 8);

        _write_print("unaligned byte", ptr->address, data);

        *(ptr->address++) = (uint16_t) new_halfword;

        // Now unaligned:
        ptr->is_aligned = true;
    }

}

// Write a half-word to the current write pointer position:
void pde_utils_write_half_word(mem_write_ptr ptr, uint16_t data)
{
    if (ptr->is_aligned) {

        _write_print("aligned halfword", ptr->address, data);

        // Can do simple write in one 16-bit chunk:
        *(ptr->address++) = (uint16_t) (data);
    } else {
        // If the word is not aligned, mask it in using 2 chunks:
        uint16_t orig_data           = (uint16_t) *(ptr->address);
        uint16_t bottom_byte         = (data & 0x00FF);
        uint16_t top_byte            = (data & 0xFF00);
        uint16_t new_bottom_halfword = (orig_data & 0x00FF) | (bottom_byte << 8);
        uint16_t new_top_halfword    = (orig_data & 0xFF00) | (top_byte >> 8);

        _write_print("unaligned halfword", ptr->address, data);

        *(ptr->address++) = (uint16_t) (new_bottom_halfword);
        *(ptr->address)   = (uint16_t) (new_top_halfword);
    }
}

// Write a word to the current write pointer position:
void pde_utils_write_word(mem_write_ptr ptr, uint32_t data)
{
    if (ptr->is_aligned) {

        _write_print("aligned word", ptr->address, data);

        // Can do simple write in 2 16-bit chunks:
        *(ptr->address++) = (uint16_t) (data &0xFFFF);
        *(ptr->address++) = (uint16_t) ((data &0xFFFF0000) >> 16);
    } else {
        // If the word is not aligned, mask it in using 3 chunks:
        uint16_t orig_data           = (uint16_t) *(ptr->address);
        uint32_t bottom_byte         = (data & 0x000000FF);
        uint32_t middle_half_word    = (data & 0x00FFFF00);
        uint32_t top_byte            = (data & 0xFF000000);
        uint16_t new_bottom_halfword = (uint16_t)((orig_data & 0x00FF) | (bottom_byte << 8));
        uint16_t new_middle_halfword = middle_half_word >> 8;
        uint16_t new_top_halfword    = (uint16_t)((orig_data & 0xFF00) | (top_byte >> 24));

        _write_print("unaligned word", ptr->address, data);

        *(ptr->address++) = (uint16_t) (new_bottom_halfword);
        *(ptr->address++) = (uint16_t) (new_middle_halfword);
        *(ptr->address)   = (uint16_t) (new_top_halfword);
    }
}


