#include <spin1_api.h>
#include <stdint.h>

typedef struct pde_spec      *pde_spec;
typedef struct mem_write_ptr *mem_write_ptr;

// Get the PDE to be read for this core
pde_spec pde_utils_get_pde();

// Read the next byte from the PDE
uint8_t pde_utils_read_next_byte(pde_spec spec);

// Read the next half-word from the PDE
uint16_t pde_utils_read_next_half_word(pde_spec spec);

// Read the next word from the PDE
uint32_t pde_utils_read_next_word(pde_spec spec);

// Create a new memory write pointer structure
mem_write_ptr pde_utils_create_write_ptr(uint32_t start_addr);

// assign new value to write pointer
void pde_utils_set_write_ptr(mem_write_ptr ptr, uint32_t new_addr);

// Get current value of the write pointer
uint32_t pde_utils_get_write_ptr(mem_write_ptr ptr);

// Write a byte to the current write pointer position
void pde_utils_write_byte(mem_write_ptr ptr, uint8_t data);

// Write a half-word to the current write pointer position
void pde_utils_write_half_word(mem_write_ptr ptr, uint16_t data);

// Write a word to the current write pointer position
void pde_utils_write_word(mem_write_ptr ptr, uint32_t data);

