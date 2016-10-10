#include "pde_utils.h"

void c_main() {
    pde_spec pde_spec = pde_utils_get_pde();
    for (uint32_t i = 0; i < 4; i++) {
        pde_utils_read_next_word(pde_spec);
        pde_utils_read_next_half_word(pde_spec);
        pde_utils_read_next_byte(pde_spec);
    }
}
