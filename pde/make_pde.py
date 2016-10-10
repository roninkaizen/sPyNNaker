import struct

# Write the delay spec - just the number of post-vertices for now
delay_spec = open("delay_spec.dat", "wb")
delay_spec.write(struct.pack("<I", 1))
delay_spec.close()

# Write a simple neuron spec
neuron_spec = open("neuron_spec.dat", "wb")

# uint32_t dse_id = pde_utils_read_next_word(pde_spec);
# uint32_t app_id = pde_utils_read_next_word(pde_spec);
# uint32_t spec_version = pde_utils_read_next_word(pde_spec);
# uint32_t memsize[11];
# for (uint32_t i = 0; i < 11; i++) {
#     memsize[i] = pde_utils_read_next_word(pde_spec);
# }
# n_subedges = pde_utils_read_next_word(pde_spec);
neuron_spec.write(struct.pack("<I", 1234))
neuron_spec.write(struct.pack("<I", 5678))
neuron_spec.write(struct.pack("<I", 0x00010001))
for i in range(0, 11):
    neuron_spec.write(struct.pack("<I", i))
neuron_spec.write(struct.pack("<I", 1))

# uint32_t edge_key = pde_utils_read_next_word(pde_spec);
# uint32_t edge_mask = pde_utils_read_next_word(pde_spec);
# uint32_t edge_delay_key = pde_utils_read_next_word(pde_spec);
# uint32_t edge_delay_mask = pde_utils_read_next_word(pde_spec);
# uint8_t n_delay_stages = pde_utils_read_next_byte(pde_spec);
# delay_placements[edge] = pde_utils_read_next_word(pde_spec);
neuron_spec.write(struct.pack("<I", 1))
neuron_spec.write(struct.pack("<I", 0xFFFFFFF0))
neuron_spec.write(struct.pack("<I", 2))
neuron_spec.write(struct.pack("<I", 0xFFFFFF00))
neuron_spec.write(struct.pack("<B", 3))
neuron_spec.write(struct.pack("<I", 0x00010001))

neuron_spec.close()