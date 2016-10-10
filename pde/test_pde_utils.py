from spinnman.transceiver import create_transceiver_from_hostname
import sys
import os
from spinn_storage_handlers.file_data_reader import FileDataReader
from spinnman.model.cpu_state import CPUState
from spinnman.messages.scp.scp_signal import SCPSignal
import struct

app_id = 30


def load_spec(transceiver, x, y, p, app_id, spec, aplx):

    print "Writing Spec"
    spec_file = open(spec, "wb")
    for i in range(4):
        spec_file.write(struct.pack("<I", 1234))
        spec_file.write(struct.pack("<H", 5678))
        spec_file.write(struct.pack("<B", 90))
    spec_file.close()

    print "Allocating SDRAM for", spec
    address_data = transceiver.get_user_0_register_address_from_core(x, y, p)
    length_data = transceiver.get_user_1_register_address_from_core(x, y, p)
    size = os.stat(spec).st_size
    address = transceiver.malloc_sdram(x, y, size, app_id)

    print "Loading spec", spec, "to", hex(address), "recording at", hex(address_data), "and", hex(length_data)
    transceiver.write_memory(x, y, address_data, address)
    transceiver.write_memory(x, y, length_data, size)
    reader = FileDataReader(spec)
    transceiver.write_memory(x, y, address, reader, size)
    reader.close()

    print "Executing binary", aplx
    aplx_size = os.stat(aplx).st_size
    aplx_reader = FileDataReader(aplx)
    transceiver.execute(x, y, [p], aplx_reader, app_id, aplx_size)
    aplx_reader.close()

if len(sys.argv) < 2:
    print "python test_pde_utils.py <hostname> [<board_version>]"
    sys.exit(0)

hostname = sys.argv[1]
version = 5
if len(sys.argv) > 2:
    version = int(sys.argv[2])

transceiver = create_transceiver_from_hostname(hostname, version)
transceiver.ensure_board_is_ready(enable_reinjector=False)

load_spec(transceiver, 0, 0, 5, app_id, "test.dat", "test_pde_utils.aplx")

print "Waiting for application to finish"
count = 0
while count < 2:
    count = transceiver.get_core_state_count(app_id, CPUState.FINISHED)

print transceiver.get_iobuf_from_core(0, 0, 5)
