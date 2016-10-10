from spinnman.transceiver import create_transceiver_from_hostname
import sys
import os
from spinn_storage_handlers.file_data_reader import FileDataReader
from spinnman.model.cpu_state import CPUState
from spinnman.messages.scp.scp_signal import SCPSignal

app_id = 30
neuron_x = 0
neuron_y = 0
neuron_p = 1
neuron_aplx = "neuron_pde.aplx"
delay_x = 0
delay_y = 1
delay_p = 1
delay_aplx = "delay_pde.aplx"


def load_spec(transceiver, x, y, p, app_id, spec, aplx):

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

if len(sys.argv) < 4:
    print "python test_pde.py <hostname> <neuron_spec> <delay_spec> [<board_version>]"
    sys.exit(0)

hostname = sys.argv[1]
neuron_spec = sys.argv[2]
delay_spec = sys.argv[3]
version = 5
if len(sys.argv) > 4:
    version = int(sys.argv[4])

transceiver = create_transceiver_from_hostname(hostname, version)
print "Booting machine"
transceiver.ensure_board_is_ready(enable_reinjector=False)
load_spec(transceiver, neuron_x, neuron_y, neuron_p, app_id, neuron_spec, neuron_aplx)
load_spec(transceiver, delay_x, delay_y, delay_p, app_id, delay_spec, delay_aplx)

print "Waiting for synchronisation"
count = 0
while count < 1:
    count = transceiver.get_core_state_count(app_id, CPUState.SYNC0)

print "Starting in sync"
transceiver.send_signal(app_id, SCPSignal.SYNC0)

print "Waiting for application to finish"
count = 0
while count < 1:
    count = transceiver.get_core_state_count(app_id, CPUState.FINISHED)

print transceiver.get_iobuf_from_core(neuron_x, neuron_y, neuron_p)
print transceiver.get_iobuf_from_core(delay_x, delay_y, delay_p)

transceiver.stop_application(app_id)
transceiver.close()
print "Done"
