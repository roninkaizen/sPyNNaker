import logging
import numpy as np
import random
import spynnaker.pyNN as sim
from spynnaker.pyNN.utilities import profiling
from six import iteritems

n_stim = 1000
stim_rate = 10.0
n_neurons = 256
row_length = 10
duration = 5000
#current = 1.49 # 60Hz
#current = 0.83 # 20Hz
#current = 0.7575 # 10Hz
#current = 0.7501 # 5Hz
current = 0.0 # 0Hz
static = False

record_spikes = True
profile = True
# FixedNumberPost not implemented - Mega yawn
def make_connection_list(num_pre, num_post, fixed_number_post, weight, delay):
    connections = []
    for i in range(num_pre):

        connections.extend((i, j, weight, delay)
                           for j in random.sample(range(0, num_post), fixed_number_post))
    return connections

n_synapses = n_stim * row_length
print("Number of synapses:%u" % n_synapses)
print("Synaptic event rate:%fHz" % (n_synapses * stim_rate))

sim.setup(timestep=1.0)

pop_stim = sim.Population(n_stim, sim.SpikeSourcePoisson, {"rate":stim_rate, "duration":duration},
                          label="stim")
pop_neurons = sim.Population(n_neurons, sim.IF_curr_exp, {"tau_refrac":2.0, "i_offset":current},
                             label="pop")

if profile:
    pop_neurons.profile(1E5)

if record_spikes:
    pop_neurons.record()

if static:
    synapse_dynamics = None
else:
    synapse_dynamics = sim.SynapseDynamics(slow=sim.STDPMechanism(
        #timing_dependence=sim.Vogels2011Rule(eta=0.005, rho=0.2),
        timing_dependence=sim.SpikePairRule(tau_plus=16.7, tau_minus=33.7),
        weight_dependence=sim.AdditiveWeightDependence(w_min=0.0, w_max=0.0, A_plus=0.0, A_minus=0.0),
        dendritic_delay_fraction=1.0))

sim.Projection(pop_stim, pop_neurons,
               sim.FromListConnector(make_connection_list(n_stim, n_neurons, row_length, 0.0, 1.0)),
    synapse_dynamics=synapse_dynamics, target="excitatory", label="proj")

sim.run(duration)

if record_spikes:
    pop_neurons_data = pop_neurons.getSpikes(compatible_output=True)
    num_spikes = np.bincount(pop_neurons_data[:,0].astype(int))
    if len(num_spikes) == 0:
        mean_out_rate = 0.0
    else:
        mean_out_rate = np.mean(num_spikes) / (duration / 1000.0)
    print("Out rate:%fHz (post spikes per pre:%f)" % (mean_out_rate, mean_out_rate/stim_rate))

if profile:
    profiling_data = pop_neurons.get_profiling_data()
    profiling.print_summary(profiling_data[0][1], duration)


# End simulation
sim.end()

