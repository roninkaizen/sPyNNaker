#!/usr/bin/env python
import unittest
import spynnaker.pyNN as pynn
from pprint import pprint as pp
#Setup
pynn.setup(timestep=1, min_delay=1, max_delay=10.0)

cell_params_lif = {'cm'  : 0.25,
             'i_offset'  : 0.0,
             'tau_m'     : 20.0,
             'tau_refrac': 2.0,
             'tau_syn_E' : 5.0,
             'tau_syn_I' : 5.0,
             'v_reset'   : -70.0,
             'v_rest'    : -65.0,
             'v_thresh'  : -50.0
             }
spike_array = {'spike_times':[0]}
#/Setup


class TestingFixedProbabilityConnector(unittest.TestCase):
    def test_generate_synapse_list(self):
        print "-----------------50% PROBABILITY--------------------"
        number_of_neurons = 5
        onep=pynn.Population(number_of_neurons,pynn.IF_curr_exp,cell_params_lif,label="One pop")
        twop=pynn.Population(number_of_neurons,pynn.IF_curr_exp,cell_params_lif,label= "Second pop")
        weight = 2
        delay = 1
        synapse_type = onep.vertex.get_synapse_id('excitatory')
        one_to_one_c = pynn.FixedProbabilityConnector(0.5,weight,delay)
        #def generate_synapse_list(self, prevertex, postvertex, delay_scale, synapse_type)
        synaptic_list = one_to_one_c.generate_synapse_list(onep.vertex,onep.vertex,1,synapse_type)
        pp(synaptic_list.get_rows())

    def test_generate_synapse_list_probability_zero(self):
        print "-----------------ZERO PROBABILITY--------------------"
        number_of_neurons = 5
        onep=pynn.Population(number_of_neurons,pynn.IF_curr_exp,cell_params_lif,label="One pop")
        twop=pynn.Population(number_of_neurons,pynn.IF_curr_exp,cell_params_lif,label= "Second pop")
        weight = 2
        delay = 1
        synapse_type = onep.vertex.get_synapse_id('excitatory')
        one_to_one_c = pynn.FixedProbabilityConnector(0,weight,delay)
        #def generate_synapse_list(self, prevertex, postvertex, delay_scale, synapse_type)
        synaptic_list = one_to_one_c.generate_synapse_list(onep.vertex,onep.vertex,1,synapse_type)
        pp(synaptic_list.get_rows())

    def test_generate_synapse_list_probability_100(self):
        print "-----------------100% PROBABILITY--------------------"
        number_of_neurons = 5
        onep=pynn.Population(number_of_neurons,pynn.IF_curr_exp,cell_params_lif,label="One pop")
        twop=pynn.Population(number_of_neurons,pynn.IF_curr_exp,cell_params_lif,label= "Second pop")
        weight = 2
        delay = 1
        synapse_type = onep.vertex.get_synapse_id('excitatory')
        one_to_one_c = pynn.FixedProbabilityConnector(1,weight,delay)
        #def generate_synapse_list(self, prevertex, postvertex, delay_scale, synapse_type)
        synaptic_list = one_to_one_c.generate_synapse_list(onep.vertex,onep.vertex,1,synapse_type)
        pp(synaptic_list.get_rows())
        self.assertEqual(synaptic_list.get_max_weight(),weight)
        self.assertEqual(synaptic_list.get_min_weight(),weight)
        self.assertEqual(synaptic_list.get_n_rows(),number_of_neurons)
        self.assertEqual(synaptic_list.get_min_max_delay(),(delay,delay))

    def test_generate_synapse_list_probability_200(self):
        print "-----------------200% PROBABILITY--------------------"
        with self.assertRaises(Exception):
            number_of_neurons = 5
            onep=pynn.Population(number_of_neurons,pynn.IF_curr_exp,cell_params_lif,label="One pop")
            twop=pynn.Population(number_of_neurons,pynn.IF_curr_exp,cell_params_lif,label= "Second pop")
            weight = 2
            delay = 1
            synapse_type = onep.vertex.get_synapse_id('excitatory')
            one_to_one_c = pynn.FixedProbabilityConnector(2,weight,delay)

    def test_synapse_list_generation_for_null_populations(self):
        print("--------------------ZERO POPULATION--------------------")
        weight = 2
        delay = 1
        zp = pynn.Population(0,pynn.IF_curr_exp,cell_params_lif,label="Zero pop")
        one_to_one_c = pynn.FixedProbabilityConnector(0.1,weight,delay)
        number_of_neurons = 10
        onep=pynn.Population(number_of_neurons,pynn.IF_curr_exp,cell_params_lif,label="One pop")
        zero_synaptic_list = one_to_one_c.generate_synapse_list(zp.vertex,onep.vertex,1,0)
        pp(zero_synaptic_list.get_rows())

    def test_synapse_list_generation_for_negative_sized_populations(self):
        print("-------------NEGATIVE % PROBABILITY---------------")
        with self.assertRaises(Exception):
            number_of_neurons = 5
            onep=pynn.Population(number_of_neurons,pynn.IF_curr_exp,cell_params_lif,label="One pop")
            twop=pynn.Population(number_of_neurons,pynn.IF_curr_exp,cell_params_lif,label= "Second pop")
            weight = 2
            delay = 1
            synapse_type = onep.vertex.get_synapse_id('excitatory')
            one_to_one_c = pynn.FixedProbabilityConnector(-0.5,weight,delay)

    def test_synapse_list_generation_for_different_sized_populations(self):
        print "-----------------10% PROBABILITY--------------------"
        number_of_neurons = 10
        onep=pynn.Population(number_of_neurons,pynn.IF_curr_exp,cell_params_lif,label="One pop")
        twop=pynn.Population(number_of_neurons + 5,pynn.IF_curr_exp,cell_params_lif,label= "Second pop")
        weight = 2
        delay = 1
        one_to_one_c = pynn.FixedProbabilityConnector(0.1,weight,delay)
        synaptic_list = one_to_one_c.generate_synapse_list(onep.vertex,twop.vertex,1,0)
        pp(synaptic_list.get_rows())


    def test_allow_self_connections(self):
        with self.assertRaises(Exception):
            print "--------------SELF CONNECTION-------------------"
            number_of_neurons = 5
            onep=pynn.Population(number_of_neurons,pynn.IF_curr_exp,cell_params_lif,label="One pop")
            twop=pynn.Population(number_of_neurons,pynn.IF_curr_exp,cell_params_lif,label= "Second pop")
            weight = 2
            delay = 1
            synapse_type = onep.vertex.get_synapse_id('excitatory')
            one_to_one_c = pynn.FixedProbabilityConnector(1,weight,delay,allow_self_connections = False)
            synaptic_list = one_to_one_c.generate_synapse_list(onep.vertex,onep.vertex,1,synapse_type)
            pp(synaptic_list.get_rows())

if __name__=="__main__":
    unittest.main()