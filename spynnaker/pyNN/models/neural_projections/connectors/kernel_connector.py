import numpy
from pyNN.random import RandomDistribution
from .abstract_connector import AbstractConnector

from . import ConvolutionKernel

from spynnaker.pyNN.utilities import utility_calls


HEIGHT, WIDTH = 0, 1
ROW, COL = 0, 1

class KernelConnector(AbstractConnector):
    """
    Where the pre- and postsynaptic populations are thought-of as a 2D array.
    Connect every post(row, col) neuron to many pre(row, col, kernel) through 
    the same set of weights and delays.  
    """

    def __init__(self, shape_pre, shape_post, shape_kernel,
                 shape_common=None, pre_sample_steps=None, pre_start_coords=None,
                 post_sample_steps=None, post_start_coords=None,
                 weights=0.0, delays=1,
                 safe=True, space=None, verbose=False, generate_on_machine=False):
        """
        :param shape_common: 
            2D shape of common coordinate system (for both pre and post, usually the 
            input image sizes)
        :param shape_pre: 
            2D shape of the pre population (rows/height, cols/width, usually the input 
            image shape)
        :param shape_post: 
            2D shape of the post population (rows/height, cols/width)
        :param shape_kernel:
            2D shape of the kernel (rows/height, cols/width)
        :param pre/post_sample_steps:
            Sampling steps/jumps for post pop <=> (startX, endX, _stepX_)
            None or 2-item array
        :param pre/post_start_coords:
            Starting row/col for sampling <=> (_startX_, endX, stepX)
            None or 2-item array

        :param weights/kernel:
            weights for connection, repeated througout
        :param delays:
            If `None`, all synaptic delays will be set
            to the global minimum delay. 
            If list/ndarray, shape must be equal to kernel's
        """


        self._kernel_w = shape_kernel[WIDTH]
        self._kernel_h = shape_kernel[HEIGHT]
        self._hlf_k_w  = shape_kernel[WIDTH]//2
        self._hlf_k_h  = shape_kernel[HEIGHT]//2

        if pre_start_coords is None:
            self._pre_start_w = 0
            self._pre_start_h = 0
        else:
            self._pre_start_w = pre_start_coords[WIDTH]
            self._pre_start_h = pre_start_coords[HEIGHT]

        if post_start_coords is None:
            self._post_start_w = 0
            self._post_start_h = 0
        else:
            self._post_start_w = post_start_coords[WIDTH]
            self._post_start_h = post_start_coords[HEIGHT]

        if pre_sample_steps is None:
            self._pre_step_w = 1
            self._pre_step_h = 1
        else:
            self._pre_step_w = pre_sample_steps[WIDTH]
            self._pre_step_h = pre_sample_steps[HEIGHT]

        if post_sample_steps is None:
            self._post_step_w = 1
            self._post_step_h = 1
        else:
            self._post_step_w = post_sample_steps[WIDTH]
            self._post_step_h = post_sample_steps[HEIGHT]

        self._delays = delays.view(ConvolutionKernel) \
                                    if isinstance(delays, numpy.ndarray) else delays
        self._weights = weights.view(ConvolutionKernel) \
                                    if isinstance(weights, numpy.ndarray) else weights

        self._krn_weights = self.get_kernel_vals(self._weights)
        self._krn_delays  = self.get_kernel_vals(self._delays)

        self._shape_common = shape_pre if shape_common is None else shape_common
        self._shape_pre  = shape_pre
        self._shape_post = shape_post 

        self._pre_in_range = {}
        self._all_post = {}
        self._all_pre_in_range = {}
        self._all_pre_in_range_delays = {}
        self._all_pre_in_range_weights = {}
        self._post_as_pre = {}
        self._num_conns = {}
        # self._gen_on_spinn = generate_on_machine

        if numpy.max(weights) > 0 and numpy.min(weights) < 0:
            signed_weights = True
        else:
            signed_weights = False
        # print("\n\nIn KERNEL CONN gen on spinn = %s\n"%generate_on_machine)
        AbstractConnector.__init__(self, safe=safe, verbose=verbose, 
                                   signed_weights=signed_weights,
                                   generate_on_machine=generate_on_machine)

    def slice_to_coords(self, slice, is_pre):
        return numpy.array([self.idx_to_coords(slice.lo_atom, is_pre), \
                            self.idx_to_coords(slice.hi_atom, is_pre)])

    def idx_to_coords(self, idx, is_pre):
        if is_pre:
            _w = self._shape_pre[WIDTH]
        else:
            _w = self._shape_post[WIDTH]

        row = idx // _w
        col = idx % _w
        return [row, col]

    def pre_to_comm(self, (pre_r, pre_c)):
        return [self._pre_start_h + pre_r * self._pre_step_h, \
                self._pre_start_w + pre_c * self._pre_step_w]


    def post_to_comm(self, (post_r, post_c)):
        return [self._post_start_h + post_r * self._post_step_h, \
                self._post_start_w + post_c * self._post_step_w]


    def to_common_coords(self, coords, from_pre):
        if from_pre:
            return self.pre_to_comm(coords)
        else:
            return self.post_to_comm(coords)

    def pre_in_range(self, pre_vertex_slice, post_vertex_slice):
        half_kh = self._kernel_h // 2
        half_kw = self._kernel_w // 2
        min_pre_orig, max_pre_orig = self.slice_to_coords(pre_vertex_slice, True)
        min_pre = self.to_common_coords(min_pre_orig, True)
        max_pre = self.to_common_coords(max_pre_orig, True)

        min_post_orig, max_post_orig = self.slice_to_coords(post_vertex_slice, False)
        min_post = self.to_common_coords(min_post_orig, False)
        max_post = self.to_common_coords(max_post_orig, False)

        min_pre_idx = min_pre[ROW] * self._shape_common[WIDTH] + min_pre[COL]
        max_pre_idx = max_pre[ROW] * self._shape_common[WIDTH] + max_pre[COL]

        min_post_idx = int(numpy.ceil(
            (min_post[ROW] - half_kh) * self._shape_common[WIDTH] +
            min_post[COL] - half_kw))
        max_post_idx = int(numpy.ceil(
            (max_post[ROW] + half_kh) * self._shape_common[WIDTH] +
            max_post[COL] + half_kw))

        min_idx = min(min_pre_idx, min_post_idx)
        max_idx = max(max_pre_idx, max_post_idx)

        # return self.comm_to_pre(numpy.arange(min_idx, max_idx + 1))
        return numpy.arange(min_idx, max_idx + 1)

        # if str(pre_vertex_slice) not in self._pre_in_range or \
        #    str(post_vertex_slice) not in self._pre_in_range[str(pre_vertex_slice)]:
        #     self.compute_statistics(pre_vertex_slice, post_vertex_slice)
        #
        # return self._pre_in_range[str(pre_vertex_slice)][str(post_vertex_slice)]

    def to_post_coords(self, post_vertex_slice):
        post = numpy.arange(post_vertex_slice.lo_atom, post_vertex_slice.hi_atom+1)

        return post//self._shape_post[WIDTH], \
               post%self._shape_post[WIDTH]

    def map_to_pre_coords(self, (post_r, post_c)):
        return self._post_start_h + post_r * self._post_step_h, \
               self._post_start_w + post_c * self._post_step_w

    def post_as_pre(self, post_vertex_slice):
        if str(post_vertex_slice) not in self._post_as_pre:
            self._post_as_pre[str(post_vertex_slice)] = self.map_to_pre_coords(
                                             self.to_post_coords(post_vertex_slice))
        return self._post_as_pre[str(post_vertex_slice)]


    def pre_as_post(self, coords):
        r = ((coords[HEIGHT] - self._pre_start_h - 1) // self._pre_step_h) + 1
        c = ((coords[WIDTH] - self._pre_start_w - 1) // self._pre_step_w) + 1
        return (r, c)

    def get_kernel_vals(self, vals):
        krn_size  = self._kernel_h*self._kernel_w
        krn_shape = (self._kernel_h, self._kernel_w)
        if isinstance(self._delays, RandomDistribution):
            return numpy.array(self._delays.next(krn_size)).reshape(krn_shape)
        elif numpy.isscalar(self._delays):
            return self._delays*numpy.ones(krn_shape)
        elif (isinstance(vals, numpy.ndarray) or isinstance(vals, ConvolutionKernel)) \
                and \
             vals.shape[HEIGHT] == self._kernel_h and \
             vals.shape[WIDTH]  == self._kernel_w:
            return vals.view(ConvolutionKernel)
        else:
            raise Exception("Error generating KernelConnector values")

    def init_pre_entries(self, pre_vertex_slice_str):
        if pre_vertex_slice_str not in self._num_conns:
            self._num_conns[pre_vertex_slice_str] = {}

        if pre_vertex_slice_str not in self._pre_in_range:
            self._pre_in_range[pre_vertex_slice_str] = {}

        if pre_vertex_slice_str not in self._all_post:
            self._all_post[pre_vertex_slice_str] = {}

        if pre_vertex_slice_str not in self._all_pre_in_range:
            self._all_pre_in_range[pre_vertex_slice_str] = {}

        if pre_vertex_slice_str not in self._all_pre_in_range_delays:
            self._all_pre_in_range_delays[pre_vertex_slice_str] = {}

        if pre_vertex_slice_str not in self._all_pre_in_range_weights:
            self._all_pre_in_range_weights[pre_vertex_slice_str] = {}

    def compute_statistics(self, pre_vertex_slice, post_vertex_slice):
        # print("In kernel connector, compute_statistics")
        prevs = str(pre_vertex_slice)
        postvs = str(post_vertex_slice)
        self.init_pre_entries(prevs)

        post_as_pre_r, post_as_pre_c = self.post_as_pre(post_vertex_slice)
        coords = {}
        hh, hw = self._hlf_k_h, self._hlf_k_w
        in_range = False
        unique_pre_ids = []
        all_pre_ids = []
        all_post_ids = []
        all_delays = []
        all_weights = []
        count = 0
        post_lo = post_vertex_slice.lo_atom
        pre_lo  = pre_vertex_slice.lo_atom

        for pre_idx in range(pre_vertex_slice.lo_atom, pre_vertex_slice.hi_atom+1):
            pre_r = pre_idx // self._shape_pre[WIDTH]
            pre_c = pre_idx % self._shape_pre[WIDTH]
            coords[pre_idx] = []
            for post_idx in range(post_vertex_slice.lo_atom, post_vertex_slice.hi_atom+1):

                #convert to common coord system
                r, c = post_as_pre_r[post_idx-post_lo], post_as_pre_c[post_idx-post_lo]
                if r < 0 or r >= self._shape_common[HEIGHT] or \
                   c < 0 or c >= self._shape_common[WIDTH]:
                    continue


                r, c = self.pre_as_post((r, c))

                fr_r = max(0, r - hh)
                to_r = min(r + hh + 1, self._shape_pre[HEIGHT])
                fr_c = max(0, c - hw)
                to_c = min(c + hw + 1, self._shape_pre[WIDTH])

                if fr_r <= pre_r and pre_r < to_r and \
                   fr_c <= pre_c and pre_c < to_c:

                    if post_idx in coords[pre_idx]:
                        continue

                    coords[pre_idx].append(post_idx)

                    dr = r - pre_r
                    kr = hh - dr
                    dc = c - pre_c
                    kc = hw - dc

                    w = self._krn_weights[kr, kc]
                    if w == 0.:
                        continue
                    d = self._krn_delays[kr, kc]

                    count += 1

                    all_pre_ids.append(pre_idx)
                    all_post_ids.append(post_idx)
                    all_delays.append(d)
                    all_weights.append(w)


        self._pre_in_range[prevs][postvs] = numpy.array(unique_pre_ids)
        self._num_conns[prevs][postvs] = count
        # print("\n\n%s -> %s = %d conns\n"%(prevs, postvs, count))
        self._all_post[prevs][postvs] = numpy.array(all_post_ids, dtype='uint32')
        self._all_pre_in_range[prevs][postvs] = numpy.array(all_pre_ids, dtype='uint32')
        self._all_pre_in_range_delays[prevs][postvs] = numpy.array(all_delays)
        self._all_pre_in_range_weights[prevs][postvs] = numpy.array(all_weights)

        return self._pre_in_range[prevs][postvs]

    def min_max_coords(self, pre_r, pre_c):
        hh, hw = self._hlf_k_h, self._hlf_k_w
        return numpy.array([pre_r[0]  - hh, pre_c[0]  - hw]), \
               numpy.array([pre_r[-1] + hh, pre_c[-1] + hw])
    
    def to_pre_indices(self, pre_r, pre_c):
        return pre_r*self._shape_pre[WIDTH] + pre_c

    def gen_key(self, pre_vertex_slice, post_vertex_slice):
        return '%s->%s'%(pre_vertex_slice, post_vertex_slice)
    
    def get_num_conns(self, pre_vertex_slice, post_vertex_slice):
        if str(pre_vertex_slice) not in self._num_conns or \
           str(post_vertex_slice) not in self._num_conns[str(pre_vertex_slice)]:
            self.compute_statistics(pre_vertex_slice, post_vertex_slice)
        
        return self._num_conns[str(pre_vertex_slice)][str(post_vertex_slice)]

    def get_all_delays(self, pre_vertex_slice, post_vertex_slice):
        if str(pre_vertex_slice) not in self._all_pre_in_range_delays or \
           str(post_vertex_slice) not in \
                        self._all_pre_in_range_delays[str(pre_vertex_slice)]:
            self.compute_statistics(pre_vertex_slice, post_vertex_slice)

        return self._all_pre_in_range_delays[str(pre_vertex_slice)]\
                                                       [str(post_vertex_slice)]

    def get_delay_maximum(self):
        #way over-estimated
        n_conns = self._n_pre_neurons*self._n_post_neurons*self._kernel_w*self._kernel_h
        return float(self._get_delay_maximum(self._delays, n_conns))



    def get_delay_variance(self, pre_slices, pre_slice_index, post_slices,
                           post_slice_index, pre_vertex_slice, post_vertex_slice):

        slices = (slice(0, self._kernel_h), slice(0, self._kernel_w))
        return float(self._get_delay_variance(self._delays, slices))

    def get_n_connections_from_pre_vertex_maximum(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice,
            min_delay=None, max_delay=None):
        #max outgoing from pre connections with min_delay <= delay <= max_delay
        _pre_in_range = self.pre_in_range(pre_vertex_slice, post_vertex_slice)
        if len(_pre_in_range) == 0:
            return 0

        if isinstance(self._weights, ConvolutionKernel):
            if self._weights.size > pre_vertex_slice.n_atoms:
                # print("KERNEL CONNECTOR n_conns = %d"%pre_vertex_slice.n_atoms)
                return pre_vertex_slice.n_atoms
            elif self._weights.size > 255:
                return 255
            else:
                return int( (self._weights[self._weights != 0]).size )

        return numpy.clip(self._kernel_h * self._kernel_w, 0, 255)

    def get_n_connections_to_post_vertex_maximum(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        _pre_in_range = self.pre_in_range(pre_vertex_slice, post_vertex_slice)
        if len(_pre_in_range) == 0:
            return 0

        if isinstance(self._weights, ConvolutionKernel):
            if self._weights.size > pre_vertex_slice.n_atoms:
                # print("KERNEL CONNECTOR n_conns = %d"%pre_vertex_slice.n_atoms)
                return pre_vertex_slice.n_atoms
            elif self._weights.size > 255:
                return 255
            else:
                return int( (self._weights[self._weights != 0]).size )

        return numpy.clip(self._kernel_h * self._kernel_w, 0, 255)

    def get_weight_mean(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):

        if isinstance(self._weights, ConvolutionKernel):
            return numpy.mean(numpy.abs( self._weights[self._weights != 0] ))

        slices = (slice(0, self._kernel_h), slice(0, self._kernel_w))
        return self._get_weight_mean(self._weights, slices)


    def get_weight_maximum(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):

        if isinstance(self._weights, ConvolutionKernel):
            return numpy.max(numpy.abs( self._weights[self._weights != 0] ))

        slices = (slice(0, self._kernel_h), slice(0, self._kernel_w))
        n_conns = self._n_pre_neurons*self._n_post_neurons*self._kernel_w*self._kernel_h
        return self._get_weight_maximum(self._weights, n_conns, slices)
        
    def get_weight_variance(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):

        if isinstance(self._weights, ConvolutionKernel):
            return numpy.var(numpy.abs( self._weights[self._weights != 0] ))

        slices = (slice(0, self._kernel_h), slice(0, self._kernel_w))
        return self._get_weight_variance(self._weights, slices)

    def generate_on_machine(self):
        return ( self._gen_on_spinn and
                 self._generate_lists_on_machine(self._weights) and
                 self._generate_lists_on_machine(self._delays))

    def __repr__(self):
        return "KernelConnector"

    def create_synaptic_block(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice,
            synapse_type):
        n_connections = self.get_num_conns(pre_vertex_slice, post_vertex_slice)

        syn_dtypes = AbstractConnector.NUMPY_SYNAPSES_DTYPE
        #[('source', 'uint32'), ('target', 'uint16'), ('weight', 'float64'),
        # ('delay', 'float64'), ('synapse_type', 'uint8')]

        if n_connections <= 0:
            return numpy.zeros(0, dtype=syn_dtypes)

        prevs  = str(pre_vertex_slice)
        postvs = str(post_vertex_slice)

        #0 for exc, 1 for inh
        syn_type = numpy.array(self._all_pre_in_range_weights[prevs][postvs] < 0)
        block = numpy.zeros(n_connections, dtype=syn_dtypes)
        # print(numpy.max(self._all_post[prevs][postvs]))
        # print(numpy.max(self._all_pre_in_range[prevs][postvs]))
        # print("max ids")
        block["source"] = self._all_pre_in_range[prevs][postvs]
        block["target"] = self._all_post[prevs][postvs]
        block["weight"] = self._all_pre_in_range_weights[prevs][postvs]
        block["delay"]  = self._all_pre_in_range_delays[prevs][postvs]
        block["synapse_type"] =  syn_type.astype('uint8')
        return block


    def gen_on_machine_info(self):
        def shape2word(sw, sh):
            return ( (numpy.uint32(sw) & 0xFFFF) << 16 ) | \
                     (numpy.uint32(sh) & 0xFFFF)

        block = []

        block.append( shape2word(self._shape_common[WIDTH],
                                 self._shape_common[HEIGHT]) )

        block.append( shape2word(self._shape_pre[WIDTH],
                                 self._shape_pre[HEIGHT]) )

        block.append( shape2word(self._shape_post[WIDTH],
                                 self._shape_post[HEIGHT]) )

        block.append( shape2word(self._pre_start_w,  self._pre_start_h) )

        block.append( shape2word(self._post_start_w, self._post_start_h) )

        block.append( shape2word(self._pre_step_w,   self._pre_step_h) )

        block.append( shape2word(self._post_step_w,  self._post_step_h) )

        block.append( shape2word(self._kernel_w,     self._kernel_h) )

        return block


    def get_max_num_connections(self, pre_slice, post_slice):
        return post_slice.n_atoms * self._kernel_w * self._kernel_h