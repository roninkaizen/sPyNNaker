from .abstract_connector import AbstractConnector
from spynnaker.pyNN import exceptions
from spynnaker.pyNN.utilities import utility_calls
import logging
import numpy

logger = logging.getLogger(__name__)


class FromListConnector(AbstractConnector):
    """ Make connections according to a list.

    :param: conn_list:
        a list of tuples, one tuple for each connection. Each
        tuple should contain::

         (pre_idx, post_idx, weight, delay)

        where pre_idx is the index (i.e. order in the Population,
        not the ID) of the presynaptic neuron, and post_idx is
        the index of the postsynaptic neuron.
    """

    CONN_LIST_DTYPE = numpy.dtype([
        ("source", numpy.uint32), ("target", numpy.uint32),
        ("weight", numpy.float64), ("delay", numpy.float64)])

    def __init__(self, conn_list, safe=True, verbose=False, index_by_post=False):
        """
        Creates a new FromListConnector.
        """
        AbstractConnector.__init__(self, safe, verbose)
        if conn_list is None or len(conn_list) == 0:
            raise exceptions.InvalidParameterType(
                "The connection list for the FromListConnector must contain"
                " at least a list of tuples, each of which should contain:"
                " (pre_idx, post_idx)")

        self._conn_list = conn_list
        self._index_by_post = index_by_post
        # supports setting these at different times
        self._weights = None
        self._delays = None
        self._converted_weights_and_delays = False

        self._list_indices = {}
        self._list_ranges = {}
        self._pre_indices = {}
        self._post_indices = {}
        self._delay_maximum = None

    def _to_post_dict(self, conn_list):
        post_conn_dict = {}
        weights = []
        delays = []
        for i, conn in enumerate(conn_list):
            pre, post, w, d = conn
            weights.append(w)
            delays.append(d)
            if post not in post_conn_dict.keys():
                post_conn_dict[post] = {pre: i}
            else:
                post_conn_dict[post][pre] = i

        return post_conn_dict, weights, delays

    def _slices_str(self, pre_vtx_slice, post_vtx_slice):
        return "---%s\n---%s" % (pre_vtx_slice, post_vtx_slice)


    def _pre_indices_for_pre_post(self, pre_vertex_slice, post_vertex_slice):
        key = self._slices_str(pre_vertex_slice, post_vertex_slice)
        if key not in self._pre_indices:

            post_range = [post for post in self._post_dict.keys()
                            if post_vertex_slice.lo_atom <= post and \
                                post <= post_vertex_slice.hi_atom ]

            indices = numpy.asarray(\
                [pre for post in post_range for pre in self._post_dict[post]
                          if pre_vertex_slice.lo_atom <= pre and \
                              pre <= pre_vertex_slice.hi_atom], dtype='uint32')

            self._pre_indices[key] = indices
        else:
            indices = self._pre_indices[key]

        return indices

    def _post_indices_for_pre_post(self, pre_vertex_slice, post_vertex_slice):
        key = self._slices_str(pre_vertex_slice, post_vertex_slice)
        if key not in self._post_indices:

            post_range = [post for post in self._post_dict.keys()
                            if post_vertex_slice.lo_atom <= post and \
                                post <= post_vertex_slice.hi_atom ]

            indices = numpy.asarray( \
                [post for post in post_range for pre in self._post_dict[post]
                          if pre_vertex_slice.lo_atom <= pre and \
                              pre <= pre_vertex_slice.hi_atom], dtype='uint32')

            self._post_indices[key] = indices
        else:
            indices = self._post_indices[key]

        return indices

    def _indices_for_pre_post(self, pre_vertex_slice, post_vertex_slice):
        key = self._slices_str(pre_vertex_slice, post_vertex_slice)
        if key not in self._list_indices:

            post_range = [post for post in self._post_dict.keys()
                            if post_vertex_slice.lo_atom <= post and \
                                post <= post_vertex_slice.hi_atom ]

            self._list_ranges[key] = post_range

            indices = numpy.asarray(\
                [self._post_dict[post][pre] for post in post_range
                        for pre in self._post_dict[post]
                            if pre_vertex_slice.lo_atom <= pre and \
                                pre <= pre_vertex_slice.hi_atom], dtype='uint32')

            self._list_indices[key] = indices
        else:
            indices = self._list_indices[key]

        return indices

    @staticmethod
    def _split_conn_list(conn_list, column_names):
        """ takes the conn list and separates them into the blocks needed

        :param conn_list: the original conn list
        :param column_names: the column names if exist
        :return: source dest list, weights list, delays list, extra list
        """

        # weights and delay index
        weight_index = None
        delay_index = None

        # conn lists
        weights = None
        delays = None

        # locate weights and delay index in the listings
        if "weight" in column_names:
            weight_index = column_names.index("weight")
        if "delay" in column_names:
            delay_index = column_names.index("delay")
        element_index = range(2, len(column_names))

        # figure out where other stuff is
        conn_list = numpy.array(conn_list)
        source_destination_conn_list = conn_list[:, [0, 1]]

        if weight_index is not None:
            element_index.remove(weight_index)
            weights = conn_list[:, weight_index]
        if delay_index is not None:
            element_index.remove(delay_index)
            delays = conn_list[:, delay_index]

        # build other data element conn list (with source and destination)
        other_conn_list = None
        other_element_column_names = list()
        for element in element_index:
            other_element_column_names.append(column_names[element])
        if len(element_index) != 0:
            other_conn_list = conn_list[:, element_index]
            other_conn_list.dtype.names = other_element_column_names

        # hand over splitted data
        return source_destination_conn_list, weights, delays, other_conn_list

    def set_weights_and_delays(self, weights, delays):
        """ allows setting of the weights and delays at seperate times to the
        init, also sets the dtypes correctly.....

        :param weights:
        :param delays:
        :return:
        """
        # set the data if not already set (supports none overriding via
        # synapse data)
        if self._weights is None:
            self._weights = utility_calls.convert_param_to_numpy(
                weights, len(self._conn_list))
        if self._delays is None:
            self._delays = utility_calls.convert_param_to_numpy(
                delays, len(self._conn_list))

        # if got data, build connlist with correct dtypes
        if (self._weights is not None and self._delays is not None and not
        self._converted_weights_and_delays):

            if not self._index_by_post:
                # add weights and delays to the conn list
                temp_conn_list = numpy.dstack(
                    (self._conn_list[:, 0], self._conn_list[:, 1],
                     self._weights, self._delays))[0]

                self._conn_list = list()
                for element in temp_conn_list:
                    self._conn_list.append((element[0], element[1], element[2],
                                            element[3]))

                # set dtypes (cant we just set them within the array?)
            self._conn_list = numpy.asarray(self._conn_list,
                                            dtype=self.CONN_LIST_DTYPE)

            self._converted_weights_and_delays = True

    def get_delay_maximum(self):
        if self._delay_maximum is None:
            if self._index_by_post:
                print("post - max delay %s"%numpy.max(self._delays))
                d_max = numpy.max(self._delays)
            else:
                print("standard - max delay %s" % numpy.max(self._conn_list["delay"]))
                d_max = numpy.max(self._conn_list["delay"])
            self._delay_maximum = d_max


        return self._delay_maximum

    def get_delay_variance(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        if self._index_by_post:
            mask = self._indices_for_pre_post(pre_vertex_slice, post_vertex_slice)
            delays = self._delays[mask]
        else:
            mask = ((self._conn_list["source"] >= pre_vertex_slice.lo_atom) &
            (self._conn_list["source"] <= pre_vertex_slice.hi_atom) &
            (self._conn_list["target"] >= post_vertex_slice.lo_atom) &
            (self._conn_list["target"] <= post_vertex_slice.hi_atom))

            delays = self._conn_list["delay"][mask]
        print("%s - delay var %s"%\
              ("post" if self._index_by_post else "standard",
                numpy.var(delays)))
        if delays.size == 0:
            return 0
        return numpy.var(delays)

    def get_n_connections_from_pre_vertex_maximum(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice,
            min_delay=None, max_delay=None):

        mask = None
        if min_delay is None or max_delay is None:
            if self._index_by_post:
                sources = self._pre_indices_for_pre_post(
                                    pre_vertex_slice, post_vertex_slice)
            else:
                mask = ((self._conn_list["source"] >= pre_vertex_slice.lo_atom) &
                (self._conn_list["source"] <= pre_vertex_slice.hi_atom) &
                (self._conn_list["target"] >= post_vertex_slice.lo_atom) &
                (self._conn_list["target"] <= post_vertex_slice.hi_atom))
        else:
            if self._index_by_post:
                all_mask = self._indices_for_pre_post(pre_vertex_slice, post_vertex_slice)
                mask = [i for i in all_mask
                            if min_delay <= self._delays[i] and \
                                self._delays[i] <= max_delay]

                sources = self._conn_list["source"][mask]
            else:
                mask = ((self._conn_list["source"] >= pre_vertex_slice.lo_atom) &
                (self._conn_list["source"] <= pre_vertex_slice.hi_atom) &
                (self._conn_list["target"] >= post_vertex_slice.lo_atom) &
                (self._conn_list["target"] <= post_vertex_slice.hi_atom) &
                (self._conn_list["delay"] >= min_delay) &
                (self._conn_list["delay"] <= max_delay))

                sources = self._conn_list["source"][mask]

        print("%s - n from pre %s"%\
              ("post" if self._index_by_post else "standard",
               numpy.max(numpy.bincount(sources.view('int32')))))


        if sources.size == 0:
            return 0
        return numpy.max(numpy.bincount(sources.view('int32')))

    def get_n_connections_to_post_vertex_maximum(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        if self._index_by_post:
            targets = self._post_indices_for_pre_post(pre_vertex_slice, post_vertex_slice)
        else:
            mask = ((self._conn_list["source"] >= pre_vertex_slice.lo_atom) &
            (self._conn_list["source"] <= pre_vertex_slice.hi_atom) &
            (self._conn_list["target"] >= post_vertex_slice.lo_atom) &
            (self._conn_list["target"] <= post_vertex_slice.hi_atom))
            targets = self._conn_list["target"][mask]

        print("%s - n to post %s"%\
              ("post" if self._index_by_post else "standard",
               numpy.max(numpy.bincount(targets.view('int32')))))

        if targets.size == 0:
            return 0
        return numpy.max(numpy.bincount(targets.view('int32')))



    def get_weight_mean(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        if self._index_by_post:
            mask = self._indices_for_pre_post(pre_vertex_slice, post_vertex_slice)
            weights = self._weights[mask]
        else:
            mask = ((self._conn_list["source"] >= pre_vertex_slice.lo_atom) &
            (self._conn_list["source"] <= pre_vertex_slice.hi_atom) &
            (self._conn_list["target"] >= post_vertex_slice.lo_atom) &
            (self._conn_list["target"] <= post_vertex_slice.hi_atom))
            weights = self._conn_list["weight"][mask]

        print("%s - weight mean %s"%\
              ("post" if self._index_by_post else "standard",
               numpy.mean(weights)
               ))

        if weights.size == 0:
            return 0
        return numpy.mean(weights)

    def get_weight_maximum(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        if self._index_by_post:
            mask = self._indices_for_pre_post(pre_vertex_slice, post_vertex_slice)
            weights = self._weights[mask]
        else:
            mask = ((self._conn_list["source"] >= pre_vertex_slice.lo_atom) &
            (self._conn_list["source"] <= pre_vertex_slice.hi_atom) &
            (self._conn_list["target"] >= post_vertex_slice.lo_atom) &
            (self._conn_list["target"] <= post_vertex_slice.hi_atom))
            weights = self._conn_list["weight"][mask]

        print("%s - weight max %s"%\
              ("post" if self._index_by_post else "standard",
               numpy.max(weights)
               ))

        if weights.size == 0:
            return 0
        return numpy.max(weights)

    def get_weight_variance(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice):
        if self._index_by_post:
            mask = self._indices_for_pre_post(pre_vertex_slice, post_vertex_slice)
            weights = self._weights[mask]
        else:
            mask = ((self._conn_list["source"] >= pre_vertex_slice.lo_atom) &
            (self._conn_list["source"] <= pre_vertex_slice.hi_atom) &
            (self._conn_list["target"] >= post_vertex_slice.lo_atom) &
            (self._conn_list["target"] <= post_vertex_slice.hi_atom))
            weights = self._conn_list["weight"][mask]

        print("%s - weight variance %s"%\
              ("post" if self._index_by_post else "standard",
               numpy.var(weights)
               ))

        if weights.size == 0:
            return 0
        return numpy.var(weights)

    def generate_on_machine(self):
        return False

    def create_synaptic_block(
            self, pre_slices, pre_slice_index, post_slices,
            post_slice_index, pre_vertex_slice, post_vertex_slice,
            synapse_type):
        # print("FromListConnector - create_synaptic_block: pre = %s, post = %s"%
        #       (pre_slice_index, post_slice_index))
        if self._index_by_post:
            mask = self._indices_for_pre_post(pre_vertex_slice, post_vertex_slice)
        else:
            mask = ((self._conn_list["source"] >= pre_vertex_slice.lo_atom) &
            (self._conn_list["source"] <= pre_vertex_slice.hi_atom) &
            (self._conn_list["target"] >= post_vertex_slice.lo_atom) &
            (self._conn_list["target"] <= post_vertex_slice.hi_atom))


        if self._index_by_post:
            block = numpy.zeros(
                        len(mask), dtype=AbstractConnector.NUMPY_SYNAPSES_DTYPE)
            block[:] = self._conn_list[mask]
            block["delay"] = self._clip_delays(block["delay"])
            block["synapse_type"] = synapse_type
        else:
            items = self._conn_list[mask]
            block = numpy.zeros(
                        items.size, dtype=AbstractConnector.NUMPY_SYNAPSES_DTYPE)
            block["source"] = items["source"]
            block["target"] = items["target"]
            block["weight"] = items["weight"]
            block["delay"] = self._clip_delays(items["delay"])
            block["synapse_type"] = synapse_type
        return block

    def __repr__(self):
        return "FromListConnector(n_connections={})".format(
            len(self._conn_list))

    @property
    def conn_list(self):
        return self._conn_list

    @conn_list.setter
    def conn_list(self, new_value):
        self._conn_list = new_value

    def _set_data(self, new_value, name):
        for index in self._conn_list:
            for (source, dest) in self._conn_list[index]:  # @UnusedVariable
                pass

    def gen_on_machine_info(self):
        return []
