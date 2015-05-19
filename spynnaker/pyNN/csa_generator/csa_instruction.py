from six import add_metaclass
from abc import ABCMeta
from abc import abstractmethod


@add_metaclass(ABCMeta)
class CSAInstruction(object):

    @abstractmethod
    def write_bytecode(self, dsg):
        """ Writes out the bytecode to the output

        :param dsg: A data specification generator to write using
        :type dsg:\
                   :py:class:`data_specification.data_specification_generator.DataSpecificationGenerator`
        """
