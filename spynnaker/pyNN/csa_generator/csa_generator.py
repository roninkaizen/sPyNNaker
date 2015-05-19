class CSAGenerator(object):
    """ Generates the bytecode for a series of CSA instructions
    """

    def __init__(self, dsg):
        """

        :param dsg: The data specification generator
        :type dsg:\
                   :py:class:`data_specification.data_specification_generator.DataSpecificationGenerator`
        """
        self._dsg = dsg

    def write_bytecode(self, csa_instruction):
        """ Write a CSA instruction to the generator
        """
        csa_instruction.write_bytecode(self._dsg)
