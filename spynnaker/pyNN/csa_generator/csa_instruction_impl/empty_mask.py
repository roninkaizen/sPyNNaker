from spynnaker.pyNN.csa_generator.csa_instruction import CSAInstruction
from spynnaker.pyNN.csa_generator.csa_instruction_type\
    import CSAInstructionType
from spynnaker.pyNN.csa_generator import csa_utils


class EmptyMask(CSAInstruction):

    def write_bytecode(self, dsg):

        # Write the parameters (there are none)

        # Write the instruction to call
        csa_utils.write_instruction_code(CSAInstructionType.EMPTY_MASK, dsg)
