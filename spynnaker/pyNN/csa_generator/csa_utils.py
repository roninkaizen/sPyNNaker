from data_specification.enums.data_type import DataType


def write_instruction_code(instruction_type, dsg):
    dsg.write_data(instruction_type.value, data_type=DataType.UINT8)


def write_uint(value, dsg):
    if value <= 0xFF:
        dsg.write_data(1, data_type=DataType.UINT8)
        dsg.write_data(value, data_type=DataType.UINT8)
    elif value <= 0xFFFF:
        dsg.write_data(2, data_type=DataType.UINT8)
        dsg.write_data(value, data_type=DataType.UINT16)
    elif value <= 0xFFFFFFFF:
        dsg.write_data(4, data_type=DataType.UINT8)
        dsg.write_data(value, data_type=DataType.UINT32)


def write_accum(value, dsg):
    # TODO
    pass


def write_fract(value, dsg):
    # TODO
    pass
