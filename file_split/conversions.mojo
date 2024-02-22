fn uint8_to_int32(uint8_tensor: Tensor[DType.uint8]) raises -> SIMD[DType.int32, 1]:
    let result = (
        (uint8_tensor[0].to_int() << 24)
        | (uint8_tensor[1].to_int() << 16)
        | (uint8_tensor[2].to_int() << 8)
        | uint8_tensor[3].to_int()
    )
    return SIMD[DType.int32, 1](result)
