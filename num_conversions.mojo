from tensor import Tensor, TensorSpec


fn si8_to_ui8(int_tensor: Tensor[DType.int8]) raises -> Tensor[DType.uint8]:
    let spec = TensorSpec(DType.uint8, int_tensor.shape())
    var unsigned_ints = Tensor[DType.uint8](spec)
    for i in range(int_tensor.num_elements()):
        let signed_int = int_tensor[i]
        let unsigned_int = signed_int.cast[DType.uint8]()
        unsigned_ints[i] = unsigned_int
    return unsigned_ints


fn int8_to_int32(uint8_tensor: Tensor[DType.uint8]) raises -> SIMD[DType.int32, 1]:
    if uint8_tensor.shape().rank() > 1 and not uint8_tensor.shape()[0] == 1:
        raise 'Wrong shape'
    let result = (
            (uint8_tensor[0].to_int() << 24) |
            (uint8_tensor[1].to_int() << 16) |
            (uint8_tensor[2].to_int() << 8) |
            uint8_tensor[3].to_int()
        )
    return SIMD[DType.int32, 1](result)
