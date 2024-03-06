from tensor import Tensor, TensorShape
from utils.index import Index
from algorithm import parallelize, vectorize
from algorithm.reduction import sum
from memory.buffer import Buffer

alias nelts = simdwidthof[DType.float32]()

fn get_slice[type: DType](tensor: Tensor[type], start_index: Int, end_index: Int) raises -> Tensor[type]:
    if end_index < start_index:
        raise "End index more than start index."
    elif end_index == start_index:
        var output_tensor = Tensor[type](1)
        output_tensor[0] = tensor[start_index]

        return output_tensor
    else:
        var output_tensor = Tensor[type](end_index - start_index)
        for i in range(start_index, end_index):
            output_tensor[i - start_index] = tensor[i]
        return output_tensor


fn tensor_print[type: DType](index: Int, tensor: Tensor[type]):
    var cur_line: String

    fn get_char_for_pixel(pixel_value: Int) -> String:
        if pixel_value == 0:
            return " "
        elif pixel_value < 32:
            return "."
        elif pixel_value < 64:
            return ","
        elif pixel_value < 96:
            return ":"
        elif pixel_value < 128:
            return ";"
        elif pixel_value < 160:
            return "o"
        elif pixel_value < 192:
            return "O"
        elif pixel_value < 224:
            return "X"
        else:
            return "#"

    for j in range(tensor.shape()[1]):
        cur_line = ""
        for k in range(tensor.shape()[2]):
            cur_line += get_char_for_pixel(tensor[Index(index, j, k)].to_int()) + " "
        print(cur_line)


fn matmul[type: DType](first_matrix: Tensor[type], second_matrix: Tensor[type]) raises -> Tensor[DType.float32]:
    let f_m = first_matrix.astype[DType.float32]()
    let s_m = second_matrix.astype[DType.float32]()
    if first_matrix.rank() != 2 or second_matrix.rank() != 2:
        raise 'At least one of the tensors is not a matrix'
    if first_matrix.dim(1) != second_matrix.dim(0):
        raise 'Then matrices are not compatible for matrix multiplication'

    let o_m_rows = f_m.dim(0)
    let o_m_columns = s_m.dim(1)
    var o_m = Tensor[DType.float32](o_m_rows * o_m_columns)
    for i in range(o_m_rows):
        for j in range(f_m.dim(1)):
            for k in range(o_m_columns):
                o_m[i*o_m_columns+k] += f_m[i, j] * s_m[j, k]
    return o_m.reshape(TensorShape(o_m_rows, o_m_columns))


fn matmul_vectorized_working[type: DType](first_matrix: Tensor[type], second_matrix: Tensor[type]) raises -> Tensor[DType.float32]:
    let f_m = first_matrix.astype[DType.float32]()
    let s_m = second_matrix.astype[DType.float32]()
    if first_matrix.rank() != 2 or second_matrix.rank() != 2:
        raise 'At least one of the tensors is not a matrix'
    if first_matrix.dim(1) != second_matrix.dim(0):
        raise 'Then matrices are not compatible for matrix multiplication'

    let o_m_rows = f_m.dim(0)
    let o_m_cols = s_m.dim(1)
    var o_m = Tensor[DType.float32](o_m_rows * o_m_cols)
    for i in range(o_m_rows):
        for j in range(f_m.dim(1)):
            for kv in range(0, nelts*(o_m_cols//nelts), nelts):
                o_m.simd_store[nelts](i*o_m_cols+kv, o_m.simd_load[nelts](i*o_m_cols+kv) + f_m.simd_load[1](i*f_m.dim(1)+j) * s_m.simd_load[nelts](j*s_m.dim(1)+kv))
            for k in range(nelts*(o_m_cols//nelts), o_m_cols):
                o_m.simd_store[1](i*o_m_cols+k, o_m.simd_load[1](i*o_m_cols+k) + f_m.simd_load[1](i*f_m.dim(1)+j) * s_m.simd_load[1](j*s_m.dim(1)+k))
    return o_m.reshape(TensorShape(o_m_rows, o_m_cols))


fn sum_on_rows(dvalues: Tensor[DType.float32]) raises -> Tensor[DType.float32]:
    var rows = dvalues.dim(0)
    var cols = dvalues.dim(1)

    var output_tensor = Tensor[DType.float32](rows, 1)
    
    for i in range(rows):
        for j in range(0, nelts*(cols//nelts), nelts):
            var sum_buffer = Buffer[DType.float32, nelts]()
            print(sum_buffer.simd_load[nelts](0))
            sum_buffer.simd_store[nelts](0, dvalues.simd_load[nelts](i*cols+j))
            print('a')
            print(sum(sum_buffer))
            output_tensor.simd_store[1](i, sum(sum_buffer) + output_tensor.simd_load[1](i))
        for k in range(nelts*(cols//nelts), cols):
            output_tensor.simd_store[1](i, dvalues.simd_load[1](i*cols+k) + output_tensor.simd_load[1](i))
    return output_tensor
