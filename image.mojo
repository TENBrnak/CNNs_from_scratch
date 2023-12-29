from tensor import TensorSpec, Tensor

struct Image:
    var width: Int
    var height: Int
    var img_data: Tensor[DType.int16]

    fn __init__(inout self, width: Int, height: Int):
        self.width = width
        self.height = height
        let spec = TensorSpec(DType.int16, width, height)
        self.img_data = Tensor[DType.int16](spec)

    fn populate_image(inout self, data: Tensor[DType.int16]):
        pass