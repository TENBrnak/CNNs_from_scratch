from conversions import uint8_to_int32
from tensor_addons import get_slice
from tensor import TensorShape

fn read_data_as_images(images_path: Path) raises -> Tensor[DType.uint8]:
    if images_path.exists():
        let image_file = images_path.read_bytes().astype[DType.uint8]()
        let num_images = uint8_to_int32(get_slice[DType.uint8](image_file, 4, 8))
        let width = uint8_to_int32(get_slice[DType.uint8](image_file, 8, 12))
        let height = uint8_to_int32(get_slice[DType.uint8](image_file, 12, 16))

        var image_data = get_slice(image_file, 16, image_file.num_elements())

        let images_shape = TensorShape(
            num_images.to_int(), height.to_int(), width.to_int()
        )

        let images = image_data.reshape(images_shape)
        return images
    raise "The images directory does not exist."


fn read_data_as_labels(labels_path: Path) raises -> Tensor[DType.uint8]:
    if labels_path.exists():
        let label_file = labels_path.read_bytes().astype[DType.uint8]()
        let num_labels = uint8_to_int32(get_slice[DType.uint8](label_file, 4, 8))

        let labels = get_slice(label_file, 8, num_labels.to_int() + 8)

        return labels
    raise "The labels directory does not exist."