from data_load import read_data_as_images, read_data_as_labels
from tensor_addons import matmul_vectorized_working
from random import rand

fn main() raises:
    let base_dir = "/Users/tprazak/Documents/seminary_work_nn/MNIST_in_mojo/"
    let image_path_mnist = base_dir + "mnist/train-images.idx3-ubyte"
    let labels_path_mnist = base_dir + "mnist/train-labels.idx1-ubyte"
    let image_path_fashion = base_dir + "fashion_mnist/train-images-idx3-ubyte"
    let labels_path_fashion = base_dir + "fashion_mnist/train-labels-idx1-ubyte"
    let images = read_data_as_images(image_path_mnist)
    let labels = read_data_as_labels(labels_path_mnist)
    let fashion_images = read_data_as_images(image_path_fashion)
    let fashion_labels = read_data_as_labels(labels_path_fashion)

    let a = rand[DType.float32](5, 8)
    let b = rand[DType.float32](8, 24)
    let result = matmul_vectorized_working(a, b)
    if result != control:
        print(result)
        print(control)