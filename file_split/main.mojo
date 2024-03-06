from data_load import read_data_as_images, read_data_as_labels
from tensor_addons import matmul_vectorized_working, sum_on_rows
from random import rand

fn main() raises:
    var base_dir = "/Users/tprazak/Documents/seminary_work_nn/MNIST_in_mojo/"
    var image_path_mnist = base_dir + "mnist/train-images.idx3-ubyte"
    var labels_path_mnist = base_dir + "mnist/train-labels.idx1-ubyte"
    var image_path_fashion = base_dir + "fashion_mnist/train-images-idx3-ubyte"
    var labels_path_fashion = base_dir + "fashion_mnist/train-labels-idx1-ubyte"
    var images = read_data_as_images(image_path_mnist)
    var labels = read_data_as_labels(labels_path_mnist)
    var fashion_images = read_data_as_images(image_path_fashion)
    var fashion_labels = read_data_as_labels(labels_path_fashion)

    var test = rand[DType.float32](4, 8)
    print(test)
    print(sum_on_rows(test))