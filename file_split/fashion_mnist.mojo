from collections.dict import Dict, KeyElement

@value
struct IntKey(KeyElement):
    var n: Int

    fn __init__(inout self, owned n: Int):
        self.n = n

    fn __hash__(self) -> Int:
        return hash(self.n)

    fn __eq__(self, other: Self) -> Bool:
        return self.n == other.n

fn fashion_mnist_key() -> Dict[IntKey, String]:
    let fashion_mnist_key = Dict[IntKey, String]()
    fashion_mnist_key[0] = "T-shirt/top"
    fashion_mnist_key[1] = "Trouser"
    fashion_mnist_key[2] = "Pullover"
    fashion_mnist_key[3] = "Dress"
    fashion_mnist_key[4] = "Coat"
    fashion_mnist_key[5] = "Sandal"
    fashion_mnist_key[6] = "Shirt"
    fashion_mnist_key[7] = "Sneaker"
    fashion_mnist_key[8] = "Bag"
    fashion_mnist_key[9] = "Ankle boot"
    return fashion_mnist_key