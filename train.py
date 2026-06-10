import os

import mnist_loader
from network import Network


def main():
    training_data, validation_data, test_data = (
        mnist_loader.load_data_wrapper()
    )

    training_data = list(training_data)
    test_data = list(test_data)

    net = Network([784, 15, 15, 15, 10])

    net.SGD(
        training_data,
        epochs=50,
        mini_batch_size=10,
        eta=3.0,
        test_data=test_data
    )

    os.makedirs("models", exist_ok=True)
    net.save("models/784-15-15-15-10.pkl")
    net.save_json("models/784-15-15-15-10.json")


if __name__ == "__main__":
    main()