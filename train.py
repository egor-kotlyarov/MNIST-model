import os

import mnist_loader
from network import Network


def main():
    training_data, validation_data, test_data = (
        mnist_loader.load_data_wrapper()
    )

    training_data = list(training_data)
    test_data = list(test_data)

    network = [784, 15, 15, 15, 10]
    name = ''
    net = Network(network)

    net.SGD(
        training_data,
        epochs=50,
        mini_batch_size=16,
        eta=3.0,
        test_data=test_data
    )

    os.makedirs("models", exist_ok=True)

    for i in range(len(network) - 1):
        name += str(network[i])
        name += '-'

    name += str(network[-1])

    net.save(f"models/{name}.pkl")
    net.save_json(f"models/{name}.json")
    print(f"{name}'s training is complete")


if __name__ == "__main__":
    main()