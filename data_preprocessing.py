import numpy as np
import tarfile
import pickle
import pandas as pd


def main():
    (
        train_1_data,
        train_2_data,
        train_3_data,
        train_4_data,
        train_5_data,
        test_data,
        class_names,
    ) = unpickle()

    filtered_labels, filtered_test_data = filter_data_by_class(
        train_1_data,
        train_2_data,
        train_3_data,
        train_4_data,
        train_5_data,
        test_data,
        class_names,
    )
    X_train, Y_train = combine_dataset1_train_data(
        train_1_data,
        train_2_data,
        train_3_data,
        train_4_data,
        train_5_data,
    )
    X_train, Y_train, X_test, Y_test = check_data(X_train, Y_train, test_data)


def extract_datasets():
    # Read CIFAR-10 File
    with tarfile.open("Datasets/cifar-10-python.tar.gz", "r") as tar:
        tar.extractall(path="./dataset1_classes")

    # Read the CIFAR-100 File
    with tarfile.open("Datasets/cifar-100-python.tar.gz", "r") as tar:
        tar.extractall(path="./dataset2_classes")

    # # Read the class names file
    # with tarfile.open("Datasets/cifar-100-python.tar.gz", "r") as tar:
    #     tar.extractall(path = "./classes.meta")


def unpickle():
    with open("dataset1_classes/cifar-10-batches-py/train_1", "rb") as f:
        train_1_data = pickle.load(f, encoding="bytes")

    with open("dataset1_classes/cifar-10-batches-py/train_2", "rb") as f:
        train_2_data = pickle.load(f, encoding="bytes")

    with open("dataset1_classes/cifar-10-batches-py/train_3", "rb") as f:
        train_3_data = pickle.load(f, encoding="bytes")

    with open("dataset1_classes/cifar-10-batches-py/train_4", "rb") as f:
        train_4_data = pickle.load(f, encoding="bytes")

    with open("dataset1_classes/cifar-10-batches-py/train_5", "rb") as f:
        train_5_data = pickle.load(f, encoding="bytes")

    with open("dataset1_classes/cifar-10-batches-py/test", "rb") as f:
        test_data = pickle.load(f, encoding="bytes")

    with open("dataset1_classes/cifar-10-batches-py/classes.meta", "rb") as f:
        class_names = pickle.load(f, encoding="bytes")

    # class_names = meta["label_names"]
    # class_names = pickle.load(f, encoding="bytes")

    return (
        train_1_data,
        train_2_data,
        train_3_data,
        train_4_data,
        train_5_data,
        test_data,
        class_names,
    )


def filter_data_by_class(
    train_1_data,
    train_2_data,
    train_3_data,
    train_4_data,
    train_5_data,
    test_data,
    class_names,
):
    # automobile, bird, cat, deer, dog, horse, and truck
    required_classes = [1, 2, 5, 3, 4, 7, 9]
    training_data = [
        train_1_data,
        train_2_data,
        train_3_data,
        train_4_data,
        train_5_data,
    ]
    filtered_train_data = []

    for train_data in training_data:
        for label in train_data[b"labels"]:
            if label in required_classes:
                filtered_train_data.append(label)
                print("Found", label)

    filtered_test_data = []

    for label in test_data[b"labels"]:
        if label in required_classes:
            filtered_test_data.append(label)
            print("Test: found", label)

    print("new training classes", len(filtered_train_data))
    # print("new test classes", len(filtered_test_data))
    return filtered_test_data, filtered_test_data


# we could potentially cut out the need for this function, and just defin
def combine_dataset1_train_data(
    train_1_data, train_2_data, train_3_data, train_4_data, train_5_data
):

    X_train = np.vstack(
        [
            train_1_data[b"data"],
            train_2_data[b"data"],
            train_3_data[b"data"],
            train_4_data[b"data"],
            train_5_data[b"data"],
        ]
    )

    Y_train = np.hstack(
        [
            train_1_data[b"labels"],
            train_2_data[b"labels"],
            train_3_data[b"labels"],
            train_4_data[b"labels"],
            train_5_data[b"labels"],
        ]
    )
    print(X_train.shape)
    print(Y_train.shape)
    return X_train, Y_train


def check_data(X_train, Y_train, test_data):
    # we have already defined the features and labels for x_train and y_train

    # print("Show type train: ", type(train_data))
    print("Show type test:", type(test_data))

    X_test = test_data[b"data"]
    Y_test = np.array(test_data[b"labels"])
    # print(X_test.shape)
    assert (
        X_train.shape[0] == Y_train.shape[0]
    ), "The number of training images is not equal to the number of labels"
    assert (
        X_test.shape[0] == Y_test.shape[0]
    ), "The number of testing images is not equal to the number of labels"

    # Reshape - as dimensions were not 32 x 32 3 after making changes to the data
    X_train = X_train.reshape(50000, 32, 32, 3)
    X_test = X_test.reshape(10000, 32, 32, 3)

    # checking the images are same size
    assert X_train.shape[1:] == (
        32,
        32,
        3,
    ), "Xtrain: The dimensions of the training images are not 32 x 32 x 3"
    assert X_test.shape[1:] == (
        32,
        32,
        3,
    ), "Xtest: The dimensions of the testing images are not 32 x 32 x 3"
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    main()
