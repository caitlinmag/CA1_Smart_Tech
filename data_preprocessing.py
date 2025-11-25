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
        train_dataset_2,
        test_dataset_2,
        classes,
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

    filter_dataset_2_by_class(
        train_dataset_2,
        test_dataset_2,
        classes,
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

    with open("dataset2_classes/cifar-100-python/train", "rb") as f:
        train_dataset_2 = pickle.load(f, encoding="bytes")

    with open("dataset2_classes/cifar-100-python/test", "rb") as f:
        test_dataset_2 = pickle.load(f, encoding="bytes")

    with open("dataset2_classes/cifar-100-python/meta", "rb") as f:
        classes = pickle.load(f, encoding="bytes")

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
        train_dataset_2,
        test_dataset_2,
        classes,
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


def filter_dataset_2_by_class(
        train_dataset_2,
        test_dataset_2,
        classes,
):
    required_classes = [b"cattle", b"fox", b"baby", b"boy", b"girl", b"man", b"woman", b"rabbit", b"squirrel", b"bicycle", b"bus", b"motorcycle", b"pickup_truck", b"train", b"lawn_mower", b"tractor"]
    required_classes_2 = [2, 8, 11, 13, 19, 34, 35, 41, 46, 48, 58, 65, 80, 89, 90, 98]

    filtered_fine_classes = []

    for label in classes[b"fine_label_names"]:
        if label in required_classes:
            # removes the bytes from beginning of strings
            label = label.decode("utf-8")
            filtered_fine_classes.append(label)
            print("Class: Found", label)

    filtered_fine_train_data_2 = []

    for label in train_dataset_2[b"fine_labels"]:
        if label in required_classes_2:
            filtered_fine_train_data_2.append(label)
            print("Train: Found", label)

    filtered_fine_test_data_2 = []

    for label in test_dataset_2[b"fine_labels"]:
        if label in required_classes_2:
            filtered_fine_test_data_2.append(label)
            print("Test Label: Found", label)

    filtered_coarse_classes = []

    for label in classes[b"coarse_label_names"]:
        if label in required_classes:
            # removes the bytes from beginning of strings
            label = label.decode("utf-8")
            filtered_coarse_classes.append(label)
            print("Coarse Class: Found", label)

    filtered_coarse_train_data_2 = []

    for label in train_dataset_2[b"coarse_labels"]:
        if label in required_classes_2:
            filtered_coarse_train_data_2.append(label)
            print("Coarse Train: Found", label)

    filtered_coarse_test_data_2 = []

    for label in test_dataset_2[b"coarse_labels"]:
        if label in required_classes_2:
            filtered_coarse_test_data_2.append(label)
            print("Coarse Test Label: Found", label)

    
    return filtered_fine_classes, filtered_coarse_classes, filtered_fine_train_data_2, filtered_coarse_train_data_2, filtered_fine_test_data_2, filtered_coarse_test_data_2


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
