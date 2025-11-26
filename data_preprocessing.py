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
        cifar10_classes,
        train_dataset_2,
        test_dataset_2,
        cifar100_classes,
    ) = unpickle()

    X_train_1, Y_train_1, X_test_1, Y_test_1 = filter_cifar10_by_class(
        train_1_data,
        train_2_data,
        train_3_data,
        train_4_data,
        train_5_data,
        test_data,
        cifar10_classes,
    )

    X_train_2, Y_train_2, X_test_2, Y_test_2 = filter_cifar100_by_class(
        train_dataset_2,
        test_dataset_2,
        cifar100_classes,
    )

    X_train, Y_train, X_test, Y_test = combine_datasets(
        X_train_1,
        Y_train_1,
        X_test_1,
        Y_test_1,
        X_train_2,
        Y_train_2,
        X_test_2,
        Y_test_2,
    )

    # X_train, Y_train, X_test, Y_test = check_data(X_train, Y_train, test_data)


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
        cifar10_classes = pickle.load(f, encoding="bytes")

    with open("dataset2_classes/cifar-100-python/train", "rb") as f:
        train_dataset_2 = pickle.load(f, encoding="bytes")

    with open("dataset2_classes/cifar-100-python/test", "rb") as f:
        test_dataset_2 = pickle.load(f, encoding="bytes")

    with open("dataset2_classes/cifar-100-python/meta", "rb") as f:
        cifar100_classes = pickle.load(f, encoding="bytes")

    # class_names = meta["label_names"]
    # class_names = pickle.load(f, encoding="bytes")

    return (
        train_1_data,
        train_2_data,
        train_3_data,
        train_4_data,
        train_5_data,
        test_data,
        cifar10_classes,
        train_dataset_2,
        test_dataset_2,
        cifar100_classes,
    )


def filter_cifar10_by_class(
    train_1_data,
    train_2_data,
    train_3_data,
    train_4_data,
    train_5_data,
    test_data,
    cifar10_classes,
):
    train_data_list = []
    train_labels_list = []
    test_data_list = []
    test_labels_list = []

    # automobile, bird, cat, deer, dog, horse, and truck
    # checked the required class numbers by using simple print statements
    required_classes = [1, 2, 5, 3, 4, 7, 9]
    training_data = [
        train_1_data,
        train_2_data,
        train_3_data,
        train_4_data,
        train_5_data,
    ]

    for train in training_data:
        train_data = train[b"data"]
        train_labels = train[b"labels"]

        for i in range(len(train_labels)):
            if train_labels[i] in required_classes:
                train_data_list.append(train_data[i])
                train_labels_list.append(train_labels[i])

    # Convert lists to the xtrain, ytrain
    X_train_1 = np.array(train_data_list)
    Y_train_1 = np.array(train_labels_list)

    # print("X train shape:", X_train_1.shape)
    # print("Y train shape:", Y_train_1.shape)
    # print("Y train", np.unique(Y_train_1))  # checking the labels

    data = test_data[b"data"]
    labels = test_data[b"labels"]

    for i in range(len(labels)):
        if labels[i] in required_classes:
            test_data_list.append(data[i])
            test_labels_list.append(labels[i])

    # Convert lists to the xtrain, ytrain
    X_test_1 = np.array(test_data_list)
    Y_test_1 = np.array(test_labels_list)

    # print("X test shape:", X_test_1.shape)
    # print("Y test shape:", Y_test_1.shape)
    # print("Y test", np.unique(Y_test_1))  # checking the labels
    return X_train_1, Y_train_1, X_test_1, Y_test_1


def filter_cifar100_by_class(
    train_dataset_2,
    test_dataset_2,
    cifar100_classes,
):
    # required_classes = [
    #     b"cattle",
    #     b"fox",
    #     b"baby",
    #     b"boy",
    #     b"girl",
    #     b"man",
    #     b"woman",
    #     b"rabbit",
    #     b"squirrel",
    #     b"bicycle",
    #     b"bus",
    #     b"motorcycle",
    #     b"pickup_truck",
    #     b"train",
    #     b"lawn_mower",
    #     b"tractor",
    # ]
    # required_superclasses = [
    #     "trees"
    # ]

    # cattle, fox, baby, boy, girl, man, woman, rabbit, squirrel, bicycle, bus, motorcycle, pickup_truck, train, lawn_mower, tractor
    required_classes_num = [
        2,
        8,
        11,
        13,
        19,
        34,
        35,
        41,
        46,
        48,
        58,
        65,
        80,
        89,
        90,
        98,
    ]
    # trees
    required_superclasses_num = [17]

    train_data_list = []
    test_data_list = []

    test_fine_labels_list = []

    test_coarse_labels_list = []

    train_labels_list = []

    train_data = train_dataset_2[b"data"]
    train_fine_labels = train_dataset_2[b"fine_labels"]
    train_coarse_labels = train_dataset_2[b"coarse_labels"]

    test_data = test_dataset_2[b"data"]
    test_fine_labels = test_dataset_2[b"fine_labels"]
    test_coarse_labels = test_dataset_2[b"coarse_labels"]

    for i in range(len(train_fine_labels)):
        if train_fine_labels[i] in required_classes_num:
            train_data_list.append(train_data[i])
            train_labels_list.append(train_fine_labels[i])

    for i in range(len(train_coarse_labels)):
        if train_coarse_labels[i] in required_superclasses_num:
            train_data_list.append(train_data[i])
            train_labels_list.append(train_coarse_labels[i])

    # Combine Class and Superclass Label Lists
    # train_labels_list = train_fine_labels_list + train_fine_labels_list

    # Convert Lists to X_train, Y_train
    X_train_2 = np.array(train_data_list)
    Y_train_2 = np.array(train_labels_list)

    for i in range(len(test_fine_labels)):
        if test_fine_labels[i] in required_classes_num:
            test_data_list.append(test_data[i])
            test_fine_labels_list.append(test_fine_labels[i])

    for i in range(len(test_coarse_labels)):
        if test_coarse_labels[i] in required_superclasses_num:
            test_data_list.append(test_data[i])
            test_coarse_labels_list.append(test_coarse_labels[i])

    # Combine Class and Superclass Label Lists
    test_labels_list = test_fine_labels_list + test_coarse_labels_list
    # Convert Lists to X_test, Y_test
    X_test_2 = np.array(test_data_list)
    Y_test_2 = np.array(test_labels_list)

    # print("X train shape:", X_train_2.shape)
    # print("Y train shape:", Y_train_2.shape)
    # print("Y train", np.unique(Y_train_2))
    # print("X test shape:", X_test_2.shape)
    # print("Y test shape:", Y_test_2.shape)
    # print("Y test", np.unique(Y_test_2))
    return X_train_2, Y_train_2, X_test_2, Y_test_2


def combine_datasets(
    X_train_1, Y_train_1, X_test_1, Y_test_1, X_train_2, Y_train_2, X_test_2, Y_test_2
):
    cifar_10_labels = [
        b"automobile",
        b"bird",
        b"cat",
        b"deer",
        b"dog",
        b"horse",
        b"truck",
    ]
    cifar_100_labels = [
        b"cattle",
        b"fox",
        b"baby",
        b"boy",
        b"girl",
        b"man",
        b"woman",
        b"rabbit",
        b"squirrel",
        b"bicycle",
        b"bus",
        b"motorcycle",
        b"pickup_truck",
        b"train",
        b"lawn_mower",
        b"tractor",
        b"trees",
    ]

    # mapping unique ids to each class name
    all_labels = sorted(list(set(cifar_10_labels) | set(cifar_100_labels)))
    unique_id = {name: i for i, name in enumerate(all_labels)}

    cifar_10_new = np.array([unique_id[name] for name in cifar_10_labels])
    cifar_100_new = np.array([unique_id[name] for name in cifar_100_labels])

    X_train = np.vstack([X_train_1, X_train_2])
    Y_train = np.hstack([cifar_10_new, cifar_100_new])

    X_test = np.vstack([X_test_1, X_test_2])
    Y_test = np.hstack([Y_test_1, Y_test_2])

    print("X train:", X_train.shape)
    print("Y train:", Y_train.shape)
    print("X test:", X_test.shape)
    print("Y test:", Y_test.shape)
    print("Y train", np.unique(Y_train))
    print("Y test", np.unique(Y_test))

    train_count = len(np.unique(Y_train))
    print("Y train count:", train_count)

    test_count = len(np.unique(Y_test))
    print("Y test count", test_count)
    return X_train, Y_train, X_test, Y_test


def check_data(X_train, Y_train, test_data):
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
