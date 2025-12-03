import numpy as np
import tarfile
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import math

import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import Conv2D
from keras.utils import to_categorical
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
import cv2
import requests
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(0)


def main():
    data = [
        "automobile",  # 0
        "bird",  # 1
        "cat",  # 2
        "deer",  # 3
        "dog",  # 4
        "horse",  # 5
        "truck",  # 6
        "cattle",  # 7
        "fox",  # 8
        "baby",  # 9
        "boy",  # 10
        "girl",  # 11
        "man",  # 12
        "woman",  # 13
        "rabbit",  # 14
        "squirrel",  # 15
        "bicycle",  # 16
        "bus",  # 17
        "motorcycle",  # 18
        "pickup_truck",  # 19
        "train",  # 20
        "lawn_mower",  # 21
        "tractor",  # 22
        "trees",  # 23
    ]

    data_df = pd.DataFrame(data)
    print(data_df)
    datagen = create_data_generator()
    num_classes = 24

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
    X_train, Y_train, X_test, Y_test = check_data(X_train, Y_train, X_test, Y_test)
    num_of_each_image = show_training_samples(
        data_df, data, X_train, Y_train, num_classes
    )
    plot_sample_distribution(num_of_each_image, num_classes)
    print("Images per Classes:")

    images_per_class = pd.DataFrame(
        {
            "Class Numbers": range(num_classes),
            "Class Name": data,
            "Image Count": num_of_each_image,
        }
    )
    print(images_per_class)

    examine_typical_image(X_train, Y_train)
    X_train, X_test = apply_preprocessing(X_train, X_test)
    examine_random_image_after_preprocessing(X_train)
    X_train, X_test = reshape_for_cnn(X_train, X_test)
    Y_train, Y_test = one_hot_encode(num_classes, Y_train, Y_test)
    model = build_model(num_classes)
    evaluate_model(model, X_train, Y_train, X_test, Y_test)
    print("Tree, 23")
    url_tree = "https://images.pexels.com/photos/11996445/pexels-photo-11996445.jpeg"
    test_model_with_images(model, url_tree)
    print("Woman, 13")
    url_woman = "https://images.pexels.com/photos/1036623/pexels-photo-1036623.jpeg"
    test_model_with_images(model, url_woman)
    print("Deer, 3")
    url_deer = "https://images.pexels.com/photos/785059/pexels-photo-785059.jpeg"
    test_model_with_images(model, url_deer)
    print("Dog, 4")
    url_dog = "https://images.pexels.com/photos/19962782/pexels-photo-19962782.jpeg"
    test_model_with_images(model, url_dog)
    print("Bicycle, 16")
    url_bicycle = "https://images.pexels.com/photos/276517/pexels-photo-276517.jpeg"
    test_model_with_images(model, url_bicycle)
    print("Truck, 6")
    url_truck = "https://images.pexels.com/photos/3089685/pexels-photo-3089685.jpeg"
    test_model_with_images(model, url_truck)
    print("Bird, 1")
    url_bird = "https://images.pexels.com/photos/1661179/pexels-photo-1661179.jpeg"
    test_model_with_images(model, url_bird)
    print("Motorcycle, 18")
    url_motorbike = "https://images.pexels.com/photos/163210/motorcycles-race-helmets-pilots-163210.jpeg"
    test_model_with_images(model, url_motorbike)
    explore_datagen(datagen, X_train, Y_train)


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
    # now setting a new number to each class
    required_classes = {1: 0, 2: 1, 5: 2, 3: 3, 4: 4, 7: 5, 9: 6}
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
                train_labels_list.append(required_classes[train_labels[i]])

    # Convert lists to the xtrain, ytrain
    X_train_1 = np.array(train_data_list)
    Y_train_1 = np.array(train_labels_list)

    # print("X train data", X_train_1)
    # print("X train shape:", X_train_1.shape)
    # print("Y train shape:", Y_train_1.shape)
    # print("Y train", np.unique(Y_train_1))  # checking the labels

    data = test_data[b"data"]
    labels = test_data[b"labels"]

    for i in range(len(labels)):
        if labels[i] in required_classes:
            test_data_list.append(data[i])
            test_labels_list.append(required_classes[labels[i]])

    # Convert lists to the xtrain, ytrain
    X_test_1 = np.array(test_data_list)
    Y_test_1 = np.array(test_labels_list)

    # print("X test data:", X_test_1)
    # print("X test shape:", X_test_1.shape)
    # print("Y test shape:", Y_test_1.shape)
    # print("Y test", np.unique(Y_test_1))  # checking the labels
    return X_train_1, Y_train_1, X_test_1, Y_test_1


def filter_cifar100_by_class(
    train_dataset_2,
    test_dataset_2,
    cifar100_classes,
):
    required_classes_num = {
        2: 7,
        8: 8,
        11: 9,
        13: 10,
        19: 11,
        34: 12,
        35: 13,
        41: 14,
        46: 15,
        48: 16,
        58: 17,
        65: 18,
        80: 19,
        89: 20,
        90: 21,
        98: 22,
    }
    # trees - change tree class number 17 to 23 the final class number
    required_superclasses_num = {17: 23}

    train_data_list = []
    test_data_list = []

    train_labels_list = []
    test_labels_list = []

    train_data = train_dataset_2[b"data"]
    train_fine_labels = train_dataset_2[b"fine_labels"]
    train_coarse_labels = train_dataset_2[b"coarse_labels"]

    test_data = test_dataset_2[b"data"]
    test_fine_labels = test_dataset_2[b"fine_labels"]
    test_coarse_labels = test_dataset_2[b"coarse_labels"]

    for i in range(len(train_fine_labels)):
        if train_fine_labels[i] in required_classes_num:
            train_data_list.append(train_data[i])
            train_labels_list.append(required_classes_num[train_fine_labels[i]])

    for i in range(len(train_coarse_labels)):
        if train_coarse_labels[i] in required_superclasses_num:
            train_data_list.append(train_data[i])
            train_labels_list.append(required_superclasses_num[train_coarse_labels[i]])

    # Convert Lists to X_train, Y_train
    X_train_2 = np.array(train_data_list)
    Y_train_2 = np.array(train_labels_list)

    for i in range(len(test_fine_labels)):
        if test_fine_labels[i] in required_classes_num:
            test_data_list.append(test_data[i])
            test_labels_list.append(required_classes_num[test_fine_labels[i]])

    for i in range(len(test_coarse_labels)):
        if test_coarse_labels[i] in required_superclasses_num:
            test_data_list.append(test_data[i])
            test_labels_list.append(required_superclasses_num[test_coarse_labels[i]])

    # test_labels_list = test_fine_labels + test_coarse_labels_list

    # Convert Lists to X_test, Y_test
    X_test_2 = np.array(test_data_list)
    Y_test_2 = np.array(test_labels_list)

    # print("X train data", X_train_2)
    # print("X train shape:", X_train_2.shape)
    # print("Y train shape:", Y_train_2.shape)
    # print("Y train", np.unique(Y_train_2))

    # # print("X test data:", X_test_2)
    # print("X test shape:", X_test_2.shape)
    # print("Y test shape:", Y_test_2.shape)
    # print("Y test", np.unique(Y_test_2))

    return X_train_2, Y_train_2, X_test_2, Y_test_2


def combine_datasets(
    X_train_1, Y_train_1, X_test_1, Y_test_1, X_train_2, Y_train_2, X_test_2, Y_test_2
):

    # set the x, y for train and test
    X_train = np.vstack([X_train_1, X_train_2])
    Y_train = np.hstack([Y_train_1, Y_train_2])
    X_test = np.vstack([X_test_1, X_test_2])
    Y_test = np.hstack([Y_test_1, Y_test_2])
    # print("Combine:")
    # print("X train:", X_train.shape)
    # print("Y train:", Y_train.shape)
    # print("X test:", X_test.shape)
    # print("Y test:", Y_test.shape)
    # print("Y train", np.unique(Y_train))
    # print("Y test", np.unique(Y_test))
    return X_train, Y_train, X_test, Y_test


def check_data(X_train, Y_train, X_test, Y_test):
    assert (
        X_train.shape[0] == Y_train.shape[0]
    ), "The number of training images is not equal to the number of labels"
    assert (
        X_test.shape[0] == Y_test.shape[0]
    ), "The number of testing images is not equal to the number of labels"

    # Reshape - as dimensions were not 32 x 32 3 after making changes to the data

    X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
    X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)

    X_train = X_train.transpose(0, 2, 3, 1)
    X_test = X_test.transpose(0, 2, 3, 1)

    # print("After X train reshape:", X_train.shape)
    # print("After X test reshape:", X_test.shape)

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


def show_training_samples(data_df, data, X_train, Y_train, num_classes):
    num_of_samples = []
    cols = 5  # print 5 images from each class
    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 50))
    fig.tight_layout()

    for j in range(num_classes):  # get the row
        X_selected = X_train[Y_train == j]
        count = len(X_selected)
        num_of_samples.append(count)

        for i in range(cols):
            if count == 0:  # deal with empty classes
                for i in range(cols):
                    axs[j][i].axis("off")
            elif count > 0:  # there is images in class
                index = np.random.randint(0, len(X_selected))
                # axs[j][i].imshow(X_selected[index, :, :], cmap=plt.get_cmap("grey"))
                axs[j][i].imshow(X_selected[index, :, :])
                axs[j][i].axis("off")
    plt.show()

    return num_of_samples


def plot_sample_distribution(num_of_each_image, num_classes):
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_each_image)
    plt.title("Distribution of the training set")
    plt.xlabel("Class Type")
    plt.ylabel("Number of Images")
    plt.show()


def examine_typical_image(X_train, Y_train):
    plt.figure()
    plt.imshow(X_train[1000])
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    img = preprocessing(X_train[1000])

    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title("Preprocessed Image")
    plt.axis("off")
    plt.show()

    # pre_img = grayscale(X_train[25000])
    # plt.imshow(pre_img)
    # plt.show()
    # img = preprocessing(X_train[25000])
    # plt.imshow(X_train[25000])
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()
    # print("X train[1000] shape", X_train[1000].shape)
    # print("Y train[1000]", Y_train[1000])
    # print("Image shape:", img.shape)


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = equalize(img)
    img = img / 255
    return img


def apply_preprocessing(X_train, X_test):
    X_train = np.array(list(map(preprocessing, X_train)))
    X_test = np.array(list(map(preprocessing, X_test)))

    return X_train, X_test


def equalize(img):
    img = cv2.equalizeHist(img)
    # checking original vs Equalized image:
    # res = np.hstack((eq, img))
    # plt.figure(figsize=(10, 5))
    # plt.imshow(res, cmap="gray")
    # plt.title("Original vs Equalized Image")
    # plt.axis("off")
    # plt.show()
    return img


def examine_random_image_after_preprocessing(X_train):
    plt.imshow(X_train[np.random.randint(0, len(X_train) - 1)])
    plt.axis("off")
    plt.show()
    # Before: 32 x 32 x 3
    # After: 32 x 32
    # print("X train shape after preprocessing", X_train.shape)


def reshape_for_cnn(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)
    return X_train, X_test


def one_hot_encode(num_classes, Y_train, Y_test):
    Y_train = to_categorical(Y_train, num_classes)
    Y_test = to_categorical(Y_test, num_classes)
    return Y_train, Y_test


def build_model(num_classes):
    model = Sequential()
    model.add(
        Conv2D(64, (3, 3), padding="same", input_shape=(32, 32, 1), activation="relu")
    )
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.7))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    print(model.summary())
    batch_size = 32
    steps_per_epoch = math.ceil(len(X_train) / batch_size)

    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=15,
        validation_data=(X_test, Y_test),
        verbose=1,
        shuffle=True,
    )

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.legend(["training", "validation"])
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.show()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["training", "validation"])
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.show()

    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Test score: ", score[0])
    print("Test accuracy: ", score[1])


def create_data_generator():
    datagen = ImageDataGenerator(
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.3,
        rotation_range=20,
        horizontal_flip=True,
    )
    return datagen


def explore_datagen(datagen, X_train, Y_train):
    datagen.fit(X_train)
    batches = datagen.flow(X_train, Y_train, batch_size=20)
    X_batch, Y_batch = next(batches)
    fig, axs = plt.subplots(1, 15, figsize=(20, 5))
    fig.tight_layout()
    for i in range(15):
        axs[i].imshow(X_batch[i].reshape(32, 32))
        axs[i].axis("off")
    plt.show()


def test_model_with_images(model, url):
    r = requests.get(url, stream=True)
    img = Image.open(r.raw)
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    prediction = model.predict(img, verbose=0)
    predicted_class = np.argmax(prediction)
    print("Predicted class: ", predicted_class)
    # predicted_class = np.argmax(model.predict(img, verbose=0), axis=1)
    # print("Predicted class:" + str(np.argmax(model.predict(img), axis=1)))
    # print("URL: ", url, "Predicted class:", predicted_class)


if __name__ == "__main__":
    main()
