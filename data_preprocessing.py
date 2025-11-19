import tarfile

# Read CIFAR-10 File
with tarfile.open("Datasets/cifar-10-python.tar.gz", "r") as tar:
    print("Opened file")
    tar.extractall(path="./dataset1_classes")
    print("All files extracted from dataset 1")

# Read the CIFAR-100 File
with tarfile.open("Datasets/cifar-100-python.tar.gz", "r") as tar:
    print("Opened dataset 2")
    tar.extractall(path="./dataset2_classes")
    print("All files extracted from dataset 2")
