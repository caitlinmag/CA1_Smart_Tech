import tarfile

# Read CIFAR-10 File
with tarfile.open("Datasets/cifar-10-python.tar.gz", "r") as tar:
    print("Opened file")
    tar.extractall(path="cifar-10-python.tar.gz")
    print(tar.getmembers())

# Read CIFAR-100 File
with tarfile.open("Datasets/cifar-100-python.tar.gz", "r") as tar2:
    print("Opened file")
