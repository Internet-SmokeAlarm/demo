from .dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

class MNISTDataset(Dataset):

    MNIST_DATA_PATH = '~/Desktop/test2/data'

    def __init__(self, logger):
        super(MNISTDataset, self).__init__(logger)

    def load_train_dataset(self):
        self.get_logger().debug("Loading MNIST train data")

        train_dataset = datasets.MNIST(MNISTDataset.MNIST_DATA_PATH, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        train_data = self.get_tuple_from_data_loader(train_loader)

        self.get_logger().debug("Finished loading MNIST train data")

        return train_data

    def load_test_dataset(self):
        self.get_logger().debug("Loading MNIST test data")

        test_dataset = datasets.MNIST(MNISTDataset.MNIST_DATA_PATH, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_logger().debug("Finished loading MNIST test data")

        return test_data
