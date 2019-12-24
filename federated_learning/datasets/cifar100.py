from .dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

class CIFAR100Dataset(Dataset):

	CIFAR100_DATA_PATH = '~/Desktop/test2/data'

	def __init__(self, logger):
		super(CIFAR100Dataset, self).__init__(logger)

	def load_train_dataset(self):
		self.get_logger().debug("Loading CIFAR100 train data")

		CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
		CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
		transform = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(15),
			transforms.ToTensor(),
			transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
		])
		train_dataset = datasets.CIFAR100(root=CIFAR100Dataset.CIFAR100_DATA_PATH, train=True, download=True, transform=transform)
		train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

		train_data = self.get_tuple_from_data_loader(train_loader)

		self.get_logger().debug("Finished loading CIFAR100 train data")

		return train_data

	def load_test_dataset(self):
		self.get_logger().debug("Loading CIFAR100 test data")

		CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
		CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
		transform = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(15),
			transforms.ToTensor(),
			transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
		])
		test_dataset = datasets.CIFAR100(root=CIFAR100Dataset.CIFAR100_DATA_PATH, train=False, download=True, transform=transform)
		test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

		test_data = self.get_tuple_from_data_loader(test_loader)

		self.get_logger().debug("Finished loading CIFAR100 test data")

		return test_data
