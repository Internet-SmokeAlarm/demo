from .datasets import MNISTDataset
from .datasets import CIFAR10Dataset
from .datasets import CIFAR100Dataset
from .datasets import FashionMNISTDataset
from .nets import SimpleNet
from .nets import SmallNet
from .nets import VGGLite
from .nets import VGGLiteBinaryPoisoning
from .nets import VGGLiteLinearRegressionPoisoning
from .nets import VGG11
from .nets import VGG16
from .nets import FashionMNISTCNN
from .nets import FashionMNISTCNNLinearRegression
import torch.nn.functional as F
import torch

# Setting the seed for Torch
SEED = 1
torch.manual_seed(SEED)

class Arguments:

    def __init__(self, logger):
        self.logger = logger

        self.batch_size = 4
        self.test_batch_size = 1000
        self.epochs = 50
        self.lr = 0.001
        self.momentum = 0.9
        self.cuda = True
        self.shuffle = False
        self.log_interval = 100
        self.kwargs = {}

        self.scheduler_step_size = 10
        self.scheduler_gamma = 0.1
        self.min_lr = 1e-10

        self.use_random_worker_subset = False
        self.num_random_workers = 25
        self.aggregate_random_worker_subset_only = True

        self.save_model = True
        self.save_epoch_interval = 1
        self.save_model_path = "models"
        self.epoch_save_start_suffix = "start"
        self.epoch_save_end_suffix = "end"

        self.num_workers = 50
        self.num_poisoned_workers = 25

        self.num_shards_per_client = 10
        self.class_proportionality_vector = [0.5, 0.5]

        #self.dataset = MNISTDataset(logger)
        #self.dataset = CIFAR10Dataset(logger)
        #self.dataset = CIFAR100Dataset(logger)
        self.dataset = FashionMNISTDataset(logger)

        #self.net = SimpleNet
        #self.net = SmallNet
        #self.net = VGGLite
        #self.net = VGGLiteBinaryPoisoning
        #self.net = VGGLiteLinearRegressionPoisoning
        #self.net = VGG11
        #self.net = VGG16
        self.net = FashionMNISTCNN
        #self.net = FashionMNISTCNNLinearRegression

        self.train_data_loader_pickle_path = "/Users/valetolpegin/Desktop/fashion_mnist_train_data_loader.pickle"
        self.test_data_loader_pickle_path = "/Users/valetolpegin/Desktop/fashion_mnist_test_data_loader.pickle"

        #self.loss_function = torch.nn.MSELoss
        self.loss_function = torch.nn.CrossEntropyLoss

        self.is_classification_problem = True

        self.default_model_folder_path = "/Users/valetolpegin/Desktop/default_models"

        self.experiment_results_save_path = "results.csv"

    def get_epoch_save_start_suffix(self):
        return self.epoch_save_start_suffix

    def get_epoch_save_end_suffix(self):
        return self.epoch_save_end_suffix

    def get_is_classification_problem(self):
        return self.is_classification_problem

    def set_train_data_loader_pickle_path(self, path):
        self.train_data_loader_pickle_path = path

    def get_train_data_loader_pickle_path(self):
        return self.train_data_loader_pickle_path

    def set_test_data_loader_pickle_path(self, path):
        self.test_data_loader_pickle_path = path

    def get_test_data_loader_pickle_path(self):
        return self.test_data_loader_pickle_path

    def get_cuda(self):
        return self.cuda

    def get_scheduler_step_size(self):
        return self.scheduler_step_size

    def get_scheduler_gamma(self):
        return self.scheduler_gamma

    def get_min_lr(self):
        return self.min_lr

    def get_class_proportionality_vector(self):
        return self.class_proportionality_vector

    def get_num_shards_per_client(self):
        return self.num_shards_per_client

    def set_experiment_results_save_path(self, experiment_results_save_path):
        self.experiment_results_save_path = experiment_results_save_path

    def get_experiment_results_save_path(self):
        return self.experiment_results_save_path

    def get_default_model_folder_path(self):
        return self.default_model_folder_path

    def get_num_epochs(self):
        return self.epochs

    def set_num_poisoned_workers(self, num_poisoned_workers):
        self.num_poisoned_workers = num_poisoned_workers

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers

    def set_model_save_path(self, save_model_path):
        self.save_model_path = save_model_path

    def get_dataset(self):
        return self.dataset

    def get_logger(self):
        return self.logger

    def get_loss_function(self):
        return self.loss_function

    def get_net(self):
        return self.net

    def get_num_workers(self):
        return self.num_workers

    def get_num_poisoned_workers(self):
        return self.num_poisoned_workers

    def get_learning_rate(self):
        return self.lr

    def get_momentum(self):
        return self.momentum

    def get_shuffle(self):
        return self.shuffle

    def get_batch_size(self):
        return self.batch_size

    def get_test_batch_size(self):
        return self.test_batch_size

    def get_log_interval(self):
        return self.log_interval

    def get_save_model_folder_path(self):
        return self.save_model_path

    def get_use_random_worker_subset(self):
        return self.use_random_worker_subset

    def get_num_random_workers(self):
        return self.num_random_workers

    def get_aggregate_random_worker_subset_only(self):
        return self.aggregate_random_worker_subset_only

    def should_save_model(self, epoch_idx):
        """
        Returns true/false models should be saved.

        :param epoch_idx: current training epoch index
        :type epoch_idx: int
        """
        if not self.save_model:
            return False

        if epoch_idx == 1 or epoch_idx % self.save_epoch_interval == 0:
            return True

    def log(self):
        """
        Log this arguments object to the logger.
        """
        self.logger.debug("Arguments: {}", str(self))

    def __str__(self):
        return "[Batch Size: " + str(self.batch_size) + " Test Batch Size: " + str(self.test_batch_size) + " Epochs: " + str(self.epochs) + " LR: " + str(self.lr) + " Momentum: " + str(self.momentum) + " CUDA: " + str(self.cuda) + " Shuffle: " + str(self.shuffle) + " Log Interval: " + str(self.log_interval) + " Use Random Worker Subset: " + str(self.use_random_worker_subset) + " Number random workers: " + str(self.num_random_workers) + " Save Model: " + str(self.save_model) + " Save Epoch Interval: " + str(self.save_epoch_interval) + " Save Model Path: " + str(self.save_model_path) + " Num Workers: " + str(self.num_workers) + " Num Poisoned Workers: " + str(self.num_poisoned_workers) + " Scheduler Step Size: " + str(self.scheduler_step_size) + " Min LR: " + str(self.min_lr) + " Scheduler Gamma: " + str(self.scheduler_gamma) + "]"
