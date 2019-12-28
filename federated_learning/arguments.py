from .nets import VGGLite
from .nets import FashionMNISTCNN
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
        self.lr = 0.001
        self.momentum = 0.9
        self.cuda = True
        self.shuffle = False
        self.log_interval = 100
        self.kwargs = {}

        self.scheduler_step_size = 10
        self.scheduler_gamma = 0.1
        self.min_lr = 1e-10

        #self.net = VGGLite
        self.net = FashionMNISTCNN

        self.train_data_loader_pickle_path = "/Users/valetolpegin/Desktop/data_loaders/fashionmnist/data_dist_a/train_loader.pickle"
        self.test_data_loader_pickle_path = "/Users/valetolpegin/Desktop/data_loaders/fashionmnist/data_dist_a/test_loader.pickle"

        self.loss_function = torch.nn.CrossEntropyLoss

        self.default_model_folder_path = "/Users/valetolpegin/Desktop/default_models"

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

    def get_default_model_folder_path(self):
        return self.default_model_folder_path

    def get_logger(self):
        return self.logger

    def get_loss_function(self):
        return self.loss_function

    def get_net(self):
        return self.net

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
