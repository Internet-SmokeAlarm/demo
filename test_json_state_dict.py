from loguru import logger
import json
import numpy

from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from trainer import TrainingClient
from federated_learning.arguments import Arguments

class DiskModelStorage:

    @staticmethod
    def store_model(file_obj, state_dict):
        """
        :param file_obj: file-like object
        :param state_dict: dictionary (Must be json serializable)
        """
        json.dump(state_dict, file_obj)

    @staticmethod
    def load_model(file_obj):
        """
        :param file_obj: file-like object
        :param state_dict: dictionary (Must be json serializable)
        """
        return json.load(file_obj)

def deserialize_state_dict(state_dict):
    """
    :param state_dict: dictionary
    """
    return {key : numpy.asarray(item, dtype=numpy.float32) for key, item in state_dict.items()}

def load_state_dict(file_path):
    logger.info("Loading state dict: {}", file_path)

    with open(file_path, "r") as f:
        return deserialize_state_dict(DiskModelStorage.load_model(f))

if __name__ == '__main__':
    model_param_file_path = "1099371747173474"
    loaded_state_dict = load_state_dict(model_param_file_path)

    args = Arguments(logger)
    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)

    train_client = TrainingClient(args, 0, train_data_loader, test_data_loader)
    train_client.update_nn_parameters(loaded_state_dict)

    train_client.test()
