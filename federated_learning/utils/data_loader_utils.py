import numpy
from .label_replacement import apply_class_label_replacement
import os
import pickle
import random

def generate_data_loaders_from_distributed_dataset(distributed_dataset, dataset, batch_size):
    """
    Generate data loaders from a distributed dataset.

    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    :param dataset: Dataset
    :type dataset: Dataset
    :param batch_size: batch size for data loader
    :type batch_size: int
    """
    data_loaders = []
    for worker_training_data in distributed_dataset:
        data_loaders.append(dataset.get_data_loader_from_data(batch_size, worker_training_data[0], worker_training_data[1], shuffle=True))

    return data_loaders

def load_train_data_loader(logger, args):
    """
    Loads the training data DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param args: Arguments
    """
    if os.path.exists(args.get_train_data_loader_pickle_path()):
        return load_data_loader_from_file(logger, args.get_train_data_loader_pickle_path())
    else:
        logger.warning("Couldn't find train data loader stored in file. Generating data loader.")

        return generate_train_loader(args)

def generate_train_loader(args):
    train_dataset = args.get_dataset().get_train_dataset()
    X, Y = shuffle_data(args, train_dataset)

    return args.get_dataset().get_data_loader_from_data(args.get_batch_size(), X, Y)

def load_test_data_loader(logger, args):
    """
    Loads the test data DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param args: Arguments
    """
    if os.path.exists(args.get_test_data_loader_pickle_path()):
        return load_data_loader_from_file(logger, args.get_test_data_loader_pickle_path())
    else:
        logger.warning("Couldn't find test data loader stored in file. Generating data loader.")

        return generate_test_loader(args)

def load_data_loader_from_file(logger, filename):
    """
    Loads DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param filename: string
    """
    logger.info("Loading data loader from file: {}".format(filename))

    with open(filename, "rb") as f:
        return load_saved_data_loader(f)

def generate_test_loader(args):
    test_dataset = args.get_dataset().get_test_dataset()
    X, Y = shuffle_data(args, test_dataset)

    return args.get_dataset().get_data_loader_from_data(args.get_test_batch_size(), X, Y)

def shuffle_data(args, dataset):
    data = list(zip(dataset[0], dataset[1]))
    random.shuffle(data)
    X, Y = zip(*data)
    X = numpy.asarray(X)
    Y = numpy.asarray(Y)

    return X, Y

def load_saved_data_loader(file_obj):
    return pickle.load(file_obj)

def save_data_loader_to_file(data_loader, file_obj):
    pickle.dump(data_loader, file_obj)
