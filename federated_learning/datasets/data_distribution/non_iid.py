import random
import numpy

def calculate_num_shards(num_workers, num_shards_per_client):
    """
    Calculates the number of shards needed.

    :param num_workers: Number of workers
    :type num_workers: int
    :param num_shards_per_client: Number of shards given to each client
    :type num_shards_per_client: int
    """
    return num_workers * num_shards_per_client

def split_data_by_class(X, Y):
    """
    Splits a given set of data into a list of classes of data. Each element represents a single class of the data.

    :param X: data features
    :type X: List
    :param Y: classes
    :type Y: List
    """
    classes = []
    for value in set(Y):
        data_class = []
        for idx in range(len(X)):
            if Y[idx] == value:
                data_class.extend([X[idx]])
        classes.append([data_class, [value for i in range(len(data_class))]])

    return classes

def get_num_classes(data_by_class):
    """
    Returns the number of classes.

    :param data_by_class: Data split into classes
    :type data_by_class: list
    """
    return len(data_by_class)

def calculate_num_shards_per_class(num_classes, num_shards):
    """
    Calculates the number of shards that should be created from each class.

    :param num_classes: Number of classes
    :type num_classes: int
    :param num_shards: Number of shards needed in total
    :type num_shards: int
    """
    return num_shards / num_classes

def calculate_shard_size(class_size, num_shards_per_class):
    """
    Calculates the number of samples to include in a single shard.

    :param class_size: Number of samples for each class
    :type class_size: int
    :param num_shards_per_class: Number of shards per class
    :type num_shards_per_class: int
    """
    return class_size / num_shards_per_class

def split_data_into_shards(dataset, num_shards):
    """
    Split dataset into the specified number of shards.

    NOTE: Assumes each class within the dataset has EQUAL number of samples and each shard is identical in size.

    :param dataset: Dataset
    :type dataset: tuple
    :param num_shards: Number of shards to split the dataset into
    :type num_shards: int
    """
    data_by_class = split_data_by_class(dataset[0], dataset[1])
    num_classes = get_num_classes(data_by_class)
    num_shards_per_class = calculate_num_shards_per_class(num_classes, num_shards)
    shard_size = calculate_shard_size(len(data_by_class[0][0]), num_shards_per_class)

    shards = []
    for class_idx in range(num_classes):
        class_shards = []
        curr_data_idx = 0
        for shard_idx in range(int(num_shards_per_class)):
            class_shards.append((data_by_class[class_idx][0][int(curr_data_idx) : int(curr_data_idx + shard_size)], data_by_class[class_idx][1][int(curr_data_idx) : int(curr_data_idx + shard_size)]))

            curr_data_idx += shard_size

        shards.append([class_shards[0][1][0], class_shards])

    return shards

def convert_class_proportionality_vector_to_num_shards(num_shards_per_client, class_proportionality_vector):
    """
    Converts a class proportionality vector into number of shards.

    :param num_shards_per_client: Number of shards to give to each client
    :type num_shards_per_client: int
    :param class_proportionality_vector: Class distribution
    :type class_proportionality_vector: list(float)
    """
    return [int(num_shards_per_client * a) for a in class_proportionality_vector]

def distribute_batches_non_iid(logger, dataset, num_workers, num_shards_per_client, class_proportionality_vector):
    """
    Distributes data in a non-iid setting.

    :param logger: Logger
    :type logger: loguru.logger
    :param dataset: Dataset
    :type dataset: tuple
    :param num_workers: Number of workers
    :type num_workers: int
    :param num_shards_per_client: Number of shards given to each client
    :type num_shards_per_client: int
    :param class_proportionality_vector: Proportion of class distribution
    :type class_proportionality_vector: list(float)
    """
    num_shards = calculate_num_shards(num_workers, num_shards_per_client)
    shards = split_data_into_shards(dataset, num_shards)
    distributed_dataset = [[[], []] for i in range(num_workers)]

    for i in range(num_workers):
        shards_class_vector = convert_class_proportionality_vector_to_num_shards(num_shards_per_client, class_proportionality_vector)

        shard_classes_used = []
        for num_shards_in_class in shards_class_vector:
            shard_classes = list(set([i for i in range(len(shards))]) - set(shard_classes_used))

            try:
                shard_class = shard_classes.pop(random.randint(0, len(shard_classes) - 1))
                while len(shards[shard_class][1]) < num_shards_in_class:
                    shard_class = shard_classes.pop(random.randint(0, len(shard_classes) - 1))
            except:
                logger.warning("Could not distribute data correctly")

                return distribute_batches_non_iid(logger, dataset, num_workers, num_shards_per_client, class_proportionality_vector)

            for shard_idx in range(num_shards_in_class):
                new_shard = shards[shard_class][1].pop(0)
                distributed_dataset[i][0].extend(new_shard[0])
                distributed_dataset[i][1].extend(new_shard[1])

            shard_classes_used.append(shard_class)

            logger.info("Worker {} given {} shards from class {}", i, num_shards_in_class, shard_class)
            logger.debug("{} shards remaining in class {}", len(shards[shard_class][1]), shard_class)

    distributed_dataset = [(numpy.array(worker_data[0]), numpy.array(worker_data[1])) for worker_data in distributed_dataset]

    for shard_class in range(len(shards)):
        logger.debug("{} shards remaining in class {}", len(shards[shard_class][1]), shard_class)

    return distributed_dataset
