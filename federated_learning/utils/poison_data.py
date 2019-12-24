from .identify_random_elements import identify_random_elements
from .label_replacement import apply_class_label_replacement
from .client_utils import log_client_data_statistics

def poison_data(logger, distributed_dataset, num_workers, num_poisoned_workers, replacement_method):
    """
    Poison worker data

    :param logger: logger
    :type logger: loguru.logger
    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    :param num_workers: Number of workers overall
    :type num_workers: int
    :param num_poisoned_workers: Number of poisoned workers
    :type num_poisoned_workers: int
    :param replacement_method: Replacement methods to use to replace
    :type replacement_method: list(method)
    """
    # TODO: Add support for multiple replacement methods?
    poisoned_dataset = []

    poisoned_worker_ids = identify_random_elements(num_workers, num_poisoned_workers)
    class_labels = list(set(distributed_dataset[0][1]))

    logger.info("Poisoning data for workers: {}".format(str(poisoned_worker_ids)))

    for worker_idx in range(num_workers):
        if worker_idx in poisoned_worker_ids:
            poisoned_dataset.append(apply_class_label_replacement(distributed_dataset[worker_idx][0], distributed_dataset[worker_idx][1], replacement_method))
        else:
            poisoned_dataset.append(distributed_dataset[worker_idx])

    log_client_data_statistics(logger, class_labels, poisoned_dataset)

    return poisoned_dataset
