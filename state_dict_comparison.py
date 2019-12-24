import os
from loguru import logger
import numpy
import json

STATE_DICTS_TO_COMBINE_PATH = ["/Users/valetolpegin/Downloads/1734845290075935", "/Users/valetolpegin/Downloads/4958422905188338"]
STATE_DICT_COMBINED_PATH = "1099371747173474"

def average_nn_parameters(parameters):
    """
    Averages passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    new_params = {}
    for name in parameters[0].keys():
        new_params[name] = sum([param[name] for param in parameters])

    return new_params

def combine_models(model_1, model_2):
    return average_nn_parameters([model_1, model_2])

def scale_model(model, num_models):
    new_params = {}
    for name in model.keys():
        new_params[name] = model[name] / num_models

    return new_params

def deserialize_state_dict(state_dict):
    """
    :param state_dict: dictionary
    """
    return {key : numpy.asarray(item, dtype=numpy.float32) for key, item in state_dict.items()}

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

def compare_state_dicts(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
        if numpy.all(key_item_1[1] == key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                logger.error('Mismatch found at {}', key_item_1[0])
                logger.debug("Model 1 value: {}", str(key_item_1[1]))
                logger.debug("Model 2 value: {}", str(key_item_2[1]))
            else:
                raise Exception
    if models_differ == 0:
        logger.info('Models match perfectly!')

def load_state_dict(file_path):
    logger.info("Loading state dict: {}", file_path)

    with open(file_path, "r") as f:
        return deserialize_state_dict(DiskModelStorage.load_model(f))

if __name__ == '__main__':
    combined_state_dict = load_state_dict(STATE_DICT_COMBINED_PATH)

    combined_model = load_state_dict(STATE_DICTS_TO_COMBINE_PATH[0])
    for state_dict_to_combine in STATE_DICTS_TO_COMBINE_PATH[1:]:
        state_dict = load_state_dict(state_dict_to_combine)

        combined_model = combine_models(combined_model, state_dict)

    scaled_model = scale_model(combined_model, len(STATE_DICTS_TO_COMBINE_PATH))

    logger.info("Comparing state dict...")
    compare_state_dicts(combined_state_dict, scaled_model)
