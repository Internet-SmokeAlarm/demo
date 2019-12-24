import torch
import numpy

def serialize_state_dict(state_dict):
    """
    :param state_dict: dict
    """
    return {key : item.detach().numpy().tolist() for key, item in state_dict.items()}

def deserialize_state_dict(state_dict_json):
    """
    :param state_dict_json: dict
    """
    return {key : torch.tensor(numpy.asarray(item)) for key, item in state_dict_json.items()}
