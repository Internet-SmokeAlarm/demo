import requests
from loguru import logger
import json
import torch
import numpy

from fedlearn import FedLearnApi
from fedlearn.models import RoundConfiguration
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from trainer import TrainingClient
from federated_learning.arguments import Arguments
from serde import serialize_state_dict
from serde import deserialize_state_dict

def handle_add_client(input_str):
    elements = input_str.split(" ")

    ip_address = elements[1]
    port = elements[2]

    return (ip_address, port)

def register_client(api, group_id, device_info):
    device = api.register_device(group_id)

    url = "http://" + str(device_info[0]) + ":" + str(device_info[1]) + "/register"
    data = {
        "device_id" : device.get_id(),
        "device_api_key" : device.get_api_key(),
        "group_id" : group_id
    }

    response = requests.post(url, json=data)

    logger.info("Register client: {} with response: {}".format(str(device_info), str(response.status_code)))

if __name__ == '__main__':
    api = FedLearnApi("API_KEY_NOT_NECESSARY_RIGHT_NOW")
    group = api.create_group("demo_test")
    round = None

    args = Arguments(logger)
    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)

    train_client = TrainingClient(args, 0, train_data_loader, test_data_loader)
    parameters = serialize_state_dict(train_client.get_nn_parameters())
    api.submit_initial_group_model(parameters, group.get_id())

    while True:
        user_input = input("> ")

        if "add_client" in user_input:
            device_info = handle_add_client(user_input)

            register_client(api, group.get_id(), device_info)
        elif "start_round" in user_input:
            round = api.start_round(group.get_id(), RoundConfiguration("2", "RANDOM"))

            logger.info("Round ID: {}".format(round.get_id()))
        elif "round_state" in user_input:
            round = api.get_round_state(group.get_id(), round.get_id())

            logger.info("Round ID: {}".format(round.get_id()))
            logger.info("State: {}".format(round.get_status().value))
        elif "test" in user_input:
            model_params = deserialize_state_dict(api.get_round_aggregate_model(group.get_id(), round.get_id()))

            train_client.update_nn_parameters(model_params)

            train_client.test()
        elif "quit" in user_input:
            api.delete_group(group.get_id())

            exit(0)
