import argparse
from flask import Flask
from flask import request
import multiprocessing
from multiprocessing import Process
from time import sleep
from multiprocessing import Manager
from loguru import logger

from fedlearn import FedLearnApi
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from trainer import TrainingClient
from federated_learning.arguments import Arguments
from serde import serialize_state_dict
from serde import deserialize_state_dict

app = Flask(__name__)

manager = Manager()
device_info = manager.dict()

@app.route("/register", methods=["POST"])
def register():
    global device_info

    req_json = request.get_json()
    device_info["device_id"] = req_json["device_id"]
    device_info["device_api_key"] = req_json["device_api_key"]
    device_info["group_id"] = req_json["group_id"]

    return {"success" : True}

def load_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("port", help="port to run client HTTP server on")

    args = parser.parse_args()
    port = int(args.port)

    return port

def is_registration_complete(json_data):
    return "device_id" in json_data and "device_api_key" in json_data

def handle_machine_learning(device_info):
    device_id = device_info["device_id"]
    device_api_key = device_info["device_api_key"]
    group_id = device_info["group_id"]

    args = Arguments(logger)
    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)

    train_client = TrainingClient(args, str(device_id), train_data_loader, test_data_loader)

    api_client = FedLearnApi(device_api_key)

    round_num = 1
    round_last_executed = ""

    while True:
        while not api_client.is_device_active(group_id, device_id):
            print("Waiting for device to activate...")
            sleep(5)

        print("Device active")

        current_round_id_info = api_client.get_group_current_round_id(group_id)
        print("Current Round ID: {}".format(current_round_id_info.get_id()))

        if current_round_id_info.get_id() != round_last_executed:
            current_round = api_client.get_round_state(group_id, current_round_id_info.get_id())

            if current_round.get_previous_round_id() == "N/A":
                starting_params = api_client.get_initial_group_model(group_id)
            else:
                starting_params = api_client.get_round_aggregate_model(group_id, current_round.get_previous_round_id())

            print("Updating parameters with new starting parameters")
            train_client.update_nn_parameters(deserialize_state_dict(starting_params))

            print("Training client")
            train_client.train(round_num)
            train_client.test()

            print("Uploading new parameters to the cloud")
            parameters = serialize_state_dict(train_client.get_nn_parameters())
            api_client.submit_model_update(parameters, group_id, current_round_id_info.get_id(), device_id)

            round_num = round_num + 1
            round_last_executed = current_round_id_info.get_id()

def run_client(device_info):
    while not is_registration_complete(device_info):
        print("Waiting for registration information...")
        sleep(1)

    handle_machine_learning(device_info)

if __name__ == '__main__':
    port = load_arguments()

    client_proc = Process(target=run_client, args=(device_info,))
    client_proc.start()

    app.run(host="0.0.0.0", port=port)

    client_proc.join()
