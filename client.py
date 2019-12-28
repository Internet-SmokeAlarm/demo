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
from fedlearn.serde.pytorch import PyTorchSerializer
from fedlearn.serde.pytorch import PyTorchDeserializer

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

def execute_federated_learning(device_info):
    device_id = device_info["device_id"]
    device_api_key = device_info["device_api_key"]
    group_id = device_info["group_id"]

    args = Arguments(logger)
    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)

    train_client = TrainingClient(args, str(device_id), train_data_loader, test_data_loader)

    api_client = FedLearnApi(device_api_key)

    while True:
        while not api_client.is_device_active(group_id, device_id):
            logger.info("Waiting for device to activate...")
            sleep(5)

        logger.info("Device active")

        current_round_id = api_client.get_group_current_round_id(group_id)
        logger.info("Current Round ID: {}".format(current_round_id))

        starting_params = api_client.get_round_start_model(group_id, current_round_id)

        logger.info("Updating parameters with new starting parameters")
        train_client.set_state_dict(PyTorchDeserializer().deserialize(starting_params))

        logger.info("Training client")
        train_client.train()
        train_client.test()

        logger.info("Uploading new parameters to the cloud")
        parameters = PyTorchSerializer().serialize(train_client.get_state_dict())
        api_client.submit_model_update(parameters, group_id, current_round_id, device_id)

def is_registration_complete(json_data):
    return len(list(json_data.keys())) != 0

def wait_for_registration(device_info):
    while not is_registration_complete(device_info):
        logger.info("Waiting for registration information...")
        sleep(1)

def run_client(device_info):
    wait_for_registration(device_info)

    execute_federated_learning(device_info)

if __name__ == '__main__':
    port = load_arguments()

    client_proc = Process(target=run_client, args=(device_info,))
    client_proc.start()

    app.run(host="0.0.0.0", port=port)

    client_proc.join()
