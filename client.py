import argparse
from flask import Flask
from flask import request
import multiprocessing
from multiprocessing import Process
from time import sleep
from multiprocessing import Manager

app = Flask(__name__)

manager = Manager()
device_info = manager.dict()

@app.route("/register", methods=["POST"])
def register():
    global device_info

    req_json = request.get_json()
    device_info["device_id"] = req_json["device_id"]
    device_info["device_api_key"] = req_json["device_api_key"]

    return {"success" : True}

def load_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("port", help="port to run client HTTP server on")

    args = parser.parse_args()
    port = int(args.port)

    return port

def is_registration_complete(json_data):
    return "device_id" in json_data and "device_api_key" in json_data

def run_client(device_info):
    while True:
        if is_registration_complete(device_info):
            # todo : Execute machine learning here

            print(device_info)
            sleep(1)
        else:
            print("Waiting for registration information...")

            sleep(1)

if __name__ == '__main__':
    port = load_arguments()

    client_proc = Process(target=run_client, args=(device_info,))
    client_proc.start()

    app.run(host="0.0.0.0", port=port)

    client_proc.join()
