import requests
from fedlearn import FedLearnApi

API_KEY = "NOT_NECESSARY_RIGHT_NOW"

def handle_add_client(input_str):
    elements = input_str.split(" ")

    ip_address = elements[1]
    port = elements[2]

    return (ip_address, port)

def register_client(api, group_id, device_info):
    device = api.register_device(group_id)

    url = "http://" + str(device_info[0]) + ":" + str(device_info[1]) + "/register"
    data = {"device_id" : device.get_id(), "device_api_key" : device.get_api_key()}

    response = requests.post(url, json=data)

    print("Register client: {} with response: {}".format(str(device_info), str(response.status_code)))

if __name__ == '__main__':
    api = FedLearnApi(API_KEY)
    group = api.create_group("demo_test")
    round = None

    while True:
        user_input = input("> ")

        if "add_client" in user_input:
            device_info = handle_add_client(user_input)

            register_client(api, group.get_id(), device_info)
        elif "start_round" in user_input:
            round = api.start_round(group.get_id())

            print("Round ID: {}".format(round.get_id()))
        elif "round_state" in user_input:
            round = api.get_round_status(group.get_id(), round.get_id())

            print("Round ID: {}".format(round.get_id()))
            print("Models: {}".format(str(round.get_models())))
        elif "quit" in user_input:
            api.delete_group(group.get_id())

            exit(0)
