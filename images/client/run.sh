#!/bin/bash

# run.sh
#
# Script to run the Verge AI demo client.
#
# Usage:
# ./run.sh <Client ID> <API Key>

CLIENT_ID=$1
API_KEY=$2

docker run -e CLIENT_ID=$CLIENT_ID -e API_KEY=$API_KEY -it vergeai-client
