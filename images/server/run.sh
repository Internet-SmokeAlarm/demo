#!/bin/bash

# run.sh
#
# Script to run the Verge AI demo server.
#
# Usage:
# ./run.sh <Your API Key>

API_KEY=$1

docker run -e API_KEY=$API_KEY -it vergeai-server
