#!/bin/bash

# build.sh
#
# Script to build the Verge AI demo client.
#
# Usage:
# ./build.sh

cp -r ../../../vergeai-python .
cp -r ../../../pytorch-serde .

docker build -t vergeai-server .

rm -rf vergeai-python
rm -rf pytorch-serde
