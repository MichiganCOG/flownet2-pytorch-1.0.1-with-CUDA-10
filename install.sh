#!/bin/bash
set -e

SCRIPT_DIR="$(cd $(dirname $0); pwd)"

cd "$SCRIPT_DIR/src/networks/correlation_package"
rm -rf *_cuda.egg-info build dist __pycache__
python setup.py install --user

cd "$SCRIPT_DIR/src/networks/resample2d_package"
rm -rf *_cuda.egg-info build dist __pycache__
python setup.py install --user

cd "$SCRIPT_DIR/src/networks/channelnorm_package"
rm -rf *_cuda.egg-info build dist __pycache__
python setup.py install --user
