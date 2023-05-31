#!/usr/bin/env bash

set -eoux pipefail

patch -i ./RF_fextract.py.patch
patch -i ./build_dnn.py.patch && mv Model_NoDef.py build_dnn.py
