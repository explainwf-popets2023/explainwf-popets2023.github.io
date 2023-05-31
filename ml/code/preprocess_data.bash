#!/usr/bin/env bash

set -eoux pipefail

SCRIPT=./extract.py

mkdir -p $CLEAN_DIR/section3

parallel python3 -u $SCRIPT -n 200 -r -c {} $TBB_URLS $CLEAN_DIR/section3/{/.}.pkl ::: $FIDELITY_DIR/*.log

mkdir -p $CLEAN_DIR/section45

parallel -j 3 -k --lb python3 -u $SCRIPT -f -n 300 --nkeep_port_80 30000 -c {} $TBB_URLS $CLEAN_DIR/section45/{/.}.pkl ::: $ROBUST_DIR/*.log

SCRIPT=./make_labels.py

mkdir -p $LABEL_DIR

python3 $SCRIPT $LABEL_DIR/section3_labels.pkl $CLEAN_DIR/section3/*.pkl
python3 $SCRIPT $LABEL_DIR/section45_labels_closed.pkl $CLEAN_DIR/section45/*.pkl
python3 $SCRIPT -n 5 $LABEL_DIR/section45_labels_open.pkl $CLEAN_DIR/section45/*.pkl

SCRIPT=./split_train_test.py

mkdir -p $SPLIT_DIR/{section3,section45_closed,section45_open}

parallel python3 $SCRIPT {} $LABEL_DIR/section3_labels.pkl $SPLIT_DIR/section3/{/} ::: $CLEAN_DIR/section3/*.pkl
parallel python3 $SCRIPT {} $LABEL_DIR/section45_labels_closed.pkl $SPLIT_DIR/section45_closed/{/} ::: $CLEAN_DIR/section45/*.pkl
parallel python3 $SCRIPT {} $LABEL_DIR/section45_labels_open.pkl $SPLIT_DIR/section45_open/{/} ::: $CLEAN_DIR/section45/*.pkl
