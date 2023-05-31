#!/usr/bin/env bash

export TBB_URLS=/mnt/bind/data/urls.txt

export FIDELITY_DIR=/mnt/bind/data/section3
export ROBUST_DIR=/mnt/bind/data/section4and5

export OUTPUT_DIR=./out
export LABEL_DIR=$OUTPUT_DIR/labels
export CLEAN_DIR=$OUTPUT_DIR/cleaned
export SPLIT_DIR=$OUTPUT_DIR/split
export MODEL_DIR=$OUTPUT_DIR/models

mkdir -p $RESULT_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $CLEAN_DIR
mkdir -p $SPLIT_DIR
mkdir -p $MODEL_DIR
