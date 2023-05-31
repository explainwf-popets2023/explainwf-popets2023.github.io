#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
import pickle
import sys

import common
import classifiers
import json

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.utils import to_categorical


def main(args):
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logging.info(f"Program arguments: {args}")

    _, X_test, _, y_test = pickle.load(args.examples_file)
    args.examples_file.close()

    for model_filepath in args.model_filepaths:
        logging.info("BEGIN model load")
        model = classifiers.read_classifier_from_tarfile_gzipped(model_filepath)
        logging.info("END model load")

        logging.info(f"Model Features: {model.metadata}")

        features = model.metadata["input"]

        if features == "svm":
            logging.info("CUMUL classifier")
            X = common.convert_records_to_cumul_repr(X_test)
        elif features == "kfp":
            logging.info("k-fingerprinting Classifier")
            parallel = True
            X = common.convert_records_to_kfp(X_test, parallel)
        elif features == "nn":
            logging.info("DF classifier")
            X = common.convert_records_to_nn_repr(X_test)
        elif features == "tt":
            logging.info("Tik-Tok classifier")
            X = common.convert_records_to_tiktok_repr(X_test)
        else:
            assert False

        logging.info("BEGIN model predict")
        y_pred = model.predict(X)
        logging.info("END model predict")

        binary = len(set(y_test)) == 2

        stats = classifiers.compute_stats(y_test, y_pred, binary=binary)
        stats["classifier"] = features
        stats["training_set"] = model.metadata["training"]
        stats["training_stem"] = Path(model.metadata["training"]).stem
        stats["confusion"] = stats["confusion"].tolist()
        stats["test_set"] = args.examples_file.name
        stats["test_stem"] = Path(args.examples_file.name).stem

        if args.log_stats:
            logging.info(stats)

        json.dump(stats, args.output_file)
        args.output_file.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("examples_file", type=argparse.FileType("rb"))
    parser.add_argument("output_file", type=argparse.FileType("w"))
    parser.add_argument("model_filepaths", nargs="*")
    parser.add_argument("-l", "--log_stats", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
