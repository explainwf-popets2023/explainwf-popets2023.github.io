#!/usr/bin/env python3

import os
NGPUS = 4

try:
    GPU = int(os.environ["JOB"]) % NGPUS
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU}"
except:
    pass

import argparse
import logging
from pathlib import Path
import pickle
import sys

import common
import classifiers

import numpy as np
from tensorflow.keras.utils import to_categorical


def log_training_accuracy(model, X_train, y_train):
    y_pred = model.predict(X_train)
    stats = classifiers.compute_stats(y_train, y_pred)
    logging.info(f"Training perf: {stats}")


def save_model(model_dirpath, model_desc, model):
    output_path = Path(model_dirpath, f"{model_desc}.tar.gz")
    output_path.parent.mkdir(exist_ok=True)
    classifiers.write_classifier_into_tarfile_gzipped(model, output_path)


def main(args):
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logging.info(f"Program arguments: {args}")

    X_train, _, y_train, _ = pickle.load(args.examples_file)
    args.examples_file.close()

    separate_cv = False
    X_val = None
    y_val = None

    if args.cv_file is not None:
        separate_cv = True
        logging.info(f"Using separate examples from {args.cv_file.name} for validation")
        X_val, _, y_val, _ = pickle.load(args.cv_file)

    if not args.no_kfp:
        parallel = True
        X_train_kfp = common.convert_records_to_kfp(X_train, parallel)

        if args.open_world_nmon_pages is not None:
            open_world = True
            # From kFP paper
            npos = 60 * args.open_world_nmon_pages
            nneg = 3500
            kfp = classifiers.KFP(open_world, npos, nneg)
        else:
            kfp = classifiers.KFP()

        kfp.metadata["input"] = "kfp"
        kfp.metadata["training"] = args.examples_file.name

        kfp.fit(X_train_kfp, y_train)
        save_model(args.model_dirpath, f"kfp{args.model_tag}", kfp)

    if not args.no_nn:
        X_train_nn = common.convert_records_to_nn_repr(X_train)
        nclasses = np.max(y_train) + 1
        npackets = X_train_nn.shape[1]
        nepochs = 100
        y_train_nn = to_categorical(y_train, num_classes=nclasses)

        nn = classifiers.SirinamDF(npackets, nepochs, nclasses)
        nn.metadata["input"] = "nn"
        nn.metadata["training"] = args.examples_file.name
        nn.fit(X_train_nn, y_train_nn)

        save_model(args.model_dirpath, f"nn{args.model_tag}", nn)

    if not args.no_tt:
        X_train_tt = common.convert_records_to_tiktok_repr(X_train)
        nclasses = np.max(y_train) + 1
        npackets = X_train_tt.shape[1]
        nepochs = 100
        y_train_nn = to_categorical(y_train, num_classes=nclasses)

        nn = classifiers.SirinamDF(npackets, nepochs, nclasses)
        nn.metadata["input"] = "tt"
        nn.metadata["training"] = args.examples_file.name
        nn.fit(X_train_tt, y_train_nn)

        save_model(args.model_dirpath, f"tt{args.model_tag}", nn)

    if not args.no_svm:
        X_train_cumul = common.convert_records_to_cumul_repr(X_train)

        X_val_cumul = None
        if separate_cv:
            X_val_cumul = common.convert_records_to_cumul_repr(X_val)

        svm = classifiers.SVM()
        svm.metadata["input"] = "svm"
        svm.metadata["training"] = args.examples_file.name
        svm.fit(X_train_cumul, y_train, verbose=1, cv_X=X_val_cumul, cv_y=y_val)
        save_model(args.model_dirpath, f"svm{args.model_tag}", svm)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("examples_file", type=argparse.FileType("rb"))
    parser.add_argument("model_dirpath")
    parser.add_argument("--no_nn", action="store_true")
    parser.add_argument("--no_svm", action="store_true")
    parser.add_argument("--no_kfp", action="store_true")
    parser.add_argument("--no_tt", action="store_true")
    parser.add_argument("-t", "--model_tag", default="")
    parser.add_argument("-o", "--open_world_nmon_pages", type=int, default=None)
    parser.add_argument("-c", "--cv_file", type=argparse.FileType("rb"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
