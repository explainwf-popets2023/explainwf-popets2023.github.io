#!/usr/bin/env python3

from abc import ABC, ABCMeta, abstractmethod
from functools import lru_cache
import logging
import pathlib
from pathlib import Path
import json
import pickle
import random
import shutil
import tarfile

import build_dnn
from util import *

import numpy as np
from pathos.multiprocessing import ProcessingPool
from scipy.spatial.distance import hamming
from sklearn import tree
import sklearn.metrics, sklearn.preprocessing
import sklearn.svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.utils import to_categorical


def make_probs_from_binary_labels(labels):
    n = len(labels)
    retval = np.zeros(shape=(n, 2), dtype=float)

    for idx, label in enumerate(labels):
        retval[idx][int(label)] = 1.0

    return retval


def make_probs_from_positive_prs(positive_prs):
    n = len(positive_prs)
    retval = np.zeros(shape=(n, 2), dtype=float)

    for idx, pr in enumerate(positive_prs):
        retval[idx][0] = 1.0 - pr
        retval[idx][1] = pr

    return retval


def save_keras_model_to_file(model, filepath):
    model.save_weights(filepath)
    return filepath


def save_classifier_to_file(classifier, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(classifier, f)
    return filepath


def save_fileinto_to_file(classifier_filepath, weights_filepath, filepath):
    fileinfo = {
        "classifier_filepath": str(classifier_filepath.name),
        "weights_filepath": str(weights_filepath.name),
    }

    with open(filepath, "w") as outf:
        json.dump(fileinfo, outf)

    return filepath


def write_classifier_into_tarfile_gzipped(classifier, tarfile_path):
    tmp_dirpath = make_tempdir()

    weights_filepath = get_uuid_filepath(tmp_dirpath, ".h5")
    classifier.save_weights(weights_filepath)

    classifier_filepath = get_uuid_filepath(tmp_dirpath, ".pkl")
    save_classifier_to_file(classifier, classifier_filepath)

    info_filepath = pathlib.Path(tmp_dirpath, "info.json")
    save_fileinto_to_file(classifier_filepath, weights_filepath, info_filepath)

    write_tarfile_gzipped(
        tarfile_path, (weights_filepath, classifier_filepath, info_filepath)
    )

    shutil.rmtree(tmp_dirpath)

    return tarfile_path


def read_classifier_from_tarfile_gzipped(tarfile_path):
    tmp_dirpath = make_tempdir()

    with tarfile.TarFile.open(tarfile_path, "r|gz") as tarf:
        tarf.extractall(tmp_dirpath)

    with open(Path(tmp_dirpath, "info.json"), "r") as inf:
        fileinfo = json.load(inf)

    with open(Path(tmp_dirpath, fileinfo["classifier_filepath"]), "rb") as inf:
        classifier = pickle.load(inf)

    classifier.load_weights(Path(tmp_dirpath, fileinfo["weights_filepath"]))

    shutil.rmtree(tmp_dirpath)

    return classifier


def compute_stats(ytrue, ypred, binary=True):
    acc = sklearn.metrics.accuracy_score(ytrue, ypred)
    cm = sklearn.metrics.confusion_matrix(ytrue, ypred)

    if binary:
        recall = sklearn.metrics.recall_score(ytrue, ypred)
        precision = sklearn.metrics.precision_score(ytrue, ypred)

        # First index is true label
        # Second index is predicted label
        tn = int(cm[0, 0])
        tp = int(cm[1, 1])
        fn = int(cm[1, 0])
        fp = int(cm[0, 1])

        if fp + tn > 0:
            fpr = float(fp / (fp + tn))
        else:
            fpr = float("nan")
    else:
        average = "macro"
        recall = sklearn.metrics.recall_score(ytrue, ypred, average=average)
        precision = sklearn.metrics.precision_score(ytrue, ypred, average=average)
        tn = None
        tp = None
        fn = None
        fp = None
        fpr = None

    return {
        "acc": acc,
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "confusion": cm,
    }


class Classifier(metaclass=ABCMeta):
    def __init__(self):
        self._metadata = dict()

    @abstractmethod
    def fit(self, X, y):
        pass

    def transform_records(self, records):
        return records

    @abstractmethod
    def load_weights(self, filepath):
        pass

    @abstractmethod
    def save_weights(self, filepath):
        f = open(filepath, "w")
        f.close()

    def set_metadata(self, metadata):
        self._metadata = metadata

    def update_metadata(self, key, value):
        self._metadata[key] = value

    @property
    def metadata(self):
        return self._metadata

    @abstractmethod
    def predict(self, X):
        pass


class KFP(Classifier):
    # Open-world assumes binary
    def __init__(self, open_world=False, npos=None, nneg=None, k=3):
        super().__init__()
        self._model = RandomForestClassifier(n_estimators=1000, oob_score=True)
        self._open_world = open_world
        self._npos = npos
        self._nneg = nneg
        self._k = k
        self._training_points = []

    def fit(self, X, y):
        logging.info("BEGIN fit kfp")
        self._model.fit(X, y)

        if self._open_world:
            assert len(set(y)) == 2
            X = self._model.apply(X)
            examples_and_labels = list(zip(X, y))
            negative_examples = [e for e in examples_and_labels if e[1] == 0]
            positive_examples = [e for e in examples_and_labels if e[1] == 1]
            neg = random.sample(
                negative_examples, k=min(self._nneg, len(negative_examples))
            )
            pos = random.sample(
                positive_examples, k=min(self._npos, len(positive_examples))
            )
            self._training_points.extend(neg)
            self._training_points.extend(pos)
        logging.info("END fit kfp")

    def save_weights(self, filepath):
        super().save_weights(filepath)

    def load_weights(self, filepath):
        pass

    def _label_point(self, p):
        training_distances = sorted(
            [(x, y, hamming(p, x)) for x, y in self._training_points],
            key=lambda x: x[2],
        )

        if all(map(lambda x: x == 1, [e[1] for e in training_distances[: self._k]])):
            return 1
        else:
            return 0

    def predict(self, X, parallel=True):
        if not self._open_world:
            return self._model.predict(X)
        else:
            X = self._model.apply(X)

            if not parallel:
                return np.fromiter((self._label_point(p) for p in X), dtype=int)
            else:
                p = ProcessingPool()
                return np.fromiter(
                    p.map(self._label_point, X), dtype=int
                )


class SVM(Classifier):
    def __init__(self):
        super().__init__()
        self._model = sklearn.svm.SVC(kernel="rbf")
        self._scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))

    def fit(self, X, y, multiprocess=True, cv=True, verbose=0, cv_X=None, cv_y=None):
        X_scaled = self._scaler.fit_transform(X)

        if cv:
            param_grid = {
                "C": np.logspace(-5, 15, 10, base=2),
                "gamma": np.logspace(-15, 3, 10, base=2),
            }

            if multiprocess:
                njobs = -1
            else:
                njobs = 1

            best_params = None
            if cv_X is not None:
                assert cv_y is not None
                X_val_scaled = self._scaler.transform(cv_X)

                X_concat = np.concatenate((X_scaled, X_val_scaled))
                y_concat = np.concatenate((y, cv_y))

                train_idx = np.arange(0, len(X_scaled))
                test_idx = np.arange(len(X_scaled), len(X_scaled) + len(X_val_scaled))

                grid_search = GridSearchCV(
                    self._model,
                    param_grid,
                    cv=[(train_idx, test_idx)],
                    scoring="accuracy",
                    n_jobs=njobs,
                    verbose=verbose,
                )
                grid_search.fit(X_concat, y_concat)
                best_params = grid_search.best_params_
            else:
                grid_search = GridSearchCV(
                    self._model,
                    param_grid,
                    cv=3,
                    scoring="accuracy",
                    n_jobs=njobs,
                    verbose=verbose,
                )
                grid_search.fit(X_scaled, y)
                best_params = grid_search.best_params_

            logging.info(f"Best params: {best_params}")
            self._model = sklearn.svm.SVC(kernel="rbf", **grid_search.best_params_)

        self._model.fit(X_scaled, y)

    def save_weights(self, filepath):
        super().save_weights(filepath)

    def load_weights(self, filepath):
        pass

    def predict(self, X):
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)


class DTree(Classifier):
    def __init__(self, input_format, positive_class):
        super().__init__()
        self._input_format = input_format
        self._positive_class = positive_class
        self._model = tree.DecisionTreeClassifier()

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict_proba(self, X):
        return make_probs_from_binary_labels(self._model.predict(X))

    def save_weights(self, filepath):
        super().save_weights(filepath)

    def _fine_tune(self, X, y):
        pass

    @property
    def input_format(self):
        return self._input_format

    @property
    def positive_class(self):
        return self._positive_class

    def load_weights(self, filepath):
        pass

class SirinamDF(Classifier):
    def __init__(self, npackets, nepochs, nclasses):
        super().__init__()
        self._model = build_dnn.build((npackets, 1), nclasses)
        self._npackets = npackets
        self._nepochs = nepochs

    def fit(self, X, y):
        self._model.fit(X, y, batch_size=256, epochs=self._nepochs, verbose=1)

    def predict_proba(self, X):
        X_trunc = X[:, : self._npackets]
        return make_probs_from_positive_prs(self._model.predict(X_trunc))

    def save_weights(self, filepath):
        save_keras_model_to_file(self._model, filepath)

    def _fine_tune(self, X, y):
        best_youdens_thresh, best_f1_thresh = dl_classifier_tune(self, X, y)
        self.update_metadata("best_youdens_thresh", best_youdens_thresh)
        self.update_metadata("best_f1_thresh", best_f1_thresh)
        self._thresh = best_f1_thresh

    @property
    def input_format(self):
        return "sized"

    @property
    def positive_class(self):
        return self._positive_class

    def load_weights(self, filepath):
        return self._model.load_weights(filepath)

    def predict(self, X):
        pred = self._model.predict(X)
        print(pred)
        pred_labels = np.argmax(pred, axis=1)
        return pred_labels
