#!/usr/bin/env python

from collections import defaultdict
from copy import deepcopy
import csv
from dataclasses import dataclass
import logging
import multiprocessing as mp
import json

import numpy as np
from scipy.spatial.distance import cosine
from scipy.cluster.vq import whiten
from zstd import ZSTD_uncompress
import pandas as pd

import RF_fextract

CELL_PROPERTY = {
    "relative_time_s": 0,
    "side": 1,
    "direction": 2,
    "cell_command": 3,
    "relay_command": 4,
}

DIRECTION = {"client-to-server": 0, "server-to-client": 1}

CELL_COMMAND = {
    "CELL_PADDING": 0,
    "CELL_CREATE": 1,
    "CELL_CREATED": 2,
    "CELL_RELAY": 3,
    "CELL_DESTROY": 4,
    "CELL_CREATE_FAST": 5,
    "CELL_CREATED_FAST": 6,
    "CELL_VERSIONS": 7,
    "CELL_NETINFO": 8,
    "CELL_RELAY_EARLY": 9,
    "CELL_CREATE2": 10,
    "CELL_CREATED2": 11,
    "CELL_PADDING_NEGOTIATE": 12,
}


@dataclass
class IndexedRecordData:
    port2url: dict
    port2records: dict


class BiDict:
    def __init__(self):
        self._forward_dict = dict()
        self._reverse_dict = dict()

    def __setitem__(self, key, value):
        self.forward_set(key, value)

    def __delitem__(self, key):
        self.forward_del(key)

    def __getitem__(self, key):
        return self.forward_lookup(key)

    def _symmetric_delete(d1, d2, key):
        value = d1[key]
        del d1[key]
        del d2[value]

    def forward_del(self, key):
        BiDict._symmetric_delete(self._forward_dict, self._reverse_dict, key)

    def reverse_del(self, key):
        BiDict._symmetric_delete(self._reverse_dict, self._forward_dict, key)

    def _symmetric_set(d1, d2, key, value):
        try:
            BiDict._symmetric_delete(d1, d2, key)
        except:
            pass

        try:
            BiDict._symmetric_delete(d2, d1, value)
        except:
            pass

        d1[key] = value
        d2[value] = key

    def forward_set(self, key, value):
        BiDict._symmetric_set(self._forward_dict, self._reverse_dict, key, value)

    def reverse_set(self, key, value):
        BiDict._symmetric_set(self._reverse_dict, self._forward_dict, key, value)

    def forward_lookup(self, key):
        return self._forward_dict[key]

    def reverse_lookup(self, key):
        return self._reverse_dict[key]

    def __len__(self):
        """Returns the number of connections"""
        return len(self._forward_dict)

    def __repr__(self):
        return str(self._forward_dict) + " | " + str(self._reverse_dict)


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def convert_directions_to_bursts(directions):
    bursts = []

    if len(directions) == 0:
        return bursts

    current_direction = directions[0]
    nburst = 1

    for direction in directions[1:]:
        if direction != current_direction:
            bursts.append((current_direction, nburst))
            current_direction = direction
            nburst = 1
        else:
            nburst += 1

    bursts.append((current_direction, nburst))

    return bursts


def _convert_direction_vector_to_cumul_repr(directions, n=100):
    assert n >= 2

    cumul_repr = [0]

    for direction in directions:
        if direction == DIRECTION["client-to-server"]:
            cumul_repr.append(-1 + cumul_repr[-1])
        else:
            cumul_repr.append(1 + cumul_repr[-1])

    xs = np.linspace(0, 100, num=len(cumul_repr))
    ys = cumul_repr

    return np.interp(np.linspace(0, 100, num=n), xs, ys)


def convert_directions_to_cumul_repr(direction_matrix, n=100):
    return [_convert_direction_vector_to_cumul_repr(row, n) for row in direction_matrix]


def convert_records_to_cumul_repr(records, n=100):
    directions = [
        extract_cell_property(record["cells"], "direction") for record in records
    ]

    return convert_directions_to_cumul_repr(directions, n)


def _adjust_directions(directions):
    return [1 if d == 0 else -1 for d in directions]


def filter_record_cells(record, time_jump_delta_s=15, time_jump_idx_thresh=10):
    new_record = deepcopy(record)

    allowed_cell_commands = tuple(
        CELL_COMMAND[c] for c in ("CELL_RELAY", "CELL_RELAY_EARLY")
    )

    cells = new_record["cells"]
    cells = [
        c for c in cells if c[CELL_PROPERTY["cell_command"]] in allowed_cell_commands
    ]

    cells = [c for c in cells if c[CELL_PROPERTY["relay_command"]] == 0]

    times = extract_cell_property(cells, "relative_time_s")
    iats = [x[0] - x[1] for x in zip(times[1:], times)]

    largest_jump_idx = np.argmax(iats)
    largest_jump = iats[largest_jump_idx]

    if largest_jump >= time_jump_delta_s and largest_jump_idx <= time_jump_idx_thresh:
        cells = cells[largest_jump_idx + 1 :]

    # Re-offset the time of the cells
    if len(cells) > 0:
        t0 = cells[0][CELL_PROPERTY["relative_time_s"]]
        cells = [[c[0] - t0] + c[1:] for c in cells]

    new_record["cells"] = cells

    return new_record


def convert_directions_to_nn_repr(directions, n=5000):
    data = np.zeros(shape=(len(directions), n))

    for idx, direction in enumerate(directions):
        data[idx, : len(direction)] = _adjust_directions(direction)

    return data


def convert_directions_times_to_tiktok_repr(directions, times, n=5000):
    data = np.zeros(shape=(len(directions), n))

    for idx, (direction, time) in enumerate(zip(directions, times)):
        data[idx, : len(direction)] = _adjust_directions(direction)
        data[idx, : len(time)] *= time

    return data


def convert_records_to_nn_repr(records, n=5000):
    directions = [
        extract_cell_property(record["cells"], "direction") for record in records
    ]

    return convert_directions_to_nn_repr(directions, n)


def convert_records_to_tiktok_repr(records, n=5000):
    directions = [
        extract_cell_property(record["cells"], "direction") for record in records
    ]

    times = [
        extract_cell_property(record["cells"], "relative_time_s") for record in records
    ]

    return convert_directions_times_to_tiktok_repr(directions, times, n)


def _convert_record_to_kfp(record):
    lines = []

    microsecond = 1e-6
    summed_offset_time = 0.0

    for cell in record["cells"]:
        d = cell[CELL_PROPERTY["direction"]]
        t = cell[CELL_PROPERTY["relative_time_s"]]

        if d == DIRECTION["client-to-server"]:
            token = 1
        else:
            token = -1

        line = f"{t + summed_offset_time} {token}"
        summed_offset_time += microsecond

        lines.append(line)

    return np.fromiter(RF_fextract.TOTAL_FEATURES(lines), dtype=float)


def convert_records_to_kfp(records, parallel=False):
    if not parallel:
        converted_records = [_convert_record_to_kfp(r) for r in records]
    else:
        with mp.Pool() as pool:
            converted_records = list(pool.map(_convert_record_to_kfp, records))

    concat = np.concatenate(converted_records)
    nfeatures = len(concat) // len(records)
    return concat.reshape((len(records), nfeatures))


def extract_cell_property(cells, property):
    return [x[CELL_PROPERTY[property]] for x in cells]


class Model:
    def __init__(self):
        self._model = None

    def __repr__(self):
        return "---\n" + str(self._model) + "\n---"

    def fit(self, points):
        self._model = points

    def loss(self, test_point):
        assert self._model is not None

        model_loss = 0.0
        for example_point in self._model:
            dist = cosine(example_point, test_point)
            model_loss += dist

        return model_loss


def _ransac(points, s, num_iters):
    models = []

    for _ in range(num_iters):
        sampled_indices = np.random.choice(range(len(points)), s)
        nonsampled_indices = np.setdiff1d(range(len(points)), sampled_indices)

        model = Model()
        model.fit(points[sampled_indices])

        total_loss = 0

        for idx in nonsampled_indices:
            total_loss = model.loss(points[idx])

        models.append((model, total_loss))

    best_model = sorted(models, key=lambda x: x[1])[0]

    return best_model[0]


def _compute_all_manual_features(record):
    if len(record["cells"]) < 2:
        return None

    npackets = len(record["cells"])
    directions = extract_cell_property(record["cells"], "direction")

    total_sent = sum(
        1 for _ in (d for d in directions if d == DIRECTION["client-to-server"])
    )

    total_recv = sum(
        1 for _ in (d for d in directions if d == DIRECTION["server-to-client"])
    )

    cumul = _convert_direction_vector_to_cumul_repr(directions, 100)

    bursts = convert_directions_to_bursts(directions)

    if npackets > 6:
        last_cell_offset = -7
    else:
        last_cell_offset = -1

    return {
        "npackets": npackets,
        "total_sent": total_sent,
        "total_recv": total_recv,
        "cumul_25": cumul[24],
        "cumul_50": cumul[49],
        "cumul_75": cumul[74],
        "cumul_100": cumul[99],
        "nbursts": len(bursts),
        "avg_burst": np.mean([b[1] for b in bursts]),
        "max_burst": np.max([b[1] for b in bursts]),
        "ttlb": record["cells"][last_cell_offset][CELL_PROPERTY["relative_time_s"]],
    }


def _compute_ransac_features(record):
    all_features = _compute_all_manual_features(record)
    del all_features["ttlb"]
    return all_features


def remove_outliers(records, nkeep=50):
    logging.info("BEGIN computing RANSAC features")
    df = pd.DataFrame(map(_compute_ransac_features, records))
    logging.info("END computing RANSAC features")

    points = whiten(df)

    s = 5  # Model sample size
    p = 0.999  # Probability of a good model
    num_iters = np.log(1 - p) / np.log(1 - (1 - 0.49) ** s)
    num_iters = int(np.ceil(num_iters))

    logging.info("BEGIN running RANSAC")
    model = _ransac(points, s, num_iters)
    logging.info("END running RANSAC")

    point_distances = []

    logging.info("BEGIN computing model distances")
    for idx in range(len(points)):
        loss = model.loss(points[idx])
        point_distances.append((idx, loss))
    logging.info("END computing model distances")

    point_distances = sorted(point_distances, key=lambda x: x[1])
    outliers = set((p[0] for p in point_distances[nkeep:]))
    inlier_records = [r[1] for r in enumerate(records) if r[0] not in outliers]
    return inlier_records


def read_cell_file(
    filepath,
    num_per_port=None,
    use_zstd=True,
    exclusive_ports=set(),
    filter_params=None,
):
    logging.info("BEGIN decompressing cell file")
    with open(filepath, "rb") as f:
        if use_zstd:
            data = ZSTD_uncompress(f.read()).decode("utf-8")
        else:
            data = f.read().decode("utf-8")
    logging.info("END decompressing cell file")

    port2records = defaultdict(list)

    logging.info("BEGIN organizing records by port")

    for idx, line in enumerate(data.splitlines()):
        fields = line.split()
        try:
            if fields[1] != "GWF":
                raise Exception("Unexpected format")
            record = json.loads(fields[2])

            port = record["port"]

            if exclusive_ports is not None:
                if port not in exclusive_ports:
                    continue

            if filter_params is not None:
                record = filter_record_cells(record, **filter_params)

            directions = extract_cell_property(record["cells"], "direction")
            incoming_packets = [
                d for d in directions if d == DIRECTION["server-to-client"]
            ]

            outgoing_packets = [
                d for d in directions if d == DIRECTION["client-to-server"]
            ]

            # Clearly fewer than 3 cells in either direction is not enough for a
            # complete flow. This case causes an error for kFP.
            if len(incoming_packets) < 3 or len(outgoing_packets) < 3:
                logging.info("Filtering short flow")
                continue

            if record is None or record["cells"] is None:
                continue

            port2records[port].append(record)
        except:
            logging.info(f"Error parsing line ({idx}) {line}")

    logging.info("END organizing records by port")

    return port2records


def read_url_file(filepath):
    port2label = dict()

    with open(filepath, "r") as in_f:
        reader = csv.reader(in_f, delimiter=" ")
        for fields in reader:
            ip_addr, port, label = fields
            port = int(port)
            port2label[port] = label

    return port2label


def read_jsonl_file(file):
    records = []

    for line in file:
        line = line.strip()

        if len(line) == 0:
            continue
        records.append(json.loads(line))

    return records
