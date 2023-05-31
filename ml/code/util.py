#!/usr/bin/env python3

from pathlib import Path
import logging
import tarfile
import tempfile
import uuid

def check_intersection(train_flows, test_flows):
    train_uids = [flow["uid"] for flow in train_flows]
    test_uids = [flow["uid"] for flow in test_flows]

    s1 = set(train_uids)
    s2 = set(test_uids)

    assert len(s1.intersection(s2)) == 0

def make_tempdir():
    try:
        tmp_dirpath = Path(tempfile.mkdtemp(), 'workdir')
        Path.mkdir(tmp_dirpath, parents=True)
    except Exception as e:
        logging.error(e)
        return None

    return tmp_dirpath

def partition_list(l, partition_indices):

    retval = []

    acc = 0

    for index in partition_indices:
        index = index - acc
        retval.append(l[:index])
        l = l[index:]
        acc += index

    retval.append(l)
    return retval

def get_uuid_str():
    return str(uuid.uuid4())

def get_uuid_filepath(dirpath, extension=""):
    return Path(dirpath, f"{get_uuid_str()}{extension}")

def write_tarfile_gzipped(tarfile_path, input_filepaths):
    with tarfile.TarFile.open(str(tarfile_path), "w|gz") as tarf:
        for input_filepath in input_filepaths:
            path = Path(input_filepath)
            tarf.add(input_filepath, arcname=path.name)
