#!/usr/bin/env python3

import argparse
import logging
import pickle
import sys

from sklearn.model_selection import train_test_split

from collections import Counter


def main(args):
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logging.info(f"Program arguments: {args}")

    indexed_records = pickle.load(args.records_file)
    args.records_file.close()

    page2label = pickle.load(args.labels_file)
    args.labels_file.close()

    all_records = []
    all_labels = []
    all_labels_uniq = set()

    for port, records in indexed_records.port2records.items():
        url = indexed_records.port2url[port]
        page = url.split("/")[-1]

        if page not in page2label:
            continue

        page_label = page2label[page]

        if not args.binary and page_label in all_labels_uniq:
            logging.info(f"Skipping duplicated page: {page}")
            continue
        else:
            all_labels_uniq.add(page_label)

        all_records.extend(records)
        all_labels.extend((page2label[page] for _ in range(len(records))))

    assert len(all_records) == len(all_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        all_records,
        all_labels,
        train_size=args.train_size,
        shuffle=True,
        stratify=all_labels,
    )

    pickle.dump((X_train, X_test, y_train, y_test), args.output_file)
    args.output_file.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("records_file", type=argparse.FileType("rb"))
    parser.add_argument("labels_file", type=argparse.FileType("rb"))
    parser.add_argument("output_file", type=argparse.FileType("wb"))
    parser.add_argument("-t", "--train_size", type=float, default=0.6)
    parser.add_argument("-b", "--binary", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
