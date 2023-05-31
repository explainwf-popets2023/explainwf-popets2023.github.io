#!/usr/bin/env python3

import argparse
import logging
import pickle
import random
import sys

# The cleaned data may have a port2url set that contains too many urls. We have
# to iterate through the records to limit ourselves to labeling only the
# instances we have.


def common_dict(a, b):
    retval = dict()

    for k, a_value in a.items():
        if k in b:
            b_value = b[k]
            if a_value == b_value:
                retval[k] = a_value
            else:
                raise Exception(
                    f"Value for key {k} does not agree: {a_value}, {b_value}"
                )
    return retval


def main(args):
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logging.info(f"Program arguments: {args}")

    records = []

    for f in args.examples_files:
        logging.info(f"Working on {f.name}")
        records.append(pickle.load(f))

    # Get a common URL dict
    common_port2url = records[0].port2url

    for elem in records[1:]:
        common_port2url = common_dict(common_port2url, elem.port2url)

    all_used_ports = set.union(
        *[set((s for s in record.port2records.keys())) for record in records]
    )

    all_urls = set(common_port2url[p].split("/")[-1] for p in all_used_ports)

    page2label = dict(reversed(x) for x in enumerate(sorted(all_urls)))

    if args.num_monitered > 0: # We are doing a labeled/unlabeled experiment
        unlabeled_set = (args.unlabeled_page,)
        labeled_set = random.sample(page2label.keys(), args.num_monitered)
        tmp = dict()

        for page, label in page2label.items():
            if page in unlabeled_set:
                tmp[page] = 0
            elif page in labeled_set:
                tmp[page] = 1

        page2label = tmp
    else:
        if args.unlabeled_page in page2label:
            del page2label[args.unlabeled_page]

    with open(args.output_filepath, "wb") as out_f:
        pickle.dump(page2label, out_f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_filepath")
    parser.add_argument("examples_files", nargs="*", type=argparse.FileType("rb"))
    parser.add_argument("-n", "--num_monitered", type=int, default=0)
    parser.add_argument("-u", "--unlabeled_page", default="unlabeled")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
