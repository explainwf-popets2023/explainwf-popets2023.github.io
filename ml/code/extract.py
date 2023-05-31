#!/usr/bin/env python3

import argparse
import logging
import pickle
import random
import sys

from common import IndexedRecordData, read_url_file, read_cell_file, remove_outliers


def main(args):
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logging.info(f"Program arguments: {args}")

    port2url = read_url_file(args.urls_filepath)

    if args.nkeep_port_80 > 0:
        port2url[80] = "unlabeled"

    use_zstd = True

    if args.cells_decompressed:
        use_zstd = False

    filter_params = None
    if args.filter_cells:
        filter_params = {
            "time_jump_delta_s": args.filter_cells_delta_t,
            "time_jump_idx_thresh": args.filter_cells_max_t_idx,
        }

    port2records = read_cell_file(
        args.cells_filepath,
        exclusive_ports=set(list(port2url.keys()) + [80]),
        use_zstd=use_zstd,
        filter_params=filter_params,
    )

    if args.filter_cells:
        new_port2records = dict()

        for port, records in port2records.items():
            new_port2records[port] = []

            for record in records:
                if len(record["cells"]) >= args.filter_cells_min_len:
                    new_port2records[port].append(record)

        port2records = new_port2records

    cleaned_port2records = dict()

    for idx, (port, records) in enumerate(port2records.items()):
        logging.info(f"Working on {idx}")

        n = args.nkeep
        if port == 80:
            n = args.nkeep_port_80

        if n == 0:
            continue

        n = min(n, len(records))

        if args.remove_outliers:
            cleaned_records = remove_outliers(records, n)
        else:
            cleaned_records = random.sample(records, n)

        cleaned_port2records[port] = cleaned_records

    ird = IndexedRecordData(port2url, cleaned_port2records)

    with open(args.output_filepath, "wb") as out_f:
        pickle.dump(ird, out_f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cells_filepath")
    parser.add_argument("urls_filepath")
    parser.add_argument("output_filepath")
    parser.add_argument("-n", "--nkeep", type=int, default=50)
    parser.add_argument("--nkeep_port_80", type=int, default=0)
    parser.add_argument("-c", "--cells_decompressed", action="store_true")
    parser.add_argument("-r", "--remove_outliers", action="store_true")
    parser.add_argument("-f", "--filter_cells", action="store_true")
    parser.add_argument("--filter_cells_delta_t", type=float, default=15.0)
    parser.add_argument("--filter_cells_max_t_idx", type=int, default=10)
    parser.add_argument("--filter_cells_min_len", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
