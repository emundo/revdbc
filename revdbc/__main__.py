import argparse
from datetime import datetime
import logging
import os
import tempfile
from typing import Dict, Tuple

import candumpgen
import cantools
import numpy as np

from . import revdbc


def run_common(args: argparse.Namespace) -> Tuple[str, str]:
    path = os.path.abspath(args.file)
    if not os.path.exists(path):
        raise Exception("`{}` does not point to a file/directory.".format(path))

    log_level = logging.ERROR
    if args.verbose > 0:
        log_level = logging.WARNING
    if args.verbose > 1:
        log_level = logging.INFO
    if args.verbose > 2:
        log_level = logging.DEBUG

    output_directory = os.path.abspath(args.output_directory)
    if not os.path.isdir(output_directory):
        raise Exception("`--out={}` does not point to a directory.".format(output_directory))
    run_output_directory = os.path.join(output_directory, str(datetime.now()))

    try:
        os.mkdir(run_output_directory)
    except FileExistsError:
        pass

    log_file = os.path.join(run_output_directory, "run.log")

    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=log_level,
                        handlers=[ logging.StreamHandler(), logging.FileHandler(log_file) ])

    return path, run_output_directory


def run_analysis(args: argparse.Namespace) -> None:
    path, run_output_directory = run_common(args)

    if not os.path.isfile(path):
        raise Exception("`{}` does not point to a file.".format(path))

    candump = revdbc.load_candump(path)

    identifiers = np.unique(candump["identifier"])

    print("Found identifiers: {}".format([
        (x, len(candump[candump["identifier"] == x])) for x in identifiers
    ]))

    while True:
        try:
            identifier = int(input("Select one of them: "))
            if identifier in identifiers:
                break
        except ValueError:
            pass
    bodies = candump[candump["identifier"] == identifier]["data"]
    sizes = np.unique(bodies["size"])

    if len(sizes) != 1:
        raise Exception(
            "Can't process identifier {}, whose packet sizes differ throughout the candump."
            .format(identifier)
        )

    print("{} packets found for selected identifier {}.".format(len(bodies), identifier))

    restored_dbc = revdbc.analyze_identifier(
        identifier,
        bodies["bits"],
        sizes[0],
        run_output_directory
    ).restored_dbc

    cantools.subparsers.dump._dump_can_database(restored_dbc)  # pylint: disable=protected-access


TOP_LEVEL_NAME = __name__


def run_test(args: argparse.Namespace) -> None:
    path, run_output_directory = run_common(args)

    if os.path.isfile(path):
        test_cases = [ path ]
    else:
        test_cases = [ os.path.join(path, f) for f in next(os.walk(path))[2] if f.endswith(".dbc") ]

    results: Dict[str, Tuple[float, float]] = {}
    for original_dbc in test_cases:
        distances = []
        for run in range(args.runs):
            with tempfile.TemporaryFile(mode="w+") as f:
                candumpgen.generate_candump(f, original_dbc, "can0")
                f.seek(0)
                candump = revdbc.load_candump(f)

            for identifier in np.unique(candump["identifier"]):
                bodies = candump[candump["identifier"] == identifier]["data"]
                sizes = np.unique(bodies["size"])

                if len(sizes) != 1:
                    logging.getLogger(TOP_LEVEL_NAME).warning(
                        "Skipping identifier %s, whose packet sizes differ throughout the candump.",
                        identifier
                    )
                    continue

                restored_dbc_file = revdbc.analyze_identifier(
                    identifier,
                    bodies["bits"],
                    sizes[0],
                    run_output_directory,
                    "_{}_testrun{}".format(os.path.basename(original_dbc).replace(".", "_"), run + 1)
                ).restored_dbc_file

            distance = candumpgen.dbc_dist(original_dbc, restored_dbc_file)

            logging.getLogger(TOP_LEVEL_NAME).info(
                "Distance between original and restored DBC files: %s",
                distance
            )

            distances.append(distance)

        average_distance = sum(distances) / len(distances)

        def squared_difference(distance: float, average_distance: float = average_distance) -> float:
            return (distance - average_distance) ** 2

        variance = sum(map(squared_difference, distances)) / len(distances)

        logging.getLogger(TOP_LEVEL_NAME).info(
            "Average distance: %s; Variance: %s",
            average_distance,
            variance
        )

        results[original_dbc] = (average_distance, variance)

    first_col_heading = "Results ({} iterations each)".format(args.runs)
    second_col_heading = "Average"
    third_col_heading = "Variance"

    print("")
    print("")
    print("{:30s}    {:7s}    {:8s}".format(first_col_heading, second_col_heading, third_col_heading))

    for dbc, (avg, var) in results.items():
        print("{:30s}    {:7.2f}    {:8.2f}".format(os.path.basename(dbc), avg, var))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reverse-engineer DBC definitions from CAN dumps.")

    parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0,
                        help="Increase output verbosity, up to three times.")

    parser.add_argument("-o", "--out", dest="output_directory", default=".", type=str,
                        help="The directory to store outputs in. A new subdirectory is created for"
                             " each run, next to a subdirectory for converted candumps. Defaults to"
                             " the working directory.")

    subparsers = parser.add_subparsers(title="subcommands", required=True)

    parser_analyze = subparsers.add_parser("analyze", help="Analyze a candump file.", aliases=["a"])

    parser_analyze.add_argument("file", metavar="FILE", type=str,
                                help="The candump file to load and analyze.")

    parser_analyze.set_defaults(func=run_analysis)

    parser_test = subparsers.add_parser(
        "test",
        help="Test/benchmark the performance of the analysis."
    )

    parser_test.add_argument("file", metavar="FILE", type=str, help="The DBC file to benchmark on.")

    parser_test.add_argument("-r", "--runs", dest="runs", default=8, type=int,
                             help="The number of runs to repeat and average each test case"
                                  " (defaults to 8).")

    parser_test.set_defaults(func=run_test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
