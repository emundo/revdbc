# Python 2/3 compatibility imports. See https://python-future.org/imports.html
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import *
from future import standard_library
standard_library.install_aliases()

import argparse
from datetime import datetime
import logging
import os
import tempfile

from . import revdbc

import candumpgen
import numpy as np

def run_common(args):
    path = os.path.abspath(args.file)
    if not os.path.isfile(path):
        raise Exception("`{}` does not point to a file.".format(path))

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

def run_analysis(args):
    path, run_output_directory = run_common(args)

    candump = revdbc.load_candump(path)

    # TODO: Temporary code
    identifiers = np.unique(candump["identifier"])
    print("Found identifiers: {}".format([ (x, len(candump[candump["identifier"] == x])) for x in identifiers ]))
    while True:
        try:
            identifier = int(input("Select one of them: "))
            if identifier in identifiers:
                break
        except ValueError:
            pass
    bodies = candump[candump["identifier"] == identifier]["data"]
    print("{} packets found for selected identifier {}.".format(len(bodies), identifier))

    revdbc.analyze_identifier(identifier, bodies, run_output_directory, args.show_plots)
    # TODO: End temporary code

top_level_name = __name__
def run_test(args):
    original_dbc, run_output_directory = run_common(args)

    with tempfile.TemporaryFile(mode="w+") as f:
        candumpgen.generate_candump(f, original_dbc)
        f.seek(0)
        candump = revdbc.load_candump(f)

    for identifier in np.unique(candump["identifier"]):
        bodies = candump[candump["identifier"] == identifier]["data"]
        sizes = np.unique(bodies["size"])

        if len(sizes) != 1:
            print(
                "Skipping identifier {}, whose packet sizes differ throughout the candump."
                .format(identifier)
            )
            continue

        restored_dbc = revdbc.analyze_identifier(
            identifier,
            bodies["bits"],
            sizes[0],
            run_output_directory,
            args.show_plots
        )

    distance = candumpgen.dbc_dist(original_dbc, restored_dbc)

    logging.getLogger(top_level_name).info(
        "Distance between original and restored DBC files: %s",
        distance
    )

def main():
    parser = argparse.ArgumentParser(description="Reverse-engineer DBC definitions from CAN dumps.")

    parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0,
                        help="Increase output verbosity, up to three times.")

    parser.add_argument("-o", "--out", dest="output_directory", default=".", type=str,
                        help="The directory to store outputs in. A new subdirectory is created for"
                             " each run, next to a subdirectory for converted candumps. Defaults to"
                             " the working directory.")

    parser.add_argument("--show-plots", dest="show_plots", action="store_true",
                        help="Show plots during the run in addition to saving them to the output"
                             " directory. Showing a plot blocks execution until the plot windows is"
                             " closed.")

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

    parser_test.set_defaults(func=run_test)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
