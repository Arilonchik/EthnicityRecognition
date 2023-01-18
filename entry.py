import argparse
from scripts.sync_filter import filter_sync
from scripts.async_filter import filter_async
from scripts.entry_config import *


def agr_parses():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["async", "sync"],
                        help="Working mode, sync - for online usage,"
                             " async - for sorting dataset or get meta, (Now only async is available)")
    parser.add_argument("-m", "--meta",
                        help="If passed - it makes only meta file, without sorting photos in done dir",
                        action="store_true")

    return parser.parse_args()


def main(args):

    if args.mode == "async":
        filter_async(WORK_DIR, DONE_DIR, MODEL_CONFIG, DEVICE, args.meta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["async", "sync"],
                        help="Working mode, sync - for online usage,"
                             " async - for sorting dataset or get meta, (Now only async is available)")
    parser.add_argument("-m", "--meta",
                        help="If passed - it makes only meta file, without sorting photos in done dir",
                        action="store_true")

    args = parser.parse_args()
    main(args)
