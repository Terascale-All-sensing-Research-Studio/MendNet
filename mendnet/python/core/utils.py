import os
import logging


def file_is_old(path, overwrite_time=1636410000):
    """
    Returns true if a file was last edited before the input time.
    """
    if overwrite_time is None:
        return False
    if (
        os.path.exists(path)
        and overwrite_time is not None
        and os.path.getmtime(path) < overwrite_time
    ):
        return True
    return False


def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("AutoDecoder - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def get_file(d, fe, fn=None):
    """
    Given a root directory and a list of file extensions, recursively
    return all files in that directory that have that extension.
    """
    for f in os.listdir(d):
        fp = os.path.join(d, f)
        if os.path.isdir(fp):
            yield from get_file(fp, fe, fn)
        elif os.path.splitext(fp)[-1] in fe:
            if fn is None:
                yield fp
            elif fn == os.path.splitext(f)[0]:
                yield fp


def find_specs(p):
    if os.path.basename(p) == "specs.json":
        return p
    p = p.replace("$DATADIR", os.environ["DATADIR"])
    if not os.path.exists(os.path.join(p, "specs.json")):
        p = os.path.dirname(os.path.dirname(p))
    assert os.path.isfile(
        os.path.join(p, "specs.json")
    ), "Could not find specs file in directory: {}".format(p)
    logging.info("Found specs at: {}".format(os.path.join(p, "specs.json")))
    return os.path.join(p, "specs.json")
