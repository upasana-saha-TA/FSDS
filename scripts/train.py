
import argparse
import logging
import logging.config
import os
import os.path as op
import pickle
import sys

import housinglib as hlb

parser = argparse.ArgumentParser(description="data folder path")
parser.add_argument("--input_path", nargs="?")
parser.add_argument("--output_path", nargs="?")
parser.add_argument("--log_level", nargs="?")
parser.add_argument("--log_path", nargs="?")
parser.add_argument("--no_console_log", nargs="?")
args = parser.parse_args()

if args.log_level is None:
    log_level = "DEBUG"
else:
    log_level = args.log_level

if args.log_path is None:
    log_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "logs", "train.log"
    )
else:
    log_file = args.log_path

log_dir = os.path.dirname(log_file)
os.makedirs(log_dir, exist_ok=True)

if args.no_console_log is None:
    no_console_log = True
else:
    no_console_log = args.no_console_log

HERE = op.dirname(op.abspath(__file__))

if args.input_path is None:
    HOUSING_PATH = op.join(HERE, "..", "data", "raw")
else:
    HOUSING_PATH = args.input_path

if args.output_path is None:
    path = op.join(HERE, "..", "artifacts")
else:
    path = args.output_path

LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s \
                - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(message)s"},
    },
    "root": {"level": "DEBUG"},
}


def configure_logger(
    logger=None, cfg=None, log_file=None, console=True, log_level="DEBUG"
):
    if not cfg:
        logging.config.dictConfig(LOGGING_DEFAULT_CONFIG)
    else:
        logging.config.dictConfig(cfg)

    logger = logger or logging.getLogger()

    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(getattr(logging, log_level))
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()
            sh.setLevel(getattr(logging, log_level))
            logger.addHandler(sh)

    return logger


logger = configure_logger(
    log_file=log_file, console=no_console_log, log_level=log_level
)

HERE1 = op.dirname(op.abspath(__file__))
lib_path = op.join(HERE1, "..")
sys.path.append(lib_path)


HOUSING_PATH = op.join(HERE, "..", "data", "raw")

housing_prepared, housing_labels = hlb.load_train_data(project_path=HERE)
logger.info("Splitting training data into X_train and y_train.")

# hlb.lin_reg(housing_prepared, housing_labels)

final_model = hlb.random_forest(housing_prepared, housing_labels)
logger.info(f"Best model trained: {final_model}")


with open(path + "/model_pickle", "wb") as f:
    pickle.dump(final_model, f)
logger.info("Model pickle stored in artifacts.")

# logs added