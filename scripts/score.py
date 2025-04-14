
import argparse
import json
import logging
import logging.config
import os
import os.path as op
import pickle

import housinglib as hlb

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="data folder path")
parser.add_argument("--model_path", nargs="?")
parser.add_argument("--data_path", nargs="?")
parser.add_argument("--res_path", nargs="?")
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
        os.path.dirname(os.path.abspath(__file__)), "..", "logs", "score.log"
    )
else:
    log_file = args.log_path

log_dir = os.path.dirname(log_file)
os.makedirs(log_dir, exist_ok=True)

if args.no_console_log is None:
    no_console_log = True
else:
    no_console_log = args.no_console_log

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


HERE = op.dirname(op.abspath(__file__))

logger = configure_logger(
    log_file=log_file, console=no_console_log, log_level=log_level
)

if args.model_path is None:
    path = op.join(HERE, "..", "artifacts")
    with open(path + "/model_pickle", "rb") as f:
        final_model = pickle.load(f)
else:
    with open(args.model_path, "rb") as f:
        final_model = pickle.load(f)

if args.res_path is None:
    path3 = op.join(HERE, "..", "artifacts")
else:
    path3 = args.res_path

if args.data_path is None:
    path2 = op.join(HERE, "..", "data", "processed")
else:
    path2 = args.data_path


test_data, imputer = hlb.load_test_data(project_path=HERE)
logger.info("Loaded test data.")

final_mse, final_rmse = hlb.model_score(final_model, test_data, imputer)
logger.info("MSE and RMSE calculated.")

results = {"Mean Square Error": final_mse, "Root mean square error": final_rmse}

with open(path3 + "/results.txt", "w") as convert_file:
    convert_file.write(json.dumps(results))
logger.info("Results are stored in the artifacts folder.")

# logs added
