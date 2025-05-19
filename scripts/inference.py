import logging
import logging.config
import os
import os.path as op
import sys
import mlflow
import housinglib as hlb

LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s "
                      "- %(funcName)s:%(lineno)d - %(message)s",
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
        for hdlr in logger.handlers[:]:
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


def run(path=None, log_path=None, log_level="DEBUG", no_console_log=True):
    HERE = op.dirname(op.abspath(__file__))
    HOUSING_PATH = path or op.join(HERE, "..", "data", "raw")
    log_file = log_path or os.path.join(HERE, "..", "logs", "inference_logs.log")

    os.makedirs(op.dirname(log_file), exist_ok=True)
    logger = configure_logger(log_file=log_file, console=not no_console_log, log_level=log_level)

    logger.info("Starting inference...")

    model_uri = os.getenv("MODEL_URI", "runs:/12c33291b29e439cb4f48199300a0171/model")
    logger.info(f"Loading model from: {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    try:
        hlb.load_data(HOUSING_PATH)
        hlb.data_prep(HOUSING_PATH, project_path=HERE)
        input_data = hlb.load_test_data(HOUSING_PATH, project_path=HERE)
        logger.info("Input data prepared successfully.")
    except Exception as e:
        logger.error(f"Failed to prepare input data: {e}")
        return

    try:
        predictions = model.predict(input_data)
        logger.info("Inference completed.")
        logger.info(f"Predictions: {predictions.tolist()}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on housing data")
    parser.add_argument("--path", nargs="?", default=None)
    parser.add_argument("--log_level", nargs="?", default="DEBUG")
    parser.add_argument("--log_path", nargs="?", default=None)
    parser.add_argument("--no_console_log", nargs="?", default="True")

    args = parser.parse_args()
    run(
        path=args.path,
        log_path=args.log_path,
        log_level=args.log_level,
        no_console_log=args.no_console_log.lower() == "true",
    )
