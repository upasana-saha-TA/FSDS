from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

import logging
import logging.config
import os
import os.path as op
import sys
import housinglib as hlb
import json

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

def data_drift(path=None, log_path=None, log_level="DEBUG", no_console_log=True):
    HERE = op.dirname(op.abspath(__file__))
    HOUSING_PATH = path or op.join(HERE, "..", "data", "raw")
    TEST_PATH = path or op.join(HERE, "..", "data", "processed","test_set.csv")
    log_file = log_path or os.path.join(HERE, "..", "logs", "monitoring.log")

    os.makedirs(op.dirname(log_file), exist_ok=True)
    logger = configure_logger(log_file=log_file, console=not no_console_log, log_level=log_level)

    logger.info("Starting monitoring...")

    model_uri = os.getenv("MODEL_URI", "runs:/12c33291b29e439cb4f48199300a0171/model")
    logger.info(f"Loading model from: {model_uri}")

    try:
        hlb.load_data(HOUSING_PATH)
        hlb.data_prep(HOUSING_PATH, project_path=HERE)
        refer_data=hlb.load_housing_data(HOUSING_PATH)
        input_data,imputer = hlb.load_test_data(project_path=HERE)
        logger.info("Input data prepared successfully.")
    except Exception as e:
        logger.error(f"Failed to prepare input data: {e}")
        return

    # Run the report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=refer_data, current_data=input_data)

    # Save HTML report
    report_path = os.path.join(HERE, "..", "reports", "drift_report.html")
    report.save_html(report_path)

    # Extract drift result
    results = report.as_dict()

    # Check and log if drift detected
    try:
    # Loop through metrics to find "DatasetDriftMetric"
        for metric in results["metrics"]:
            if metric["metric"] == "DatasetDriftMetric":
                drift_result = metric["result"]
                drift_detected = drift_result["dataset_drift"]
                drift_score = drift_result["drift_share"]

                if drift_detected:
                    logger.warning(f"Drift detected! Drift share: {drift_score:.2f}")
                    exit(1)  # Optional: Exit if used in CI/CD
                else:
                    logger.info(f"No significant drift detected. Drift share: {drift_score:.2f}")
                break
        else:
            logger.error("DatasetDriftMetric not found in report results.")

    except Exception as e:
        logger.error(f"Error parsing drift results: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on housing data")
    parser.add_argument("--path", nargs="?", default=None)
    parser.add_argument("--log_level", nargs="?", default="DEBUG")
    parser.add_argument("--log_path", nargs="?", default=None)
    parser.add_argument("--no_console_log", nargs="?", default="False")

    args = parser.parse_args()
    data_drift(
        path=args.path,
        log_path=args.log_path,
        log_level=args.log_level,
        no_console_log=args.no_console_log.lower() == "true",
    )
