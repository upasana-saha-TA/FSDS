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

def run(path=None, log_path=None, log_level="DEBUG", no_console_log=True, use_mlflow=False, nested=False):
    HERE = op.dirname(op.abspath(__file__))
    HOUSING_PATH = path or op.join(HERE, "..", "data", "raw")
    log_file = log_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "logs", "ingest_data.log"
    )

    os.makedirs(op.dirname(log_file), exist_ok=True)
    logger = configure_logger(log_file=log_file, console=True, log_level=log_level)

    logger.info("Starting data ingestion process...")

    if use_mlflow:
        with mlflow.start_run(nested=nested, run_name="Data Ingestion") as ingestion_run:
            logger.info(f"Ingestion run ID: {ingestion_run.info.run_id}")
            mlflow.log_param("data_path", HOUSING_PATH)
            hlb.load_data(HOUSING_PATH)
            hlb.data_prep(HOUSING_PATH, project_path=HERE)
    else:
        hlb.load_data(HOUSING_PATH)
        hlb.data_prep(HOUSING_PATH, project_path=HERE)

    logger.info("Data ingestion and preparation completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest housing data")
    parser.add_argument("--path", nargs="?", default=None)
    parser.add_argument("--log_level", nargs="?", default="DEBUG")
    parser.add_argument("--log_path", nargs="?", default=None)
    parser.add_argument("--no_console_log", nargs="?", default="True")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument("--mlflow-nested", action="store_true", help="Enable nested MLflow run")

    args = parser.parse_args()
    run(
        path=args.path,
        log_path=args.log_path,
        log_level=args.log_level,
        no_console_log=args.no_console_log.lower() == "true",
        use_mlflow=args.mlflow,
        nested=args.mlflow_nested,
    )
