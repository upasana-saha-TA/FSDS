import argparse
import logging
import logging.config
import os
import os.path as op
import pickle
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

def run(input_path=None, output_path=None, log_path=None, log_level="DEBUG", no_console_log=True, use_mlflow=False, nested=False):
    HERE = op.dirname(op.abspath(__file__))
    HOUSING_PATH = input_path or op.join(HERE, "..", "data", "raw")
    ARTIFACT_PATH = output_path or op.join(HERE, "..", "artifacts")
    LOG_PATH = log_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "logs", "train.log"
    )

    os.makedirs(op.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(ARTIFACT_PATH, exist_ok=True)

    logger = configure_logger(log_file=LOG_PATH, console=True, log_level=log_level)
    logger.info("Starting model training...")

    def train_model():
        housing_prepared, housing_labels = hlb.load_train_data(project_path=HERE)
        logger.info("Splitting training data into X_train and y_train.")

        final_model = hlb.random_forest(housing_prepared, housing_labels)
        logger.info(f"Best model trained: {final_model}.")

        with open(op.join(ARTIFACT_PATH, "model_pickle"), "wb") as f:
            pickle.dump(final_model, f)
        logger.info("Model pickle stored in artifacts.")

        return final_model

    if use_mlflow:
        with mlflow.start_run(nested=nested, run_name="Model Training") as training_run:
            mlflow.log_param("input_path", HOUSING_PATH)
            mlflow.log_param("output_path", ARTIFACT_PATH)
            mlflow.log_param("model_type", "RandomForest")
            logger.info(f"Training run ID: {training_run.info.run_id}")
            final_model = train_model()
            mlflow.sklearn.log_model(final_model, "model")
            mlflow.log_artifact(op.join(ARTIFACT_PATH, "model_pickle"), artifact_path="model")
    else:
        final_model = train_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train housing model")
    parser.add_argument("--input_path", nargs="?", default=None)
    parser.add_argument("--output_path", nargs="?", default=None)
    parser.add_argument("--log_level", nargs="?", default="DEBUG")
    parser.add_argument("--log_path", nargs="?", default=None)
    parser.add_argument("--no_console_log", nargs="?", default="True")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument("--mlflow-nested", action="store_true", help="Enable nested MLflow run")

    args = parser.parse_args()
    run(
        input_path=args.input_path,
        output_path=args.output_path,
        log_path=args.log_path,
        log_level=args.log_level,
        no_console_log=args.no_console_log.lower() == "true",
        use_mlflow=args.mlflow,
        nested=args.mlflow_nested,
    )
