import argparse
import json
import logging
import logging.config
import os
import os.path as op
import pickle
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

def run(model_path=None, data_path=None, res_path=None, log_path=None, log_level="DEBUG", no_console_log=True, use_mlflow=False, nested=False):
    HERE = op.dirname(op.abspath(__file__))

    model_path = model_path or op.join(HERE, "..", "artifacts", "model_pickle")
    res_path = res_path or op.join(HERE, "..", "artifacts")
    data_path = data_path or op.join(HERE, "..", "data", "processed")
    log_path = log_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "logs", "score.log"
    )

    os.makedirs(op.dirname(log_path), exist_ok=True)
    os.makedirs(res_path, exist_ok=True)

    logger = configure_logger(log_file=log_path, console=True, log_level=log_level)
    logger.info("Starting model scoring...")

    with open(model_path, "rb") as f:
        final_model = pickle.load(f)
    logger.info("Loaded trained model.")

    test_data, imputer = hlb.load_test_data(project_path=HERE)
    logger.info("Loaded test data.")

    final_mse, final_rmse = hlb.model_score(final_model, test_data, imputer)
    logger.info(f"Evaluation completed. MSE: {final_mse}, RMSE: {final_rmse}")

    results = {"Mean Square Error": final_mse, "Root mean square error": final_rmse}
    results_path = os.path.join(res_path, "results.txt")
    with open(results_path, "w") as f:
        f.write(json.dumps(results))
    logger.info(f"Results saved to {results_path}")

    if use_mlflow:
        with mlflow.start_run(nested=nested, run_name="Model Scoring") as scoring_run:
            mlflow.log_param("model_path", model_path)
            mlflow.log_param("data_path", data_path)
            mlflow.log_metric("MSE", final_mse)
            mlflow.log_metric("RMSE", final_rmse)
            mlflow.log_artifact(results_path, artifact_path="metrics")
            logger.info(f"Scoring run ID: {scoring_run.info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the trained model")
    parser.add_argument("--model_path", nargs="?", default=None)
    parser.add_argument("--data_path", nargs="?", default=None)
    parser.add_argument("--res_path", nargs="?", default=None)
    parser.add_argument("--log_level", nargs="?", default="DEBUG")
    parser.add_argument("--log_path", nargs="?", default=None)
    parser.add_argument("--no_console_log", nargs="?", default="True")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument("--mlflow-nested", action="store_true", help="Enable nested MLflow run")

    args = parser.parse_args()
    run(
        model_path=args.model_path,
        data_path=args.data_path,
        res_path=args.res_path,
        log_path=args.log_path,
        log_level=args.log_level,
        no_console_log=args.no_console_log.lower() == "true",
        use_mlflow=args.mlflow,
        nested=args.mlflow_nested,
    )
