import mlflow
import logging
import ingest_data
import train
import score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    with mlflow.start_run(run_name="Parent_Run") as parent_run:
        logger.info(f"Parent MLflow run ID: {parent_run.info.run_id}")
        ingest_data.run(use_mlflow=True, nested=True)
        train.run(use_mlflow=True, nested=True)
        score.run(use_mlflow=True, nested=True)

