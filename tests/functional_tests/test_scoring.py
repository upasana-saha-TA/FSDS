import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import pytest
from housinglib import load_data, data_prep, load_test_data, model_score, lin_reg, load_train_data

@pytest.fixture(scope="module")
def setup_pipeline(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("data_scoring")
    raw_path = tmp_dir / "raw"
    project_path = tmp_dir / "project"
    raw_path.mkdir(parents=True, exist_ok=True)
    project_path.mkdir(parents=True, exist_ok=True)

    load_data(raw_path)
    data_prep(raw_path, project_path)

    return project_path

@pytest.fixture(scope="module")
def trained_model(setup_pipeline):
    X_train, y_train = load_train_data(setup_pipeline)
    model = lin_reg(X_train, y_train)
    return model

def test_model_score(trained_model, setup_pipeline):
    test_data, imputer = load_test_data(setup_pipeline)
    mse, rmse = model_score(trained_model, test_data, imputer)
    assert mse > 0
    assert rmse > 0