import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import pytest
from housinglib import load_data, data_prep, load_train_data,model_score, lin_reg, desc_tree, random_forest
from sklearn.metrics import mean_squared_error

@pytest.fixture(scope="module")
def setup_pipeline(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("data_scoring")
    raw_path = tmp_dir / "raw"
    raw_path = tmp_dir / "raw"
    project_path = tmp_dir / "project"
    raw_path.mkdir(parents=True, exist_ok=True)
    project_path.mkdir(parents=True, exist_ok=True)

    load_data(raw_path)
    data_prep(raw_path, project_path)

    return project_path

@pytest.fixture(scope="module")
def train_data(setup_pipeline):
    return load_train_data(setup_pipeline)  # returns (X_train, y_train)

def test_lin_reg_model(train_data):
    X_train, y_train = train_data
    model = lin_reg(X_train, y_train)
    assert hasattr(model, "predict")

def test_decision_tree_model(train_data):
    X_train, y_train = train_data
    model = desc_tree(X_train, y_train)
    assert hasattr(model, "predict")

def test_random_forest_model(train_data):
    X_train, y_train = train_data
    model = random_forest(X_train, y_train)
    assert hasattr(model, "predict")
