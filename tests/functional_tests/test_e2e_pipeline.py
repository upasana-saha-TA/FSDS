from housinglib import (
    load_data,
    data_prep,
    load_train_data,
    lin_reg,
    load_test_data,
    model_score,
)

def test_end_to_end(tmp_path):
    raw_path = tmp_path / "raw"
    project_path = tmp_path / "project"
    raw_path.mkdir(parents=True, exist_ok=True)
    project_path.mkdir(parents=True, exist_ok=True)

    print(f"Raw path: {raw_path}")
    print(f"Project path: {project_path}")

    # Step 1: Load & prepare data
    load_data(raw_path)
    data_prep(raw_path, project_path)

    # Step 2: Train and score
    X_train, y_train = load_train_data(project_path)
    model = lin_reg(X_train, y_train)

    test_data, imputer = load_test_data(project_path)
    mse, rmse = model_score(model, test_data, imputer)

    assert mse > 0
    assert rmse > 0