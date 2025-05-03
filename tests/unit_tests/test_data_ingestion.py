import os
import pytest
import pandas as pd
from housinglib import fetch_housing_data, load_housing_data

@pytest.fixture
def tmp_data_path(tmp_path):
    return tmp_path / "data"

def test_fetch_housing_data(tmp_data_path):
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
    fetch_housing_data(url, tmp_data_path)
    assert os.path.exists(tmp_data_path / "housing.tgz")

def test_load_housing_data(tmp_data_path):
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
    fetch_housing_data(url, tmp_data_path)
    df = load_housing_data(tmp_data_path)
    assert isinstance(df, pd.DataFrame)
    assert "median_house_value" in df.columns
