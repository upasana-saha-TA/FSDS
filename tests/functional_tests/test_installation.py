def test_package_installation():
    try:
        import housinglib
    except ImportError:
        assert False, "housinglib package is not installed or not found in PYTHONPATH"

    # Check that key functions exist
    expected_functions = [
        "fetch_housing_data", "load_housing_data", "load_data",
        "data_prep", "load_train_data", "lin_reg", "desc_tree",
        "random_forest", "load_test_data", "model_score"
    ]

    for func in expected_functions:
        assert hasattr(housinglib, func), f"Function '{func}' is missing in housinglib"