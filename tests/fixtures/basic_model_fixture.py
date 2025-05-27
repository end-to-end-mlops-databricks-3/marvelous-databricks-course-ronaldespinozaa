"""Fixtures for BasicModel unit tests."""  # FIX: Added module docstring

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from bank_marketing.config import ProjectConfig, Tags
from bank_marketing.models.basic_model import BasicModel


@pytest.fixture
def mock_basic_model(config: ProjectConfig, tags: Tags, spark_session: SparkSession) -> BasicModel:
    """Fixture that provides a BasicModel instance with mocked Spark interactions.

    Dynamically generates dummy data based on the ProjectConfig's features
    to ensure consistency and avoid KeyErrors.
    """
    # Create a BasicModel instance
    model = BasicModel(config=config, tags=tags, spark=spark_session)

    # --- Dynamically create dummy pandas DataFrames based on config features ---
    num_rows_train = 10
    num_rows_test = 5

    # Define a common set of categories that might be expected in bank marketing data
    # This ensures consistency even if not all are in config.cat_features
    common_categories = {
        "job": ["admin.", "blue-collar", "management", "technician", "services", "retired", "unknown"],
        "marital": ["married", "single", "divorced"],
        "education": ["secondary", "tertiary", "primary", "unknown"],
        "default": ["no", "yes"],
        "housing": ["no", "yes"],
        "loan": ["no", "yes"],
        "contact": ["cellular", "telephone", "unknown"],
        "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        "poutcome": ["unknown", "success", "failure", "other"],
        # Add other categorical features and their possible values as needed
    }

    dummy_train_data = {}
    for col in model.num_features:
        dummy_train_data[col] = np.random.rand(num_rows_train) * 1000  # Example numeric data
    for col in model.cat_features:
        # Use common_categories if available, otherwise default to generic ones
        categories_for_col = common_categories.get(col, ["cat_A", "cat_B", "cat_C", "unknown"])
        dummy_train_data[col] = np.random.choice(categories_for_col, num_rows_train)

    # Ensure target column is present and numeric (0/1) for training
    dummy_train_data[model.target] = np.random.choice([0, 1], num_rows_train)
    dummy_train_df = pd.DataFrame(dummy_train_data)

    dummy_test_data = {}
    for col in model.num_features:
        dummy_test_data[col] = np.random.rand(num_rows_test) * 1000
    for col in model.cat_features:
        categories_for_col = common_categories.get(col, ["cat_A", "cat_B", "cat_C", "unknown"])
        dummy_test_data[col] = np.random.choice(categories_for_col, num_rows_test)

    # FIX: Make dummy_test_data[model.target] deterministic for consistent metric calculation
    dummy_test_data[model.target] = np.array([0, 1, 0, 1, 0])[:num_rows_test]  # Explicitly set for 5 rows
    dummy_test_df = pd.DataFrame(dummy_test_data)

    # --- End dynamic dummy data creation ---

    # Mock toPandas() method for Spark DataFrames
    mock_spark_df_train = MagicMock()
    mock_spark_df_train.toPandas.return_value = dummy_train_df
    mock_spark_df_test = MagicMock()
    mock_spark_df_test.toPandas.return_value = dummy_test_df

    # Mock spark.table method to return the mocked Spark DataFrames
    mock_spark = MagicMock(spec=SparkSession)
    mock_spark.table.side_effect = [mock_spark_df_train, mock_spark_df_test]

    # Mock spark.sql().collect() for data_version retrieval
    mock_spark.sql.return_value.collect.return_value = [{"version": "123"}]

    # Assign the mocked spark session to the model instance
    model.spark = mock_spark

    # Manually set attributes needed for prepare_features, train, etc.
    # These are derived from the dynamically created dummy_train_df/dummy_test_df
    model.train_set = dummy_train_df
    model.test_set = dummy_test_df
    model.X_train = dummy_train_df[model.num_features + model.cat_features]
    model.y_train = dummy_train_df[model.target]
    model.X_test = dummy_test_df[model.num_features + model.cat_features]
    model.y_test = dummy_test_df[model.target]

    # FIX: Add Spark DataFrame mocks and data_version as log_model expects them
    model.train_set_spark = MagicMock()
    model.test_set_spark = MagicMock()
    model.data_version = "test_version_from_fixture"  # Assign a dummy version

    yield model
