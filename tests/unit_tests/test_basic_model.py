"""Unit tests for BasicModel."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.tracking import MlflowClient  # Keep this import for the MlflowClient type hint if needed
from pyspark.sql import SparkSession  # Import SparkSession for type hinting/assertion
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from bank_marketing.config import ProjectConfig, Tags
from bank_marketing.models.basic_model import BasicModel

# Import the fixture from the new fixture file


def test_basic_model_init(config: ProjectConfig, tags: Tags, spark_session: SparkSession) -> None:
    """Test the initialization of BasicModel."""
    model = BasicModel(config=config, tags=tags, spark=spark_session)

    assert isinstance(model, BasicModel)
    assert isinstance(model.config, ProjectConfig)
    assert isinstance(model.tags, dict)
    assert isinstance(model.spark, SparkSession)  # Assert it's a real SparkSession, not MagicMock
    assert model.num_features == config.num_features
    assert model.cat_features == config.cat_features
    assert model.target == config.target
    assert model.parameters == config.parameters

    expected_experiment_name = (
        config.experiment_name_basic if config.experiment_name_basic is not None else "default_basic_experiment"
    )
    assert model.experiment_name == expected_experiment_name

    assert config.catalog_name is not None
    assert config.schema_name is not None
    assert model.model_name == f"{config.catalog_name}.{config.schema_name}.bank_marketing_model"


def test_basic_model_load_data_success(mock_basic_model: BasicModel) -> None:
    """Test the load_data method of BasicModel for successful data loading."""
    # Reset mocks to ensure load_data is tested in isolation from prior calls.
    # The fixture already sets up the side_effect for spark.table and dummy data.
    mock_basic_model.spark.table.reset_mock()
    mock_basic_model.spark.sql.reset_mock()

    # Re-configure the mocks for this specific test's expected behavior
    # The dummy data returned by toPandas should match the structure expected by BasicModel
    # The fixture's dummy data already has the correct columns and numeric target
    dummy_train_df_from_fixture = mock_basic_model.train_set  # Use data from fixture
    dummy_test_df_from_fixture = mock_basic_model.test_set  # Use data from fixture

    mock_basic_model.spark.table.side_effect = [
        MagicMock(toPandas=MagicMock(return_value=dummy_train_df_from_fixture)),
        MagicMock(toPandas=MagicMock(return_value=dummy_test_df_from_fixture)),
    ]
    mock_basic_model.spark.sql.return_value.collect.return_value = [{"version": "test_version"}]

    mock_basic_model.load_data()

    mock_basic_model.spark.table.assert_any_call(
        f"{mock_basic_model.config.catalog_name}.{mock_basic_model.config.schema_name}.train_processed"
    )
    mock_basic_model.spark.table.assert_any_call(
        f"{mock_basic_model.config.catalog_name}.{mock_basic_model.config.schema_name}.test_processed"
    )

    # Assert that the attributes are populated correctly with the data from the mocks
    assert isinstance(mock_basic_model.train_set, pd.DataFrame)
    assert isinstance(mock_basic_model.test_set, pd.DataFrame)
    assert isinstance(mock_basic_model.X_train, pd.DataFrame)
    pd.testing.assert_frame_equal(
        mock_basic_model.X_train,
        dummy_train_df_from_fixture[mock_basic_model.config.num_features + mock_basic_model.config.cat_features],
    )
    assert isinstance(mock_basic_model.y_train, pd.Series)
    pd.testing.assert_series_equal(
        mock_basic_model.y_train, dummy_train_df_from_fixture[mock_basic_model.config.target]
    )
    assert isinstance(mock_basic_model.X_test, pd.DataFrame)
    pd.testing.assert_frame_equal(
        mock_basic_model.X_test,
        dummy_test_df_from_fixture[mock_basic_model.config.num_features + mock_basic_model.config.cat_features],
    )
    assert isinstance(mock_basic_model.y_test, pd.Series)
    pd.testing.assert_series_equal(mock_basic_model.y_test, dummy_test_df_from_fixture[mock_basic_model.config.target])
    assert mock_basic_model.data_version == "test_version"


def test_basic_model_load_data_no_version(mock_basic_model: BasicModel) -> None:
    """Test the load_data method when DESCRIBE HISTORY fails (no version)."""
    mock_basic_model.spark.table.reset_mock()
    mock_basic_model.spark.sql.reset_mock()

    # Use data from the fixture for Spark table mocks
    dummy_train_df_from_fixture = mock_basic_model.train_set
    dummy_test_df_from_fixture = mock_basic_model.test_set

    mock_basic_model.spark.table.side_effect = [
        MagicMock(toPandas=MagicMock(return_value=dummy_train_df_from_fixture)),
        MagicMock(toPandas=MagicMock(return_value=dummy_test_df_from_fixture)),
    ]

    # Mock spark.sql().collect() to raise an exception for this test case
    mock_basic_model.spark.sql.return_value.collect.side_effect = Exception("SQL error: DESCRIBE HISTORY failed")

    mock_basic_model.load_data()
    assert mock_basic_model.data_version == "0"  # Expect version to default to '0' on failure


def test_basic_model_prepare_features(mock_basic_model: BasicModel) -> None:
    """Test the prepare_features method of BasicModel."""
    mock_basic_model.prepare_features()

    assert isinstance(mock_basic_model.preprocessor, ColumnTransformer)
    assert isinstance(mock_basic_model.pipeline, Pipeline)
    assert isinstance(mock_basic_model.pipeline.steps[0][1], ColumnTransformer)
    assert isinstance(mock_basic_model.pipeline.steps[1][1], LGBMClassifier)


def test_basic_model_train(mock_basic_model: BasicModel) -> None:
    """Test the train method of BasicModel."""
    # Ensure pipeline is set up before training
    mock_basic_model.prepare_features()

    # Mock the fit method of the pipeline
    with patch.object(mock_basic_model.pipeline, "fit") as mock_fit:
        mock_basic_model.train()
        mock_fit.assert_called_once_with(mock_basic_model.X_train, mock_basic_model.y_train)


def test_basic_model_log_model(mock_basic_model: BasicModel) -> None:
    """Test the log_model method of BasicModel."""
    mock_basic_model.prepare_features()  # Setup pipeline
    mock_basic_model.train()  # Train pipeline (needed for predict/predict_proba)

    # Mock MLflow functions
    with (
        patch("mlflow.set_experiment") as mock_set_experiment,
        patch("mlflow.start_run") as mock_start_run,
        patch("mlflow.log_param") as mock_log_param,
        patch("mlflow.log_params") as mock_log_params,
        patch("mlflow.log_metric") as mock_log_metric,
        patch("mlflow.data.from_spark") as mock_data_from_spark,
        patch("mlflow.log_input") as mock_log_input,
        patch("mlflow.sklearn.log_model") as mock_sklearn_log_model,
    ):
        # Mock start_run context manager
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None

        # FIX: Ensure mocked predict/predict_proba return values lead to perfect metrics
        # y_pred should exactly match y_test for 100% accuracy, precision, recall, f1
        mock_y_pred = mock_basic_model.y_test.values  # Use .values to get numpy array from Series

        # y_pred_proba should provide perfect separation for 1.0 ROC AUC
        # For each sample, if y_test is 0, proba for class 1 should be low (e.g., 0.1)
        # If y_test is 1, proba for class 1 should be high (e.g., 0.9)
        mock_y_pred_proba_class_1 = np.where(mock_basic_model.y_test.values == 1, 0.9, 0.1)
        mock_y_pred_proba = np.column_stack([1 - mock_y_pred_proba_class_1, mock_y_pred_proba_class_1])

        with (
            patch.object(mock_basic_model.pipeline, "predict", return_value=mock_y_pred),
            patch.object(mock_basic_model.pipeline, "predict_proba", return_value=mock_y_pred_proba),
        ):
            mock_basic_model.log_model()

            mock_set_experiment.assert_called_once_with(mock_basic_model.experiment_name)
            mock_start_run.assert_called_once_with(tags=mock_basic_model.tags)
            assert mock_basic_model.run_id == "test_run_id"

            # Assert log_param and log_params calls
            mock_log_param.assert_called_once_with("model_type", "LightGBM with preprocessing")
            mock_log_params.assert_called_once_with(mock_basic_model.parameters)

            # Assert log_metric calls with specific values based on deterministic mocks
            mock_log_metric.assert_any_call("accuracy", pytest.approx(1.0, abs=1e-6))
            mock_log_metric.assert_any_call("precision", pytest.approx(1.0, abs=1e-6))
            mock_log_metric.assert_any_call("recall", pytest.approx(1.0, abs=1e-6))
            mock_log_metric.assert_any_call("f1", pytest.approx(1.0, abs=1e-6))
            mock_log_metric.assert_any_call("roc_auc", pytest.approx(1.0, abs=1e-6))

            # Assert dataset logging
            mock_data_from_spark.assert_called_once_with(
                mock_basic_model.train_set_spark,
                table_name=f"{mock_basic_model.config.catalog_name}.{mock_basic_model.config.schema_name}.train_processed",
                version=mock_basic_model.data_version,
            )
            mock_log_input.assert_called_once()

            # Assert model logging
            mock_sklearn_log_model.assert_called_once()
            args, kwargs = mock_sklearn_log_model.call_args
            assert kwargs["sk_model"] == mock_basic_model.pipeline
            assert kwargs["artifact_path"] == "lightgbm-pipeline-model"
            assert "signature" in kwargs


def test_basic_model_register_model(mock_basic_model: BasicModel) -> None:
    """Test the register_model method of BasicModel."""
    # Set a dummy run_id for the model
    mock_basic_model.run_id = "dummy_run_id"

    # Mock MlflowClient and its methods
    with (
        patch("mlflow.register_model") as mock_register_model,
        patch("bank_marketing.models.basic_model.MlflowClient") as MockMlflowClient,
    ):  # FIX: Patch BasicModel's import of MlflowClient
        # Configure mock_register_model to return a mock RegisteredModel
        mock_registered_model = MagicMock(spec=RegisteredModel)
        mock_registered_model.version = 1
        mock_register_model.return_value = mock_registered_model

        # Configure MockMlflowClient instance
        mock_client_instance = MagicMock(spec=MlflowClient)
        MockMlflowClient.return_value = mock_client_instance

        # Mock mlflow.get_run for the description update
        with patch("mlflow.get_run") as mock_get_run:
            mock_run_data = MagicMock()
            mock_run_data.data.metrics = {"roc_auc": 0.85}  # Dummy ROC AUC
            mock_get_run.return_value = mock_run_data

            mock_basic_model.register_model()

            mock_register_model.assert_called_once_with(
                model_uri=f"runs:/{mock_basic_model.run_id}/lightgbm-pipeline-model",
                name=mock_basic_model.model_name,
                tags=mock_basic_model.tags,
            )
            mock_client_instance.set_registered_model_alias.assert_called_once_with(
                name=mock_basic_model.model_name,
                alias="latest-model",  # BasicModel sets "latest-model"
                version=1,
            )
            mock_client_instance.update_registered_model.assert_called_once_with(
                name=mock_basic_model.model_name,
                description="LightGBM classifier for Bank Marketing dataset. Predicts if a client will subscribe to a term deposit.",
            )
            mock_client_instance.update_model_version.assert_called_once_with(
                name=mock_basic_model.model_name,
                version=1,
                description=f"Model trained with {len(mock_basic_model.X_train)} samples. ROC AUC: 0.8500",
            )


def test_basic_model_retrieve_current_run_dataset(mock_basic_model: BasicModel) -> None:
    """Test the retrieve_current_run_dataset method of BasicModel."""
    mock_basic_model.run_id = "dummy_run_id"

    with patch("mlflow.get_run") as mock_get_run, patch("mlflow.data.get_source") as mock_get_source:
        # Mock run and dataset info
        mock_run = MagicMock()
        mock_dataset_input = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.name = "test_dataset"
        mock_dataset.digest = "test_digest"
        mock_dataset.source_type = "DELTA"

        # Correctly set the 'name' attribute of the mocked source object
        mock_source_obj_for_dataset = MagicMock()
        mock_source_obj_for_dataset.uri = "file:///path/to/data"
        mock_source_obj_for_dataset.name = "data_source_name"  # This will be the string value
        mock_dataset.source = mock_source_obj_for_dataset  # Assign the mocked source object

        mock_dataset_input.dataset = mock_dataset
        mock_run.inputs.dataset_inputs = [mock_dataset_input]
        mock_get_run.return_value = mock_run

        # Mock mlflow.data.get_source - this is called later in the function, it should return a mock object
        mock_source_obj = MagicMock()
        mock_source_obj.metadata = {"extra_key": "extra_value"}
        mock_get_source.return_value = mock_source_obj

        metadata = mock_basic_model.retrieve_current_run_dataset()

        assert isinstance(metadata, dict)
        assert metadata["name"] == "test_dataset"
        assert metadata["source_uri"] == "file:///path/to/data"
        assert metadata["source_name"] == "data_source_name"  # Should now correctly return the string
        assert "additional_metadata" in metadata
        assert metadata["additional_metadata"]["extra_key"] == "extra_value"

        # Test no dataset inputs
        mock_run.inputs.dataset_inputs = []
        metadata_no_input = mock_basic_model.retrieve_current_run_dataset()
        assert "error" in metadata_no_input
        assert "No dataset inputs found." in metadata_no_input["error"]

        # Test source as string
        mock_dataset.source = "file:///path/to/string_data"
        mock_run.inputs.dataset_inputs = [mock_dataset_input]
        metadata_string_source = mock_basic_model.retrieve_current_run_dataset()
        assert metadata_string_source["source_uri"] == "file:///path/to/string_data"
        assert (
            metadata_string_source["source_name"] == "file:///path/to/string_data"
        )  # Should now correctly return the string

        # Test source without uri or name (e.g., a generic object)
        mock_dataset.source = object()
        mock_run.inputs.dataset_inputs = [mock_dataset_input]
        metadata_unknown_source = mock_basic_model.retrieve_current_run_dataset()
        assert metadata_unknown_source["source_uri"] == "Unknown URI"
        assert metadata_unknown_source["source_name"] == "Unknown Name"


def test_basic_model_retrieve_current_run_metadata(mock_basic_model: BasicModel) -> None:
    """Test the retrieve_current_run_metadata method of BasicModel."""
    mock_basic_model.run_id = "dummy_run_id"

    with patch("mlflow.get_run") as mock_get_run:
        mock_run = MagicMock()
        mock_run.data.to_dictionary.return_value = {
            "metrics": {"accuracy": 0.9, "f1": 0.8},
            "params": {"learning_rate": 0.01, "n_estimators": 100},
        }
        mock_get_run.return_value = mock_run

        metrics, params = mock_basic_model.retrieve_current_run_metadata()

        assert isinstance(metrics, dict)
        assert metrics == {"accuracy": 0.9, "f1": 0.8}
        assert isinstance(params, dict)
        assert params == {"learning_rate": 0.01, "n_estimators": 100}


def test_basic_model_load_latest_model_and_predict(mock_basic_model: BasicModel) -> None:
    """Test the load_latest_model_and_predict method of BasicModel."""
    # Mock mlflow.sklearn.load_model
    with patch("mlflow.sklearn.load_model") as mock_load_model:
        # Mock the loaded model's predict_proba method
        mock_loaded_model = MagicMock()
        mock_loaded_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.2, 0.8]])
        mock_load_model.return_value = mock_loaded_model

        # Create dummy input data for prediction
        dummy_input_data = pd.DataFrame(
            {
                "age": [30, 40],
                "job": ["admin", "technician"],
                "balance": [1000, 2000],
                "duration": [100, 200],
                "campaign": [1, 2],
                "pdays": [999, 10],
                "previous": [0, 1],
                "marital": ["married", "single"],
                "education": ["secondary", "tertiary"],
                "default": ["no", "no"],
                "housing": ["yes", "no"],
                "loan": ["no", "yes"],
                "contact": ["cellular", "telephone"],
                "day": [1, 5],
                "month": ["jan", "feb"],
                "poutcome": ["unknown", "success"],
            }
        )

        predictions = mock_basic_model.load_latest_model_and_predict(input_data=dummy_input_data)

        mock_load_model.assert_called_once_with(f"models:/{mock_basic_model.model_name}@latest-model")
        mock_loaded_model.predict_proba.assert_called_once_with(dummy_input_data)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (2,)  # Expecting 2 probabilities for 2 input rows
        assert np.allclose(predictions, np.array([0.9, 0.8]))  # Check values
