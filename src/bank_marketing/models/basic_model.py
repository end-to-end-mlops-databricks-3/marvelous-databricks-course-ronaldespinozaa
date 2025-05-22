"""Basic model implementation for Bank Marketing dataset.

This module provides a BasicModel class that handles training, logging, and registering
a LightGBM classifier model for predicting if a client will subscribe to a term deposit.
"""

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_marketing.config import ProjectConfig, Tags


class BasicModel:
    """A basic model class for bank marketing prediction using LightGBM.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration.

        Args:
            config: Project configuration object
            tags: Tags object
            spark: SparkSession object

        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.bank_marketing_model"
        self.tags = tags.dict()

    def load_data(self) -> None:
        """Load training and testing data from processed tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_processed")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_processed").toPandas()

        # Get data version (could be implemented with DESCRIBE HISTORY)
        try:
            history = self.spark.sql(f"DESCRIBE HISTORY {self.catalog_name}.{self.schema_name}.train_processed LIMIT 1")
            self.data_version = str(history.collect()[0]["version"])
        except Exception as e:
            logger.warning(f"Could not retrieve table version: {e}")
            self.data_version = "0"  # Default if version can't be retrieved

        # Split features and target
        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]

        logger.info(f"✅ Data successfully loaded. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")

    def prepare_features(self) -> None:
        """Encode categorical features and define a preprocessing pipeline.

        Creates a ColumnTransformer for one-hot encoding categorical features and scaling numerical features.
        Constructs a pipeline combining preprocessing and LightGBM classification model.
        """
        logger.info("🔄 Defining preprocessing pipeline...")

        # Create transformers for different feature types
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

        # Combine transformers in a column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.num_features),
                ("cat", categorical_transformer, self.cat_features),
            ]
        )

        # Create the full pipeline with preprocessor and classifier
        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("classifier", LGBMClassifier(**self.parameters))]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self) -> None:
        """Train the model."""
        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)
        logger.info("✅ Model training completed.")

    def log_model(self) -> None:
        """Log the model using MLflow."""
        logger.info("📝 Logging model with MLflow...")
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            # Make predictions
            y_pred = self.pipeline.predict(self.X_test)
            y_pred_proba = self.pipeline.predict_proba(self.X_test)[:, 1]  # Probability of class 1

            # Evaluate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            logger.info(f"📊 Accuracy: {accuracy:.4f}")
            logger.info(f"📊 Precision: {precision:.4f}")
            logger.info(f"📊 Recall: {recall:.4f}")
            logger.info(f"📊 F1 Score: {f1:.4f}")
            logger.info(f"📊 ROC AUC: {roc_auc:.4f}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            # Infer model signature
            signature = infer_signature(model_input=self.X_train, model_output=y_pred_proba)

            # Log the dataset used for training
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_processed",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")

            # Log model with sklearn flavor
            mlflow.sklearn.log_model(
                sk_model=self.pipeline, artifact_path="lightgbm-pipeline-model", signature=signature
            )

            logger.info(f"✅ Model logged with run_id: {self.run_id}")

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in Unity Catalog...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}")

        latest_version = registered_model.version

        # Set aliases for model version
        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )

        # Add model description
        client.update_registered_model(
            name=self.model_name,
            description="LightGBM classifier for Bank Marketing dataset. Predicts if a client will subscribe to a term deposit.",
        )

        # Add version description
        client.update_model_version(
            name=self.model_name,
            version=latest_version,
            description=f"Model trained with {len(self.X_train)} samples. ROC AUC: {mlflow.get_run(self.run_id).data.metrics['roc_auc']:.4f}",
        )

    def retrieve_current_run_dataset(self) -> dict:
        """Retrieve MLflow run dataset metadata.

        Returns:
            Dictionary with dataset metadata

        """
        try:
            run = mlflow.get_run(self.run_id)
            if not run.inputs or not run.inputs.dataset_inputs:
                logger.warning("⚠️ No dataset inputs found for this run.")
                return {"error": "No dataset inputs found."}

            dataset_info = run.inputs.dataset_inputs[0].dataset

            # Initialize source_uri and source_name
            source_uri = ""
            source_name = ""

            # Check if source is a string (direct URI) or an object with a URI attribute
            if isinstance(dataset_info.source, str):
                source_uri = dataset_info.source
                source_name = dataset_info.source  # Often the URI itself can serve as a simple name
            elif hasattr(dataset_info.source, "uri"):
                source_uri = dataset_info.source.uri
                # Try to get a more specific name if available, otherwise use the URI
                source_name = getattr(dataset_info.source, "name", dataset_info.source.uri)
            else:
                logger.warning(f"⚠️ Could not determine source URI or name for type: {type(dataset_info.source)}")
                source_uri = "Unknown URI"
                source_name = "Unknown Name"

            dataset_metadata = {
                "name": dataset_info.name,
                "digest": dataset_info.digest,
                "source_type": dataset_info.source_type,
                "source_uri": source_uri,
                "source_name": source_name,
                "table_name": dataset_info.name,  # Often the dataset name itself acts as the table name
                "version": dataset_info.digest,  # Digest is a common identifier for version
            }

            # Use mlflow.data.get_source for potential additional metadata if applicable
            # This is generally robust, but its 'metadata' attribute might not always be populated.
            dataset_source_obj = mlflow.data.get_source(dataset_info)
            if hasattr(dataset_source_obj, "metadata") and isinstance(dataset_source_obj.metadata, dict):
                # Add any specific metadata from the structured source object
                dataset_metadata["additional_metadata"] = dataset_source_obj.metadata

            logger.info("✅ Dataset metadata loaded")
            return dataset_metadata
        except Exception as e:
            logger.warning(f"⚠️ Could not retrieve dataset metadata: {e}")
            return {"error": str(e)}

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve MLflow run metadata.

        Returns:
            Tuple containing metrics and parameters dictionaries

        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("✅ Run metadata loaded")
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Load the latest model from MLflow (alias=latest-model) and make predictions.

        Args:
            input_data: Pandas DataFrame containing input features for prediction.

        Returns:
            NumPy array with predicted probabilities.

        """
        logger.info("🔄 Loading model from MLflow alias 'latest-model'...")

        model_uri = f"models:/{self.model_name}@latest-model"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info("✅ Model successfully loaded")

        # Make predictions (return probabilities of positive class)
        predictions = model.predict_proba(input_data)[:, 1]

        return predictions
