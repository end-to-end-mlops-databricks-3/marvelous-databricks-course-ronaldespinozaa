"""Custom model implementation for Bank Marketing dataset.

This module implements a custom model for the Bank Marketing dataset using MLflow pyfunc.
It demonstrates how to wrap a scikit-learn model in a custom Python function model
to add additional preprocessing and postprocessing steps.
"""

from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline  # Import Pipeline for type hinting
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_marketing.config import ProjectConfig, Tags


class BankMarketingModelWrapper(PythonModel):
    """Wrapper for the Bank Marketing model to use with MLflow pyfunc.

    This class wraps a scikit-learn pipeline and adds custom preprocessing
    and postprocessing steps.
    """

    def __init__(self, pipeline: Pipeline, threshold: float = 0.5) -> None:  # Changed Any to Pipeline
        """Initialize the wrapper.

        Args:
            pipeline: Scikit-learn pipeline
            threshold: Probability threshold for classification (default: 0.5)

        """
        self.pipeline = pipeline
        self.threshold = threshold

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame) -> dict:
        """Make predictions and format the output.

        This method demonstrates customizing the prediction output format.

        Args:
            context: MLflow model context
            model_input: DataFrame with model inputs

        Returns:
            Dictionary with predictions and probabilities

        """
        # Get probability predictions
        try:
            # Try to use predict_proba first (for classifiers)
            probabilities = self.pipeline.predict_proba(model_input)[:, 1]
            predictions = (probabilities >= self.threshold).astype(int)
        except AttributeError:
            # Fall back to regular predict (for non-probabilistic models)
            predictions = self.pipeline.predict(model_input)
            probabilities = None

        # If bank_marketing module is available, apply additional processing
        try:
            # This would come from your custom package
            from bank_marketing.utils.prediction_utils import adjust_predictions

            predictions = adjust_predictions(predictions)
            logger.info("✅ Applied custom prediction adjustment")
        except ImportError:
            logger.warning("⚠️ bank_marketing package not available, using raw predictions")

        # Create custom output format
        result = {
            "predictions": predictions.tolist(),
        }

        # Add probabilities if available
        if probabilities is not None:
            result["probabilities"] = probabilities.tolist()
            result["threshold"] = self.threshold

        # Add mean and confidence interval if applicable
        if len(predictions) > 1:
            result["mean_prediction"] = float(np.mean(predictions))
            result["positive_rate"] = float(np.mean(predictions == 1))

        return result


class CustomModel:
    """A custom model class for bank marketing prediction using MLflow pyfunc.

    This class demonstrates how to implement a custom model with MLflow pyfunc,
    including custom preprocessing, prediction logic, and model packaging.
    """

    def __init__(
        self, config: ProjectConfig, tags: Tags, spark: SparkSession, code_paths: list[str] | None = None
    ) -> None:
        """Initialize the model with project configuration.

        Args:
            config: Project configuration object
            tags: Tags object
            spark: SparkSession object
            code_paths: Paths to Python packages to include with the model

        """
        self.config = config
        self.spark = spark
        self.code_paths = code_paths or []

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.bank_marketing_model_custom"
        self.tags = tags.dict()
        self.code_paths = code_paths if code_paths is not None else []

    def load_data(self) -> None:
        """Load training and testing data from processed tables or volumes.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data...")

        try:
            # Try to load from tables
            self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_processed")
            self.train_set = self.train_set_spark.toPandas()
            self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_processed").toPandas()

            # Get data version
            try:
                history = self.spark.sql(
                    f"DESCRIBE HISTORY {self.catalog_name}.{self.schema_name}.train_processed LIMIT 1"
                )
                self.data_version = str(history.collect()[0]["version"])
            except Exception as e:
                logger.warning(f"⚠️ Could not retrieve table version: {e}")
                self.data_version = "0"

            logger.info("✅ Data loaded from tables")
        except Exception as e:
            logger.warning(f"⚠️ Could not load from tables: {e}")

            # Try to load from volumes
            try:
                volume_path = f"/Volumes/{self.catalog_name}/{self.schema_name}/{self.config.volume_name}"
                train_path = f"{volume_path}/processed/train"
                test_path = f"{volume_path}/processed/test"

                # Load from volumes
                train_spark_df = self.spark.read.format("delta").load(train_path)
                test_spark_df = self.spark.read.format("delta").load(test_path)

                # Convert to pandas
                self.train_set_spark = train_spark_df
                self.train_set = train_spark_df.toPandas()
                self.test_set = test_spark_df.toPandas()
                self.data_version = "0"

                logger.info("✅ Data loaded from volumes")
            except Exception as volume_error:
                logger.error(f"❌ Failed to load data from volumes: {volume_error}")

                # Create synthetic data
                logger.warning("⚠️ Creating synthetic data for demonstration")
                import numpy as np

                # Create synthetic features
                n_train = 1000
                n_test = 200

                # Create synthetic data
                synthetic_train = {}
                for col in self.num_features:
                    synthetic_train[col] = np.random.normal(0, 1, n_train)

                for col in self.cat_features:
                    synthetic_train[col] = np.random.choice(["A", "B", "C"], n_train)

                # Create target
                synthetic_train[self.target] = np.random.choice([0, 1], n_train)

                # Create test data
                synthetic_test = {}
                for col in self.num_features:
                    synthetic_test[col] = np.random.normal(0, 1, n_test)

                for col in self.cat_features:
                    synthetic_test[col] = np.random.choice(["A", "B", "C"], n_test)

                synthetic_test[self.target] = np.random.choice([0, 1], n_test)

                # Create pandas DataFrames
                self.train_set = pd.DataFrame(synthetic_train)
                self.test_set = pd.DataFrame(synthetic_test)

                # Create spark DataFrame
                self.train_set_spark = self.spark.createDataFrame(self.train_set)
                self.data_version = "synthetic"

                logger.info("✅ Synthetic data created for demonstration")

        # Split features and target
        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]

        logger.info(f"📊 Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")

    def prepare_features(self) -> None:
        """Encode categorical features and define a preprocessing pipeline.

        Creates a ColumnTransformer for one-hot encoding categorical features and
        scaling numerical features.
        """
        logger.info("🔄 Defining preprocessing pipeline...")

        # Transformers for different feature types
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

        # Combine transformers in a column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.num_features),
                ("cat", categorical_transformer, self.cat_features),
            ]
        )

        # Create the pipeline with preprocessor and classifier
        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("classifier", LGBMClassifier(**self.parameters))]
        )

        logger.info("✅ Preprocessing pipeline defined")

    def train(self) -> None:
        """Train the model pipeline."""
        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)
        logger.info("✅ Model training completed")

    def log_model(self, dataset_type: Literal["PandasDataset", "SparkDataset"] = "SparkDataset") -> None:
        """Log the model using MLflow pyfunc."""
        logger.info("📝 Logging model with MLflow pyfunc...")
        mlflow.set_experiment(self.experiment_name)
        additional_pip_deps = ["pyspark==3.5.0"]  # Podría ser condicional también, ver abajo
        for package in self.code_paths:
            whl_name = package.split("/")[-1]
            additional_pip_deps.append(f"./code/{whl_name}")

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            # Make predictions
            # Asumo que X_test es un Pandas DataFrame o similar para predict
            y_pred = self.pipeline.predict(self.X_test)
            y_pred_proba = self.pipeline.predict_proba(self.X_test)[:, 1]

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
            mlflow.log_param("model_type", "Custom LightGBM with pyfunc wrapper")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            # Create model wrapper
            wrapped_model = BankMarketingModelWrapper(self.pipeline)

            # Compute input example and signature
            # Asegúrate de que X_train sea un Pandas DataFrame para .iloc
            input_example = self.X_train.iloc[:5]
            predictions = wrapped_model.predict(None, input_example)
            signature = infer_signature(model_input=input_example, model_output=predictions)

            # --- CORRECCIÓN AQUÍ: USO DEL PARÁMETRO dataset_type ---
            if dataset_type == "SparkDataset":
                if not hasattr(self, "train_set_spark") or self.train_set_spark is None:
                    logger.warning("⚠️ train_set_spark no está disponible. No se registrará el dataset de Spark.")
                else:
                    dataset = mlflow.data.from_spark(
                        self.train_set_spark,
                        table_name=f"{self.catalog_name}.{self.schema_name}.train_processed",
                        version=self.data_version,
                    )
                    mlflow.log_input(dataset, context="training")
            elif dataset_type == "PandasDataset":
                if not hasattr(self, "train_set_pandas") or self.train_set_pandas is None:
                    logger.warning("⚠️ train_set_pandas no está disponible. No se registrará el dataset de Pandas.")
                else:
                    dataset = mlflow.data.from_pandas(
                        self.train_set_pandas,
                        source="in_memory",  # O una ruta si lo cargas de archivo
                        name=f"{self.catalog_name}.{self.schema_name}.train_processed",
                        version=self.data_version,
                    )
                    mlflow.log_input(dataset, context="training")
            else:
                logger.warning(f"Tipo de dataset no soportado: {dataset_type}. No se registrará ningún dataset.")
            # --- FIN DE LA CORRECCIÓN ---

            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            # Log model with pyfunc flavor
            mlflow.pyfunc.log_model(
                artifact_path="pyfunc-bank-marketing-model",
                python_model=wrapped_model,
                artifacts={},
                conda_env=conda_env,
                code_paths=self.code_paths,
                input_example=input_example,
                signature=signature,
            )

            logger.info(f"✅ Model logged with run_id: {self.run_id}")

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in Unity Catalog...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/pyfunc-bank-marketing-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}")

        latest_version = registered_model.version

        # Set aliases for model version
        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="custom-model",
            version=latest_version,
        )

        # Add model description
        client.update_registered_model(
            name=self.model_name,
            description="Custom pyfunc model for Bank Marketing dataset. Provides enhanced prediction output format.",
        )

        # Add version description
        client.update_model_version(
            name=self.model_name,
            version=latest_version,
            description=f"Custom model trained with {len(self.X_train)} samples. ROC AUC: {mlflow.get_run(self.run_id).data.metrics['roc_auc']:.4f}",
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

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> dict:
        """Load the latest custom model from MLflow and make predictions.

        Args:
            input_data: Pandas DataFrame containing input features for prediction.

        Returns:
            Dictionary with predictions and additional metadata.

        """
        logger.info("🔄 Loading model from MLflow alias 'custom-model'...")

        model_uri = f"models:/{self.model_name}@custom-model"
        model = mlflow.pyfunc.load_model(model_uri)

        logger.info("✅ Model successfully loaded")

        # Make predictions using the custom format defined in the wrapper
        predictions = model.predict(input_data)

        return predictions
