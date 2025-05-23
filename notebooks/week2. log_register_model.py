# Databricks notebook source
# MAGIC %md
# MAGIC # Logging and Registering Models with MLflow
# MAGIC
# MAGIC This notebook demonstrates different approaches to registering models with MLflow
# MAGIC using Databricks Asset Bundles and Unity Catalog.

# COMMAND ----------

from pyspark.sql import SparkSession
import mlflow
import pandas as pd
import os

from bank_marketing.config import ProjectConfig
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from mlflow import MlflowClient

# Import package version if available
try:
    from bank_marketing import __version__
except ImportError:
    __version__ = "0.1.0"  # Default version

from mlflow.utils.environment import _mlflow_conda_env

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. MLflow Configuration for Databricks Asset Bundles

# COMMAND ----------

# Configure MLflow for Unity Catalog (this is automatic in Databricks)
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    print("✅ Running on Databricks - automatic configuration")
    # In Databricks, MLflow is already configured for Unity Catalog
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
else:
    print("⚠️ Running locally")
    # For local development with DAB
    try:
        from databricks.connect import DatabricksSession

        spark = DatabricksSession.builder.getOrCreate()
        print("✅ Connected using Databricks Connect")
    except ImportError:
        print("❌ Databricks Connect not available, using local Spark")
        spark = SparkSession.builder.getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Spark and Load Data

# COMMAND ----------

# Get SparkSession (in Databricks it's already available as 'spark')
try:
    # In Databricks, 'spark' is already defined
    spark = spark
    print("✅ Using Databricks SparkSession")
except NameError:
    # For local execution
    spark = SparkSession.builder.getOrCreate()
    print("✅ Local SparkSession created")

# Load training data with error handling
try:
    # Attempt to load from Unity Catalog
    train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_processed").toPandas()
    print("✅ Data loaded from Unity Catalog")
except Exception as e:
    print(f"⚠️ Could not load from Unity Catalog: {e}")

    # Create synthetic data for demonstration
    import numpy as np

    np.random.seed(42)

    # Create data that matches your configuration
    n_samples = 1000
    synthetic_data = {}

    # Create numerical features
    for col in config.num_features:
        if col == "age":
            synthetic_data[col] = np.random.randint(18, 70, n_samples)
        elif col == "balance":
            synthetic_data[col] = np.random.normal(1000, 500, n_samples)
        else:
            synthetic_data[col] = np.random.normal(0, 1, n_samples)

    # Create categorical features
    for col in config.cat_features:
        if col == "job":
            synthetic_data[col] = np.random.choice(["admin", "blue-collar", "management", "technician"], n_samples)
        elif col == "marital":
            synthetic_data[col] = np.random.choice(["married", "single", "divorced"], n_samples)
        else:
            synthetic_data[col] = np.random.choice(["A", "B", "C"], n_samples)

    # Create target
    synthetic_data[config.target] = np.random.choice([0, 1], n_samples)

    train_set = pd.DataFrame(synthetic_data)
    print("✅ Synthetic data created")

# Separate features and target
X_train = train_set[config.num_features + config.cat_features]
y_train = train_set[config.target]

print(f"📊 Data shape: {X_train.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create and Train a Basic Pipeline

# COMMAND ----------

# Define the pipeline with preprocessing for classification
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, config.num_features),
        ("cat", categorical_transformer, config.cat_features),
    ]
)

# Create the full pipeline with classifier
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**config.parameters))])

# Train the model
pipeline.fit(X_train, y_train)
print("✅ Model trained successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log the Model with MLflow using Unity Catalog

# COMMAND ----------

# Configure experiment using DAB structure
experiment_name = f"/{config.schema_name}/bank-marketing-demo"
mlflow.set_experiment(experiment_name)

# Start run and log model
with mlflow.start_run(
    run_name="dab-demo-model",
    tags={"git_sha": "dab123", "branch": "main", "bundle": "marvelous-databricks-course-ronaldespinozaa"},
    description="Demo model using Databricks Asset Bundle",
) as run:
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM with preprocessing - DAB")
    mlflow.log_params(config.parameters)

    # Calculate and log metrics
    y_pred = pipeline.predict(X_train)
    y_pred_proba = pipeline.predict_proba(X_train)[:, 1]

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    roc_auc = roc_auc_score(y_train, y_pred_proba)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    print(f"📊 Metrics - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

    # Infer model signature
    signature = infer_signature(model_input=X_train, model_output=y_pred_proba)

    # Log the model
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)

    print(f"✅ Model logged with run_id: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Register Model in Unity Catalog (Correct Format for DAB)

# COMMAND ----------

# Define model name using the correct structure for Unity Catalog
model_name = f"{config.catalog_name}.{config.schema_name}.bank_marketing_model_demo"

try:
    # Register model
    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/lightgbm-pipeline-model",
        name=model_name,
        tags={"bundle": "marvelous-databricks-course-ronaldespinozaa", "git_sha": "dab123"},
    )

    print(f"✅ Model registered: {model_name} version {model_version.version}")

    # Set alias using MLflow Client
    client = MlflowClient()

    # Set "dab-latest" alias
    client.set_registered_model_alias(name=model_name, alias="dab-latest", version=model_version.version)
    print(f"✅ Alias 'dab-latest' configured for version {model_version.version}")

except Exception as e:
    print(f"⚠️ Error registering model: {e}")
    print("This can happen if you do not have permissions in Unity Catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Load Model and Make Predictions

# COMMAND ----------

try:
    # Load model using the alias
    model_uri = f"models:/{model_name}@dab-latest"
    sklearn_pipeline = mlflow.sklearn.load_model(model_uri)

    # Make predictions on a sample
    sample_data = X_train.iloc[:5]
    predictions = sklearn_pipeline.predict(sample_data)
    probabilities = sklearn_pipeline.predict_proba(sample_data)[:, 1]

    # Display results
    result_df = pd.DataFrame(
        {"prediction": predictions, "probability": probabilities, "actual": y_train.iloc[:5].values}
    )

    display(result_df)
    print("✅ Predictions made successfully")

except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    print("Using in-memory model for demonstration...")

    # Fallback to in-memory model
    sample_data = X_train.iloc[:5]
    predictions = pipeline.predict(sample_data)
    probabilities = pipeline.predict_proba(sample_data)[:, 1]

    result_df = pd.DataFrame(
        {"prediction": predictions, "probability": probabilities, "actual": y_train.iloc[:5].values}
    )

    display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Custom Model with pyfunc and DAB

# COMMAND ----------

# Databricks notebook source
# MAGIC %md
# MAGIC # Logging and Registering Models with MLflow
# MAGIC
# MAGIC This notebook demonstrates different approaches to registering models with MLflow
# MAGIC using Databricks Asset Bundles and Unity Catalog.

# COMMAND ----------

from pyspark.sql import SparkSession
import mlflow
import pandas as pd
import os

from bank_marketing.config import ProjectConfig
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from mlflow import MlflowClient

# Import package version if available
try:
    from bank_marketing import __version__
except ImportError:
    __version__ = "0.1.0"  # Default version

from mlflow.utils.environment import _mlflow_conda_env

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. MLflow Configuration for Databricks Asset Bundles

# COMMAND ----------

# Configure MLflow for Unity Catalog (this is automatic in Databricks)
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    print("✅ Running on Databricks - automatic configuration")
    # In Databricks, MLflow is already configured for Unity Catalog
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
else:
    print("⚠️ Running locally")
    # For local development with DAB
    try:
        from databricks.connect import DatabricksSession

        spark = DatabricksSession.builder.getOrCreate()
        print("✅ Connected using Databricks Connect")
    except ImportError:
        print("❌ Databricks Connect not available, using local Spark")
        spark = SparkSession.builder.getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Spark and Load Data

# COMMAND ----------

# Get SparkSession (in Databricks it's already available as 'spark')
try:
    # In Databricks, 'spark' is already defined
    spark = spark
    print("✅ Using Databricks SparkSession")
except NameError:
    # For local execution
    spark = SparkSession.builder.getOrCreate()
    print("✅ Local SparkSession created")

# Load training data with error handling
try:
    # Attempt to load from Unity Catalog
    train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_processed").toPandas()
    print("✅ Data loaded from Unity Catalog")
except Exception as e:
    print(f"⚠️ Could not load from Unity Catalog: {e}")

    # Create synthetic data for demonstration
    import numpy as np

    np.random.seed(42)

    # Create data that matches your configuration
    n_samples = 1000
    synthetic_data = {}

    # Create numerical features
    for col in config.num_features:
        if col == "age":
            synthetic_data[col] = np.random.randint(18, 70, n_samples)
        elif col == "balance":
            synthetic_data[col] = np.random.normal(1000, 500, n_samples)
        else:
            synthetic_data[col] = np.random.normal(0, 1, n_samples)

    # Create categorical features
    for col in config.cat_features:
        if col == "job":
            synthetic_data[col] = np.random.choice(["admin", "blue-collar", "management", "technician"], n_samples)
        elif col == "marital":
            synthetic_data[col] = np.random.choice(["married", "single", "divorced"], n_samples)
        else:
            synthetic_data[col] = np.random.choice(["A", "B", "C"], n_samples)

    # Create target
    synthetic_data[config.target] = np.random.choice([0, 1], n_samples)

    train_set = pd.DataFrame(synthetic_data)
    print("✅ Synthetic data created")

# Separate features and target
X_train = train_set[config.num_features + config.cat_features]
y_train = train_set[config.target]

print(f"📊 Data shape: {X_train.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create and Train a Basic Pipeline

# COMMAND ----------

# Define the pipeline with preprocessing for classification
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, config.num_features),
        ("cat", categorical_transformer, config.cat_features),
    ]
)

# Create the full pipeline with classifier
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**config.parameters))])

# Train the model
pipeline.fit(X_train, y_train)
print("✅ Model trained successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log the Model with MLflow using Unity Catalog

# COMMAND ----------

# Configure experiment using DAB structure
experiment_name = f"/{config.schema_name}/bank-marketing-demo"
mlflow.set_experiment(experiment_name)

# Start run and log model
with mlflow.start_run(
    run_name="dab-demo-model",
    tags={"git_sha": "dab123", "branch": "main", "bundle": "marvelous-databricks-course-ronaldespinozaa"},
    description="Demo model using Databricks Asset Bundle",
) as run:
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM with preprocessing - DAB")
    mlflow.log_params(config.parameters)

    # Calculate and log metrics
    y_pred = pipeline.predict(X_train)
    y_pred_proba = pipeline.predict_proba(X_train)[:, 1]

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    roc_auc = roc_auc_score(y_train, y_pred_proba)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    print(f"📊 Metrics - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

    # Infer model signature
    signature = infer_signature(model_input=X_train, model_output=y_pred_proba)

    # Log the model
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)

    print(f"✅ Model logged with run_id: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Register Model in Unity Catalog (Correct Format for DAB)

# COMMAND ----------

# Define model name using the correct structure for Unity Catalog
model_name = f"{config.catalog_name}.{config.schema_name}.bank_marketing_model_demo"

try:
    # Register model
    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/lightgbm-pipeline-model",
        name=model_name,
        tags={"bundle": "marvelous-databricks-course-ronaldespinozaa", "git_sha": "dab123"},
    )

    print(f"✅ Model registered: {model_name} version {model_version.version}")

    # Set alias using MLflow Client
    client = MlflowClient()

    # Set "dab-latest" alias
    client.set_registered_model_alias(name=model_name, alias="dab-latest", version=model_version.version)
    print(f"✅ Alias 'dab-latest' configured for version {model_version.version}")

except Exception as e:
    print(f"⚠️ Error registering model: {e}")
    print("This can happen if you do not have permissions in Unity Catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Load Model and Make Predictions

# COMMAND ----------

try:
    # Load model using the alias
    model_uri = f"models:/{model_name}@dab-latest"
    sklearn_pipeline = mlflow.sklearn.load_model(model_uri)

    # Make predictions on a sample
    sample_data = X_train.iloc[:5]
    predictions = sklearn_pipeline.predict(sample_data)
    probabilities = sklearn_pipeline.predict_proba(sample_data)[:, 1]

    # Display results
    result_df = pd.DataFrame(
        {"prediction": predictions, "probability": probabilities, "actual": y_train.iloc[:5].values}
    )

    display(result_df)
    print("✅ Predictions made successfully")

except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    print("Using in-memory model for demonstration...")

    # Fallback to in-memory model
    sample_data = X_train.iloc[:5]
    predictions = pipeline.predict(sample_data)
    probabilities = pipeline.predict_proba(sample_data)[:, 1]

    result_df = pd.DataFrame(
        {"prediction": predictions, "probability": probabilities, "actual": y_train.iloc[:5].values}
    )

    display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Custom Model with pyfunc and DAB

# COMMAND ----------


# Wrapper class for custom model
class BankMarketingModelWrapper(mlflow.pyfunc.PythonModel):
    """Custom wrapper for the Bank Marketing model with DAB."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            # Get predictions
            predictions = self.model.predict(model_input)
            probabilities = self.model.predict_proba(model_input)[:, 1]
            return pd.DataFrame(
                {"prediction": predictions, "probability": probabilities}
            )  # Added return statement to complete the predict method


# Wrapper class for custom model
class BankMarketingModelWrapper(mlflow.pyfunc.PythonModel):
    """Custom wrapper for the Bank Marketing model with DAB."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            # Get predictions
            predictions = self.model.predict(model_input)
            probabilities = self.model.predict_proba(model_input)[:, 1]
            return pd.DataFrame(
                {"prediction": predictions, "probability": probabilities}
            )  # Added return statement to complete the predict method
