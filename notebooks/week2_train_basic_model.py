# Databricks notebook source
# MAGIC %md
# MAGIC # Week 2: Training and Registering a Basic Model with MLflow
# MAGIC This notebook demonstrates:
# MAGIC 1. Loading train/test data from the Unity Catalog
# MAGIC 2. Training a model within an MLflow run with pipeline and preprocessing
# MAGIC 3. Logging metrics, parameters, datasets, and artifacts
# MAGIC 4. Registering the model in MLflow Model Registry
# MAGIC 5. Retrieving data and model information

# COMMAND ----------

# %pip install -e ..

# COMMAND ----------

# %restart_python

# COMMAND ----------
import pandas as pd
from loguru import logger
from marvelous.logging import setup_logging
from marvelous.timer import Timer
from pyspark.sql import SparkSession

from bank_marketing.config import ProjectConfig, Tags
from bank_marketing.models.basic_model import BasicModel

# COMMAND ----------
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------
# Initialize session and configuration
# Initialize session and configuration
spark = SparkSession.builder.getOrCreate() if "spark" not in locals() else spark
# Load configuration from YAML
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

# Setup logging
setup_logging(log_file="logs/marvelous-train-model.log")

# Create tags for MLflow tracking
tags = Tags(
    git_sha="abc123",  # This would come from actual Git in production
    branch="main",  # This would come from actual Git in production
    job_run_id="manual",  # This would come from Databricks job context in production
)

logger.info(f"✅ Configuration loaded for environment: {config.catalog_name}.{config.schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize and Train the Basic Model

# COMMAND ----------

# Initialize the model
with Timer() as init_timer:
    model = BasicModel(config=config, tags=tags, spark=spark)
logger.info(f"⏱️ Model initialization: {init_timer}")

# Load data
with Timer() as load_timer:
    model.load_data()
logger.info(f"⏱️ Data loading: {load_timer}")

# Prepare features
with Timer() as prep_timer:
    model.prepare_features()
logger.info(f"⏱️ Feature preparation: {prep_timer}")

# Train model
with Timer() as train_timer:
    model.train()
logger.info(f"⏱️ Model training: {train_timer}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Log and Register the Model with MLflow

# COMMAND ----------

# Log model to MLflow
with Timer() as log_timer:
    model.log_model()
logger.info(f"⏱️ Model logging: {log_timer}")

# Register model in MLflow Model Registry
with Timer() as reg_timer:
    model.register_model()
logger.info(f"⏱️ Model registration: {reg_timer}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Retrieve Model Information and Metadata

# COMMAND ----------

# Retrieve run metadata
metrics, params = model.retrieve_current_run_metadata()

# Display metrics
logger.info("📊 Model Performance Metrics:")
for metric, value in metrics.items():
    logger.info(f"   - {metric}: {value}")

# Display parameters
logger.info("🔧 Model Parameters:")
for param, value in params.items():
    logger.info(f"   - {param}: {value}")

# Retrieve dataset info (optional)
try:
    dataset = model.retrieve_current_run_dataset()
    logger.info(f"📚 Dataset shape: {dataset.shape}")
except Exception as e:
    logger.warning(f"⚠️ Could not retrieve dataset: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Make Predictions with the Registered Model

# COMMAND ----------

# Get sample data for prediction
sample_data = model.X_test.iloc[:5]

# Load model and make predictions
with Timer() as pred_timer:
    predictions = model.load_latest_model_and_predict(sample_data)
logger.info(f"⏱️ Prediction time: {pred_timer}")

# Create a result dataframe
result_df = pd.DataFrame(
    {"probability": predictions, "prediction": (predictions >= 0.5).astype(int), "actual": model.y_test.iloc[:5].values}
)

# Use display if available, otherwise fallback to print
if "display" in locals():
    display(result_df)
else:
    print(result_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary & Next Steps

# COMMAND ----------

logger.info(f"""
✅ MLflow Model Training & Registration Summary:
   - Model Name: {model.model_name}
   - Experiment: {model.experiment_name}
   - Run ID: {model.run_id}
   - ROC AUC: {metrics.get("roc_auc", "N/A")}
   - Training samples: {len(model.X_train)}

✨ MLOps Next Steps:
   1. Model validation and promotion to staging
   2. Model deployment for batch or real-time inference
   3. Model monitoring for data drift
   4. A/B testing with champion/challenger models
""")
