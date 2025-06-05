# Databricks notebook source
# MAGIC %md
# MAGIC # Week 2: Training and Registering a Custom Model with MLflow
# MAGIC
# MAGIC This notebook demonstrates:
# MAGIC 1. Using MLflow pyfunc to create a custom model wrapper
# MAGIC 2. Customizing prediction output format
# MAGIC 3. Incorporating custom code packages with the model
# MAGIC 4. Registering the model in MLflow Model Registry with aliases

# COMMAND ----------
# Install the custom package if not already installed
# MAGIC %pip install -e ..

# COMMAND ----------
# Dont forget to run uv build to create the wheel in the terminal
# COMMAND ----------
# Restart the Python kernel to ensure the package is loaded
# MAGIC %restart_python

# COMMAND ----------

import mlflow
from pyspark.sql import SparkSession
import pandas as pd
import os
from dotenv import load_dotenv
from marvelous.common import is_databricks

from bank_marketing.config import ProjectConfig, Tags
from bank_marketing.models.custom_model import CustomModel

# Get version from package if available
try:
    from bank_marketing import __version__ as bank_marketing_v
except ImportError:
    bank_marketing_v = "0.1.0"  # Default version if not available

# COMMAND ----------
if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

# Load configuration
config = ProjectConfig.from_yaml(
    config_path="../project_config.yml",
)
spark = SparkSession.builder.getOrCreate()

# Create tags for tracking
tags = Tags(
    git_sha="abcd12345",  # This would come from real Git in production
    branch="week2",  # This would come from real Git in production
    job_run_id="notebook",  # This would come from Databricks job run ID in production
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Initialize Custom Model
# MAGIC
# MAGIC We initialize our custom model with the configuration and a path to our custom code.
# MAGIC This allows the model to include our custom package when deployed.

# COMMAND ----------

# # Initialize model with the config and custom code path
# wheel_path = f"../dist/bank_marketing-{bank_marketing_v}-py3-none-any.whl"

# # Check if wheel exists, otherwise use empty list
# try:
#     import os
#     if os.path.exists(wheel_path):
#         code_paths = [wheel_path]
#         print(f"✅ Found custom code wheel at: {wheel_path}")
#     else:
#         code_paths = []
#         print(f"⚠️ Custom code wheel not found at: {wheel_path}")
#         print("   Will continue without custom code package")
# except Exception as e:
#     code_paths = []
#     print(f"⚠️ Error checking wheel path: {e}")

custom_model = CustomModel(
    config=config, tags=tags, spark=spark, code_paths=[f"../dist/bank_marketing-{bank_marketing_v}-py3-none-any.whl"]
)

print("✅ Custom model initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Data and Prepare Features

# COMMAND ----------

# Load data and prepare features
custom_model.load_data()
custom_model.prepare_features()

print(f"✅ Data loaded - Train shape: {custom_model.X_train.shape}, Test shape: {custom_model.X_test.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train and Log Custom Model with MLflow

# COMMAND ----------

# Train the model
custom_model.train()
print("✅ Model training completed")

# Log the model with MLflow using pyfunc
custom_model.log_model()
print(f"✅ Model logged with run ID: {custom_model.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Examine the Logged Run

# COMMAND ----------

# Search for the most recent run in our experiment
run_id = custom_model.run_id

# Load the model from the run
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-bank-marketing-model")

print(f"✅ Model loaded from run: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Retrieve Dataset and Run Metadata

# COMMAND ----------

# Retrieve dataset metadata for the current run
dataset_info = custom_model.retrieve_current_run_dataset()
print(f"📊 Dataset info:", dataset_info)

# Retrieve metadata for the current run
metrics, params = custom_model.retrieve_current_run_metadata()
print(f"📊 Metrics: {metrics}")
print(f"🔧 Parameters: {params}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Register Model in Unity Catalog

# COMMAND ----------

# Register model in MLflow Model Registry
custom_model.register_model()
print(f"✅ Model registered as: {custom_model.model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Make Predictions with the Custom Model
# MAGIC
# MAGIC Now we'll use our custom model to make predictions. Notice that the output
# MAGIC format is different from the standard model - we get a dictionary with
# MAGIC additional information.

# COMMAND ----------

# Predict on the test set
test_set = custom_model.X_test.iloc[:10]
print(test_set.head())

# Get predictions using our custom model's prediction method
predictions = custom_model.load_latest_model_and_predict(test_set)

# Display the predictions in a cleaner format
display(
    pd.DataFrame(
        {
            "prediction": predictions["predictions"],
            "probability": predictions.get("probabilities", [None] * len(predictions["predictions"])),
            "actual": custom_model.y_test.iloc[:10].values,
        }
    )
)

# Show the additional information provided by our custom model
print("\n🔍 Additional Model Output Information:")
for key, value in predictions.items():
    if key not in ["predictions", "probabilities"]:
        print(f"  - {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary

# COMMAND ----------

print(f"""
✅ Custom MLflow Model Summary:
   - Model Name: {custom_model.model_name}
   - Experiment: {custom_model.experiment_name}
   - Run ID: {custom_model.run_id}
   - Custom Features:
     - Enhanced prediction output format
     - Custom code integration
     - Model aliasing

""")

# COMMAND ----------
