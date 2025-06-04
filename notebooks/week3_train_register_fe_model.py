# Databricks notebook source
# MAGIC %md
# MAGIC # Week 3: Feature Store Training - Simplified Version
# MAGIC
# MAGIC This notebook demonstrates Feature Store with a simplified, practical approach.

# COMMAND ----------

# MAGIC %pip install -e ..
# MAGIC %pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# system path update, must be after %restart_python
# caution! This is not a great approach
from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent / "src"))

# COMMAND ----------

import mlflow
from pyspark.sql import SparkSession
from loguru import logger

from bank_marketing.config import ProjectConfig, Tags
from bank_marketing.models.feature_lookup_model import FeatureLookUpModel

# COMMAND ----------

# Configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()

tags = Tags(git_sha="week3simple", branch="week3-simplified", job_run_id="simple-feature-store")

logger.info(f"✅ Configuration loaded: {config.catalog_name}.{config.schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Initialize and Run Complete Pipeline

# COMMAND ----------

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Run complete pipeline in one go
fe_model.run_full_pipeline()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Check Results

# COMMAND ----------

# Get model performance
performance = fe_model.get_model_performance()
logger.info(f"📊 Model Performance: {performance}")

# Display training data info
logger.info(f"📊 Training data shape: {fe_model.training_df.shape}")
logger.info(f"📊 Features: {[col for col in fe_model.training_df.columns if col != config.target]}")

display(fe_model.training_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Make Predictions (Fixed)

# COMMAND ----------

# Simple approach - use the trained pipeline directly
logger.info("🔮 Making predictions with trained pipeline...")

# Get sample data
sample_data = fe_model.X_test.head(5)

# Make predictions using the pipeline that's already in memory
predictions = fe_model.pipeline.predict(sample_data)
probabilities = fe_model.pipeline.predict_proba(sample_data)[:, 1]

# Create results DataFrame
import pandas as pd

results_df = pd.DataFrame(
    {"prediction": predictions, "probability": probabilities, "actual": fe_model.y_test.head(5).values}
)

display(results_df)

logger.info("✅ Predictions completed successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Summary

# COMMAND ----------

print(f"""
🎉 Feature Store Pipeline Complete!

✅ Model: {fe_model.model_name}
✅ Performance: ROC AUC = {performance.get("roc_auc", 0):.4f}
✅ Training samples: {len(fe_model.X_train)}
✅ Feature Store tables:
   - {fe_model.customer_features_table}
   - {fe_model.campaign_features_table}

🚀 Ready for production deployment!
""")

# COMMAND ----------
