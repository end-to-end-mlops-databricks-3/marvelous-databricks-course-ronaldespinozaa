# Databricks notebook source
# MAGIC %md
# MAGIC # Week 3: Feature Store Training - Simplified Version (Improved)

# COMMAND ----------

# MAGIC %pip install -e ..
# MAGIC %pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0

# COMMAND ----------

# ⚠️ Only restart if required, and avoid it during collaborative workflows
# MAGIC %restart_python

# COMMAND ----------

# Setup environment paths (MUST be after %restart_python)
from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent / "src"))

# COMMAND ----------

import mlflow
from pyspark.sql import SparkSession
from loguru import logger
import pandas as pd

from bank_marketing.config import ProjectConfig, Tags
from bank_marketing.models.feature_lookup_model import FeatureLookUpModel

# COMMAND ----------

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(git_sha="week3simple", branch="week3-simplified", job_run_id="simple-feature-store")

logger.info(f"✅ Configuration loaded: {config.catalog_name}.{config.schema_name}")

# COMMAND ----------

# Run pipeline
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)
fe_model.run_full_pipeline()

# COMMAND ----------

# Get and log performance
performance = fe_model.get_model_performance()
logger.info("📊 Model performance:")
for k, v in performance.items():
    logger.info(f"   - {k}: {v:.4f}")

logger.info(f"📊 Training shape: {fe_model.training_df.shape}")
logger.info(f"📊 Features: {[col for col in fe_model.training_df.columns if col != config.target]}")
display(fe_model.training_df.head())

# COMMAND ----------

# Make predictions on a few test samples
logger.info("🔮 Making predictions...")
sample_data = fe_model.X_test.head(5)
predictions = fe_model.pipeline.predict(sample_data)
probabilities = fe_model.pipeline.predict_proba(sample_data)[:, 1]

results_df = pd.DataFrame(
    {"prediction": predictions, "probability": probabilities, "actual": fe_model.y_test.head(5).values}
)
display(results_df)

# COMMAND ----------

# Optional: log to MLflow
with mlflow.start_run(run_name="week3_simplified"):
    if hasattr(fe_model, "params"):
        mlflow.log_params(fe_model.params)
    for metric_name, value in performance.items():
        mlflow.log_metric(metric_name, value)
    mlflow.sklearn.log_model(fe_model.pipeline, "model")
    logger.info("📦 Model logged to MLflow")


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
