"""Script to train and register a custom model for Bank Marketing prediction.

This script handles the full MLOps lifecycle from loading data to model registration
in a production environment. It's designed to be run as a job in Databricks with
proper parameter passing and error handling.
"""

import argparse

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from bank_marketing.config import ProjectConfig, Tags
from bank_marketing.models.basic_model import BasicModel

# Configure tracking uri for Databricks environment
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Define command-line arguments for the script
parser = argparse.ArgumentParser(description="Bank Marketing model training and registration")
parser.add_argument(
    "--root_path", action="store", default=None, type=str, required=True, help="Root path to the project files"
)

parser.add_argument("--env", action="store", default=None, type=str, required=True, help="Environment (dev, acc, prd)")

parser.add_argument("--git_sha", action="store", default=None, type=str, required=True, help="Git commit SHA")

parser.add_argument("--job_run_id", action="store", default=None, type=str, required=True, help="Databricks job run ID")

parser.add_argument("--branch", action="store", default=None, type=str, required=True, help="Git branch name")

# Parse the arguments
args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/project_config.yml"

# Initialize configuration and Spark session
logger.info(f"Loading configuration from {config_path} for environment {args.env}")
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Create tags for MLflow tracking
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

try:
    # Initialize model
    logger.info("Initializing Bank Marketing model...")
    basic_model = BasicModel(config=config, tags=tags, spark=spark)
    logger.info("✅ Model initialized")

    # Load data and prepare features
    logger.info("Loading data and preparing features...")
    basic_model.load_data()
    basic_model.prepare_features()
    logger.info("✅ Data loaded and features prepared")

    # Train the model
    logger.info("Starting model training...")
    basic_model.train()
    logger.info("✅ Model training completed")

    # Log model to MLflow
    logger.info("Logging model to MLflow...")
    basic_model.log_model()
    logger.info("✅ Model logged to MLflow successfully")

    # Register model in Unity Catalog
    logger.info("Registering model in Unity Catalog...")
    basic_model.register_model()
    logger.info("✅ Model registered successfully")

    # Retrieve and display metrics
    metrics, params = basic_model.retrieve_current_run_metadata()
    logger.info(f"📊 Model Performance: ROC AUC = {metrics.get('roc_auc', 'N/A')}")

    logger.info(f"""
    ✅ Model Training and Registration Complete:
       - Model: {basic_model.model_name}
       - Environment: {args.env}
       - Git SHA: {args.git_sha}
       - Branch: {args.branch}
       - Job Run ID: {args.job_run_id}
    """)

except Exception as e:
    logger.error(f"❌ Error during model training and registration: {e}")
    raise
