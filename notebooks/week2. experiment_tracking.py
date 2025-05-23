# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Experiment Tracking - Demonstration
# MAGIC
# MAGIC This notebook demonstrates the use of MLflow for experiment tracking with synthetic data.

# COMMAND ----------

import json
import mlflow
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from marvelous.common import is_databricks
from dotenv import load_dotenv


# COMMAND ----------
mlflow.get_tracking_uri()

# COMMAND ----------
if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")

mlflow.get_tracking_uri()
# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. MLflow Configuration

# COMMAND ----------

# Get the current tracking URI
print(f"Current tracking URI: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Experiment Creation and Configuration

# COMMAND ----------

# Create a demo experiment
experiment_name = "/Shared/bank-marketing-tracking-demo"
experiment = mlflow.set_experiment(experiment_name)

# Set tags for the experiment
mlflow.set_experiment_tags(
    {"project": "bank_marketing_demo", "domain": "financial_services", "purpose": "experiment_tracking_demo"}
)

print(f"Experiment configured: {experiment.name}")
print(f"Experiment ID: {experiment.experiment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Basic Run with Parameters and Metrics

# COMMAND ----------

# Basic run example
with mlflow.start_run(
    run_name="demo-basic-tracking",
    tags={"git_sha": "demo123", "branch": "experiment-tracking", "model_type": "demo"},
    description="Basic MLflow tracking demonstration",
) as run:
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

    # Log parameters (model configuration)
    mlflow.log_params({"learning_rate": 0.05, "n_estimators": 100, "max_depth": 5, "model_type": "lightgbm"})

    # Simulate training metrics
    metrics = {"accuracy": 0.85, "precision": 0.82, "recall": 0.78, "f1_score": 0.80, "auc": 0.88}

    # Log metrics
    mlflow.log_metrics(metrics)

    print("✅ Parameters and metrics logged")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Logging Dynamic Metrics (Training Epochs)

# COMMAND ----------

# Simulate training with multiple epochs
with mlflow.start_run(run_name="demo-dynamic-metrics") as run:
    # Simulate improving metrics during training
    for epoch in range(10):
        # Simulate metrics that improve over time
        train_loss = 1.0 - (epoch * 0.08) + np.random.normal(0, 0.02)
        val_loss = 1.1 - (epoch * 0.07) + np.random.normal(0, 0.03)
        accuracy = 0.5 + (epoch * 0.04) + np.random.normal(0, 0.01)

        # Log metrics with step to create time-series charts
        mlflow.log_metrics(
            {
                "train_loss": max(0.1, train_loss),
                "val_loss": max(0.15, val_loss),
                "accuracy": min(0.95, max(0.5, accuracy)),
            },
            step=epoch,
        )

    print("✅ Dynamic metrics logged")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Logging Artifacts and Visualizations

# COMMAND ----------

with mlflow.start_run(run_name="demo-artifacts") as run:
    # 1. Log plain text
    mlflow.log_text("Demonstration experiment for Bank Marketing", "experiment_notes.txt")

    # 2. Log dictionary as JSON
    model_config = {
        "architecture": "gradient_boosting",
        "features": ["age", "job", "balance", "housing"],
        "target": "subscription",
        "preprocessing": {"scaling": "standard", "encoding": "onehot"},
    }
    mlflow.log_dict(model_config, "model_config.json")

    # 3. Create and log a plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Feature importance plot (simulated)
    features = ["age", "balance", "duration", "campaign", "housing"]
    importance = [0.25, 0.30, 0.20, 0.15, 0.10]

    ax1.barh(features, importance)
    ax1.set_xlabel("Importance")
    ax1.set_title("Feature Importance")

    # Target distribution plot (simulated)
    labels = ["Not subscribed", "Subscribed"]
    sizes = [70, 30]

    ax2.pie(sizes, labels=labels, autopct="%1.1f%%")
    ax2.set_title("Target Distribution")

    plt.tight_layout()
    mlflow.log_figure(fig, "model_analysis.png")
    plt.close()

    # 4. Create and log multiple images
    for i in range(3):
        # Simulate convergence plots
        epochs = range(20)
        loss = [1.0 * (0.9**epoch) + np.random.normal(0, 0.02) for epoch in epochs]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss, label=f"Experiment {i + 1}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Model Convergence - Experiment {i + 1}")
        plt.legend()
        plt.grid(True)

        # Save as a temporary file
        temp_path = f"convergence_exp_{i}.png"
        plt.savefig(temp_path)
        plt.close()

        # Log to MLflow
        mlflow.log_artifact(temp_path, "convergence_plots")

        # Clean up temporary file
        os.remove(temp_path)

    print("✅ Artifacts and visualizations logged")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Nested Runs for Hyperparameter Tuning

# COMMAND ----------

# Nested runs demonstration
with mlflow.start_run(run_name="hyperparameter_tuning_demo") as parent_run:
    print(f"Parent run started: {parent_run.info.run_id}")

    # Experiment parameters
    mlflow.log_params(
        {"dataset": "bank_marketing_demo", "target": "subscription", "cv_folds": 5, "search_type": "grid_search"}
    )

    best_score = 0
    best_params = {}

    # Simulated grid search
    for learning_rate in [0.01, 0.05, 0.1]:
        for max_depth in [3, 5, 7]:
            for n_estimators in [50, 100]:
                # Create child run
                with mlflow.start_run(
                    run_name=f"lr_{learning_rate}_depth_{max_depth}_est_{n_estimators}", nested=True
                ) as child_run:
                    # Log specific parameters
                    params = {"learning_rate": learning_rate, "max_depth": max_depth, "n_estimators": n_estimators}
                    mlflow.log_params(params)

                    # Simulate training and evaluation
                    base_score = 0.75
                    lr_bonus = learning_rate * 2
                    depth_bonus = max_depth / 20
                    est_bonus = n_estimators / 1000

                    # Add some noise
                    noise = np.random.normal(0, 0.02)
                    score = min(0.95, base_score + lr_bonus + depth_bonus + est_bonus + noise)

                    # Simulate training time
                    training_time = 10 + (n_estimators / 10) + np.random.normal(0, 2)

                    # Log metrics
                    mlflow.log_metrics({"cv_score": score, "training_time": max(1, training_time)})

                    # Track the best model
                    if score > best_score:
                        best_score = score
                        best_params = params

                    print(
                        f"     Configuration: LR={learning_rate}, Depth={max_depth}, Est={n_estimators}, Score={score:.4f}"
                    )

    # Log the best result in the parent run
    mlflow.log_metrics({"best_cv_score": best_score})
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

    print(f"\n✅ Best configuration: {best_params} with score: {best_score:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Experiment Search and Analysis

# COMMAND ----------

# Search for runs with filters
from time import time

time_hour_ago = int(time() - 3600) * 1000

# IMPORTANT: Replace 'your_experiment_name' with the actual name of your MLflow experiment

try:
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["start_time DESC"],
        filter_string="status='FINISHED' AND "
        f"start_time>{time_hour_ago} AND "
        "tags.mlflow.runName LIKE '%demo%'",  # Changed to use tags.mlflow.runName
    )

    print(f"Runs found in the last hour: {len(runs)}")

    if not runs.empty:
        # Define the relevant columns you want to display
        relevant_columns = [
            "tags.mlflow.runName",
            "start_time",
            "metrics.best_cv_score",
            "params.model_type",
            "run_id",  # Adding run_id as it's always good for identification
        ]

        # Filter the DataFrame to only include the relevant columns, if they exist
        # Also, ensure 'start_time' is converted to a readable format if present
        display_runs = pd.DataFrame()
        if not runs.empty:
            # Select columns that exist in the DataFrame
            existing_relevant_columns = [col for col in relevant_columns if col in runs.columns]
            display_runs = runs[existing_relevant_columns].copy()

            # Convert 'start_time' to datetime if it exists
            if "start_time" in display_runs.columns:
                display_runs["start_time"] = pd.to_datetime(display_runs["start_time"], unit="ms")

        print("\nRun summary with relevant columns:")
        print(display_runs.head())

        # Show metrics of the best runs, using 'metrics.best_cv_score'
        if "metrics.best_cv_score" in runs.columns:
            best_runs = runs.nlargest(3, "metrics.best_cv_score")
            print("\nTop 3 runs by 'Best CV Score':")
            for idx, run in best_runs.iterrows():
                run_name = run["tags.mlflow.runName"] if "tags.mlflow.runName" in run else "N/A"
                print(f"- {run_name}: Best CV Score = {run['metrics.best_cv_score']:.4f}")
        else:
            print("\nThe metric 'metrics.best_cv_score' is not available in the runs.")

    else:
        print("No runs found matching the filter criteria.")

except Exception as e:
    print(f"An error occurred: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Artifact Retrieval

# COMMAND ----------

# Load artifacts from the latest run
if not runs.empty:
    latest_run = runs.iloc[0]
    artifact_uri = latest_run["artifact_uri"]

    try:
        # Try to load model configuration
        config_data = mlflow.artifacts.load_dict(f"{artifact_uri}/model_config.json")
        print("📊 Model configuration loaded:")
        for key, value in config_data.items():
            print(f"   - {key}: {value}")
    except Exception as e:
        print(f"⚠️ Could not load configuration: {e}")
