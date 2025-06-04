# Databricks notebook source
# MAGIC %md
# MAGIC # Week 3: Feature Engineering Demo - Versión Adaptada
# MAGIC
# MAGIC This notebook demonstrates:
# MAGIC 1. Loading bank marketing data from CSV using successful structure
# MAGIC 2. Advanced feature engineering with our proven approach
# MAGIC 3. Creating Feature Store tables with correct ID strategy
# MAGIC 4. Feature validation and quality checks
# MAGIC 5. Interactive exploration of engineered features

# COMMAND ----------

# MAGIC %pip install -e ..

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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sin, cos, pi, abs as spark_abs, monotonically_increasing_id
from loguru import logger

from bank_marketing.config import ProjectConfig, Tags
from bank_marketing.models.feature_lookup_model import FeatureLookUpModel

# COMMAND ----------

# Load configuration using our successful approach
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()

# Create tags for consistency
tags = Tags(git_sha="week3demo", branch="week3-feature-demo", job_run_id="feature-engineering-demo")

logger.info(f"✅ Configuration loaded for environment: dev")
logger.info(f"📊 Catalog: {config.catalog_name}, Schema: {config.schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Raw Data Using Proven Path

# COMMAND ----------

# Use the same data loading approach that worked successfully
data_path = "file:///Workspace/Users/espinozajr52@gmail.com/.bundle/marvelous-databricks-course-ronaldespinozaa/dev/files/data/data.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Display basic info about the dataset
logger.info(f"📊 Dataset shape: {df.count()} rows, {len(df.columns)} columns")
logger.info("📊 Dataset schema:")
df.printSchema()

# Show first few rows
display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Quality Assessment

# COMMAND ----------

# Convert to pandas for easier analysis
df_pandas = df.toPandas()

# Basic statistics
logger.info("📊 Dataset Statistics:")
display(df_pandas.describe())

# Check for missing values
logger.info("🔍 Missing Values Check:")
missing_values = df_pandas.isnull().sum()
if missing_values.sum() > 0:
    display(missing_values[missing_values > 0])
else:
    logger.info("✅ No missing values found")

# Check unique values in categorical columns
logger.info("🔍 Unique values in categorical columns:")
categorical_cols = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
    "Target",
]
for col_name in categorical_cols:
    if col_name in df_pandas.columns:
        unique_vals = df_pandas[col_name].unique()
        logger.info(f"{col_name}: {unique_vals}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Target Variable Analysis

# COMMAND ----------

# Analyze target distribution
target_dist = df_pandas["Target"].value_counts()
logger.info(f"📊 Target Distribution:")
display(target_dist)

# Visualize target distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
target_dist.plot(kind="bar", color=["lightcoral", "lightblue"])
plt.title("Target Distribution")
plt.ylabel("Count")
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
target_dist.plot(kind="pie", autopct="%1.1f%%", colors=["lightcoral", "lightblue"])
plt.title("Target Distribution %")
plt.ylabel("")

plt.tight_layout()
plt.show()

# Calculate target rate
target_rate = (df_pandas["Target"] == "yes").mean()
logger.info(f"📊 Positive Target Rate: {target_rate:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Initialize Feature Engineering Model

# COMMAND ----------

# Initialize our proven FeatureLookUpModel for feature engineering
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Use the successful feature engineering approach
df_features = fe_model._create_engineered_features(df)

logger.info("✅ Feature engineering completed using proven model")

# Show new features created
original_cols = set(df.columns)
new_cols = set(df_features.columns) - original_cols
logger.info(f"🆕 New features created: {sorted(new_cols)}")
logger.info(f"📊 Total features: {len(df_features.columns)}")

# Display sample of engineered features
display(df_features.select(*list(new_cols)[:10]).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feature Analysis and Validation

# COMMAND ----------

# Convert to pandas for analysis
df_features_pandas = df_features.toPandas()

# Feature correlation with target
logger.info("🔍 Feature correlation with target:")

# Convert target to binary for correlation analysis
df_features_pandas["target_binary"] = (df_features_pandas["Target"] == "yes").astype(int)

# Calculate correlations for numerical features
numerical_features = df_features_pandas.select_dtypes(include=[np.number]).columns
correlations = df_features_pandas[numerical_features].corrwith(df_features_pandas["target_binary"])
correlations = correlations.abs().sort_values(ascending=False)

logger.info("📊 Top 15 features by correlation with target:")
display(correlations.head(15))

# Visualize top correlations
plt.figure(figsize=(12, 8))
top_corr = correlations.head(15)
bars = plt.barh(range(len(top_corr)), top_corr.values)
plt.yticks(range(len(top_corr)), top_corr.index)
plt.xlabel("Absolute Correlation with Target")
plt.title("Top 15 Features by Correlation with Target")
plt.gca().invert_yaxis()

# Color bars by correlation strength
for i, bar in enumerate(bars):
    if top_corr.values[i] > 0.15:
        bar.set_color("darkgreen")
    elif top_corr.values[i] > 0.10:
        bar.set_color("green")
    else:
        bar.set_color("lightgreen")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Feature Distribution Analysis

# COMMAND ----------


def analyze_key_feature_distributions():
    """Analyze distributions of key engineered features by target."""

    # Select key engineered features for analysis
    key_features = [
        "job_category",
        "education_level",
        "balance_positive",
        "age_group",
        "campaign_intensity",
        "month_sin",
        "month_cos",
        "has_previous",
    ]

    # Filter features that actually exist
    existing_features = [f for f in key_features if f in df_features_pandas.columns]

    if len(existing_features) >= 6:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()

    for i, feature in enumerate(existing_features[: min(6, len(existing_features))]):
        if feature in df_features_pandas.columns:
            if df_features_pandas[feature].dtype in ["object", "category"]:
                # Categorical feature
                target_rates = df_features_pandas.groupby(feature)["target_binary"].agg(["mean", "count"])
                target_rates = target_rates[target_rates["count"] >= 10]  # Filter low counts

                target_rates["mean"].plot(kind="bar", ax=axes[i], color="skyblue")
                axes[i].set_title(f"Target Rate by {feature}")
                axes[i].set_ylabel("Target Rate")
                axes[i].tick_params(axis="x", rotation=45)
            else:
                # Numerical feature - create bins
                df_features_pandas[feature].hist(bins=20, ax=axes[i], alpha=0.7, color="lightgreen")
                axes[i].set_title(f"Distribution of {feature}")
                axes[i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


analyze_key_feature_distributions()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Feature Quality Validation

# COMMAND ----------


def validate_feature_quality_enhanced(df_pandas):
    """Enhanced feature quality validation."""
    logger.info("🔍 Performing comprehensive feature quality checks...")

    validation_results = {}

    # Check for missing values
    missing_counts = df_pandas.isnull().sum()
    validation_results["missing_values"] = missing_counts[missing_counts > 0].to_dict()

    # Check for infinite values
    numeric_cols = df_pandas.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df_pandas[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    validation_results["infinite_values"] = inf_counts

    # Check for constant features
    constant_features = []
    for col in df_pandas.columns:
        if df_pandas[col].nunique() <= 1:
            constant_features.append(col)
    validation_results["constant_features"] = constant_features

    # Check for highly correlated features (>0.95)
    if len(numeric_cols) > 1:
        corr_matrix = df_pandas[numeric_cols].corr().abs()
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

        validation_results["high_correlations"] = high_corr_pairs

    # Feature statistics
    validation_results["feature_stats"] = {
        "numeric_features": len(numeric_cols),
        "categorical_features": len(df_pandas.select_dtypes(include=["object", "category"]).columns),
        "total_features": len(df_pandas.columns),
        "engineered_features": len(df_pandas.columns)
        - len(config.num_features + config.cat_features + [config.target]),
    }

    return validation_results


# Run enhanced validation
validation_results = validate_feature_quality_enhanced(df_features_pandas)

# Display results with better formatting
logger.info("🔍 Feature Quality Validation Results:")
logger.info(f"📊 Total features: {validation_results['feature_stats']['total_features']}")
logger.info(f"📊 Original features: {len(config.num_features + config.cat_features + [config.target])}")
logger.info(f"📊 Engineered features: {validation_results['feature_stats']['engineered_features']}")
logger.info(f"📊 Numeric features: {validation_results['feature_stats']['numeric_features']}")
logger.info(f"📊 Categorical features: {validation_results['feature_stats']['categorical_features']}")

# Quality checks with icons
quality_checks = [
    ("Missing values", len(validation_results.get("missing_values", {})) == 0),
    ("Infinite values", len(validation_results.get("infinite_values", {})) == 0),
    ("Constant features", len(validation_results.get("constant_features", [])) == 0),
    ("High correlations", len(validation_results.get("high_correlations", [])) == 0),
]

for check_name, passed in quality_checks:
    icon = "✅" if passed else "⚠️"
    logger.info(f"{icon} {check_name}: {'PASS' if passed else 'ISSUES FOUND'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Create Feature Store Tables (Using Proven Approach)

# COMMAND ----------

# Clean existing tables first (using our successful approach)
fe_model._clean_feature_store_tables()

# Create Feature Store tables using our proven method
fe_model.create_feature_store_tables()

logger.info("✅ Feature Store tables created successfully using proven approach")
logger.info(f"📊 Customer features table: {fe_model.customer_features_table}")
logger.info(f"📊 Campaign features table: {fe_model.campaign_features_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Model Training with Engineered Features

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Prepare data for training
df_training = df_features_pandas.copy()

# Convert target to binary
df_training["target_binary"] = (df_training["Target"] == "yes").astype(int)

# Select features for training (exclude target and temp columns)
exclude_cols = ["Target", "target_binary"]
feature_columns = [col for col in df_training.columns if col not in exclude_cols]

# Identify feature types
numeric_features = [col for col in feature_columns if df_training[col].dtype in ["int64", "float64"]]
categorical_features = [col for col in feature_columns if col not in numeric_features]

logger.info(f"📊 Training with {len(feature_columns)} features:")
logger.info(f"   - Numeric: {len(numeric_features)}")
logger.info(f"   - Categorical: {len(categorical_features)}")

# Prepare features and target
X = df_training[feature_columns]
y = df_training["target_binary"]

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=config.parameters["random_state"], stratify=y
)

# Create preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Create model pipeline
model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=config.parameters["n_estimators"],
                max_depth=config.parameters["max_depth"],
                random_state=config.parameters["random_state"],
                n_jobs=-1,  # Use all cores
            ),
        ),
    ]
)

# Train model
logger.info("🚀 Training model with engineered features...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
accuracy = (y_pred == y_test).mean()

logger.info(f"📊 Model Performance with Engineered Features:")
logger.info(f"   - ROC AUC: {roc_auc:.4f}")
logger.info(f"   - Accuracy: {accuracy:.4f}")

# Detailed classification report
print("\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# Enhanced confusion matrix visualization
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Count"})
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")

plt.subplot(1, 2, 2)
# ROC curve would go here, but keeping it simple for demo
target_comparison = pd.DataFrame({"Actual": y_test.values, "Predicted_Prob": y_pred_proba})
target_comparison.groupby("Actual")["Predicted_Prob"].hist(alpha=0.7, bins=20)
plt.legend(["Negative", "Positive"])
plt.title("Prediction Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Feature Importance Analysis

# COMMAND ----------

# Get feature names after preprocessing
cat_feature_names = (
    model.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features)
)
all_feature_names = list(numeric_features) + list(cat_feature_names)

# Create feature importance DataFrame
feature_importance = pd.DataFrame(
    {"feature": all_feature_names, "importance": model.named_steps["classifier"].feature_importances_}
).sort_values("importance", ascending=False)

# Display top features
logger.info("🔝 Top 20 Most Important Features:")
display(feature_importance.head(20))

# Enhanced feature importance visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Top 15 features
top_15 = feature_importance.head(15)
bars1 = ax1.barh(range(len(top_15)), top_15["importance"])
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels(top_15["feature"])
ax1.set_xlabel("Feature Importance")
ax1.set_title("Top 15 Most Important Features")
ax1.invert_yaxis()

# Color bars by importance level
for i, bar in enumerate(bars1):
    if top_15.iloc[i]["importance"] > 0.05:
        bar.set_color("darkgreen")
    elif top_15.iloc[i]["importance"] > 0.03:
        bar.set_color("green")
    else:
        bar.set_color("lightgreen")

# Feature importance distribution
ax2.hist(feature_importance["importance"], bins=30, alpha=0.7, color="skyblue", edgecolor="black")
ax2.set_xlabel("Feature Importance")
ax2.set_ylabel("Number of Features")
ax2.set_title("Distribution of Feature Importances")
ax2.axvline(feature_importance["importance"].mean(), color="red", linestyle="--", label="Mean")
ax2.legend()

plt.tight_layout()
plt.show()

# Analyze engineered vs original features
original_features = set(config.num_features + config.cat_features)
engineered_in_top_20 = sum(
    1 for feat in feature_importance.head(20)["feature"] if not any(orig in feat for orig in original_features)
)
logger.info(f"📊 Engineered features in top 20: {engineered_in_top_20}/20")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Save Results and Log Experiment

# COMMAND ----------

# Save feature importance using TableManager
from bank_marketing.infrastructure.table_manager import TableManager

table_manager = TableManager(spark, config)
feature_importance_spark = spark.createDataFrame(feature_importance)
table_manager.save_table(df=feature_importance_spark, table_name="feature_importance_demo", mode="overwrite")

# Log comprehensive experiment with MLflow
mlflow.set_experiment(config.experiment_name_fe)

with mlflow.start_run(run_name="feature-engineering-demo-enhanced", tags=tags.dict()) as run:
    # Log parameters
    mlflow.log_param("total_features", len(feature_columns))
    mlflow.log_param("numeric_features", len(numeric_features))
    mlflow.log_param("categorical_features", len(categorical_features))
    mlflow.log_param("engineered_features", validation_results["feature_stats"]["engineered_features"])
    mlflow.log_param("model_type", "RandomForest with Enhanced Feature Engineering")
    mlflow.log_param("preprocessing", "StandardScaler + OneHotEncoder")

    # Log model parameters
    mlflow.log_params(config.parameters)

    # Log performance metrics
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("target_rate", target_rate)

    # Log feature engineering metrics
    mlflow.log_metric("features_created", len(new_cols))
    mlflow.log_metric("feature_engineering_ratio", len(new_cols) / len(df.columns))

    # Log quality metrics
    quality_score = sum(passed for _, passed in quality_checks) / len(quality_checks)
    mlflow.log_metric("feature_quality_score", quality_score)

    # Log artifacts
    mlflow.log_dict(validation_results, "feature_validation_enhanced.json")
    mlflow.log_dict(feature_importance.head(50).to_dict(), "feature_importance_top50.json")

    logger.info(f"✅ Enhanced experiment logged: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Performance Comparison & Summary

# COMMAND ----------

# Create performance summary
performance_summary = {
    "Dataset Statistics": {
        "Total Records": f"{df.count():,}",
        "Original Features": len(df.columns),
        "Engineered Features": len(df_features.columns),
        "Features Created": len(new_cols),
        "Target Rate": f"{target_rate:.2%}",
    },
    "Model Performance": {
        "ROC AUC": f"{roc_auc:.4f}",
        "Accuracy": f"{accuracy:.4f}",
        "Training Samples": f"{len(X_train):,}",
        "Test Samples": f"{len(X_test):,}",
    },
    "Feature Quality": {
        "Quality Score": f"{quality_score:.2%}",
        "Missing Values": "✅ None"
        if not validation_results["missing_values"]
        else f"⚠️ {len(validation_results['missing_values'])}",
        "Infinite Values": "✅ None"
        if not validation_results["infinite_values"]
        else f"⚠️ {len(validation_results['infinite_values'])}",
        "Constant Features": "✅ None"
        if not validation_results["constant_features"]
        else f"⚠️ {len(validation_results['constant_features'])}",
    },
    "Feature Store": {
        "Customer Features Table": "✅ Created",
        "Campaign Features Table": "✅ Created",
        "Table Approach": "Simplified with monotonic IDs",
        "Ready for Training": "✅ Yes",
    },
}

# Display comprehensive summary
print("🎉 Feature Engineering Demo - COMPLETE SUMMARY")
print("=" * 60)

for section, metrics in performance_summary.items():
    print(f"\n📊 {section}:")
    for metric, value in metrics.items():
        print(f"   • {metric}: {value}")

print(f"""
🎯 KEY ACHIEVEMENTS:
✅ Successfully engineered {len(new_cols)} new features
✅ Achieved ROC AUC of {roc_auc:.4f} (Excellent performance)
✅ Created reusable Feature Store tables
✅ Implemented proven MLOps patterns
✅ Maintained {quality_score:.0%} feature quality score
""")
