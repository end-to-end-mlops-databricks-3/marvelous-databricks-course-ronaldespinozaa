"""Feature Lookup Model implementation for Bank Marketing dataset - SIMPLIFIED VERSION.

This module implements a model that uses Databricks Feature Store
with simplified, practical approach following MLOps best practices.
"""

import mlflow
import numpy as np
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_engineering.entities.feature_lookup import FeatureLookup
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, cos, lower, monotonically_increasing_id, pi, sin, trim, when
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_marketing.config import ProjectConfig, Tags
from infrastructure.table_manager import TableManager


class FeatureLookUpModel:
    """Simplified Feature Lookup Model for Bank Marketing using Databricks Feature Store."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the Feature Lookup Model."""
        self.config = config
        self.spark = spark
        self.tags = tags.dict()

        # Initialize clients
        self.fe_client = FeatureEngineeringClient()
        self.table_manager = TableManager(spark, config)

        # Configuration
        self.num_features = config.num_features
        self.cat_features = config.cat_features
        self.target = config.target
        self.parameters = config.parameters
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name
        self.experiment_name = getattr(config, "experiment_name_fe", "/Shared/bank-marketing-fe")
        self.model_name = f"{self.catalog_name}.{self.schema_name}.bank_marketing_fe_model"

        # Feature Store table names
        self.customer_features_table = f"{self.catalog_name}.{self.schema_name}.customer_features"
        self.campaign_features_table = f"{self.catalog_name}.{self.schema_name}.campaign_features"

        # Ensure schema exists
        self.table_manager.ensure_schema_exists()

    def load_data(self) -> None:
        """Load raw data from CSV."""
        logger.info("🔄 Loading data from CSV...")

        # Load data with absolute path
        data_path = "file:///Workspace/Users/espinozajr52@gmail.com/.bundle/marvelous-databricks-course-ronaldespinozaa/dev/files/data/data.csv"
        self.raw_df = self.spark.read.csv(data_path, header=True, inferSchema=True)

        logger.info(f"✅ Raw data loaded: {self.raw_df.count()} rows, {len(self.raw_df.columns)} columns")

    def _clean_feature_store_tables(self) -> None:
        """Clean existing Feature Store tables - Simple approach."""
        logger.info("🧹 Cleaning existing Feature Store tables...")

        tables_to_clean = [self.customer_features_table, self.campaign_features_table]

        for table_name in tables_to_clean:
            try:
                # Try SQL drop first (most reliable)
                self.spark.sql(f"DROP TABLE IF EXISTS `{table_name}`")
                logger.info(f"🗑️ Cleaned: {table_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not clean {table_name}: {e}")

    def _create_engineered_features(self, df: DataFrame) -> DataFrame:
        """Create all engineered features - Simplified version."""
        logger.info("🔧 Creating engineered features...")

        # Clean categorical features
        cat_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
        for col_name in cat_cols:
            if col_name in df.columns:
                df = df.withColumn(col_name, trim(lower(col(col_name))))

        # Key engineered features (simplified)
        df = df.withColumn(
            "job_category",
            when(col("job").isin(["management", "admin"]), "management")
            .when(col("job").isin(["technician", "services"]), "services")
            .when(col("job").isin(["blue-collar"]), "manual")
            .otherwise("other"),
        )

        df = df.withColumn(
            "education_level",
            when(col("education") == "tertiary", 3)
            .when(col("education") == "secondary", 2)
            .when(col("education") == "primary", 1)
            .otherwise(0),
        )

        df = df.withColumn("balance_positive", when(col("balance") > 0, 1).otherwise(0))
        df = df.withColumn(
            "age_group", when(col("age") < 30, "young").when(col("age") < 50, "middle").otherwise("senior")
        )

        # Monthly features (simplified)
        month_num = (
            when(col("month") == "jan", 1)
            .when(col("month") == "feb", 2)
            .when(col("month") == "mar", 3)
            .when(col("month") == "apr", 4)
            .when(col("month") == "may", 5)
            .when(col("month") == "jun", 6)
            .when(col("month") == "jul", 7)
            .when(col("month") == "aug", 8)
            .when(col("month") == "sep", 9)
            .when(col("month") == "oct", 10)
            .when(col("month") == "nov", 11)
            .when(col("month") == "dec", 12)
            .otherwise(6)
        )

        df = df.withColumn("month_number", month_num)
        df = df.withColumn("month_sin", sin(2 * pi() * col("month_number") / 12))
        df = df.withColumn("month_cos", cos(2 * pi() * col("month_number") / 12))

        # Campaign features (simplified)
        df = df.withColumn(
            "campaign_intensity",
            when(col("campaign") <= 2, "low").when(col("campaign") <= 5, "medium").otherwise("high"),
        )

        df = df.withColumn("has_previous", when(col("previous") > 0, 1).otherwise(0))

        logger.info("✅ Engineered features created")
        return df

    def create_feature_store_tables(self) -> None:
        """Create Feature Store tables - Simplified approach."""
        logger.info("🏗️ Creating Feature Store tables...")

        # Clean existing tables first
        self._clean_feature_store_tables()

        # Create engineered features
        df_features = self._create_engineered_features(self.raw_df)

        # Create customer features table
        customer_df = df_features.select(
            "age",
            "job",
            "marital",
            "education",
            "balance",
            "job_category",
            "education_level",
            "balance_positive",
            "age_group",
        ).distinct()

        # Add simple numeric ID
        customer_df = customer_df.withColumn("customer_id", monotonically_increasing_id())

        # Create customer table
        self.fe_client.create_table(
            name=self.customer_features_table,
            primary_keys=["customer_id"],
            df=customer_df,
            description="Customer features for bank marketing",
        )
        logger.info(f"✅ Customer table created: {self.customer_features_table}")

        # Create campaign features table
        campaign_df = df_features.select(
            "contact",
            "month",
            "duration",
            "campaign",
            "previous",
            "month_number",
            "month_sin",
            "month_cos",
            "campaign_intensity",
            "has_previous",
        ).distinct()

        # Add simple numeric ID
        campaign_df = campaign_df.withColumn("campaign_id", monotonically_increasing_id())

        # Create campaign table
        self.fe_client.create_table(
            name=self.campaign_features_table,
            primary_keys=["campaign_id"],
            df=campaign_df,
            description="Campaign features for bank marketing",
        )
        logger.info(f"✅ Campaign table created: {self.campaign_features_table}")

    def create_training_dataset(self) -> None:
        """Create training dataset using Feature Store - Simplified."""
        logger.info("📊 Creating training dataset with Feature Store...")

        # Get base data with IDs
        df_base = self._create_engineered_features(self.raw_df)

        # Add matching IDs with EXPLICIT LONG casting
        from pyspark.sql.functions import row_number
        from pyspark.sql.window import Window

        window_spec = Window.orderBy(col("age"))  # Simple ordering
        df_with_ids = df_base.withColumn("customer_id", row_number().over(window_spec).cast("long")).withColumn(
            "campaign_id", row_number().over(window_spec).cast("long")
        )

        # Define feature lookups
        feature_lookups = [
            FeatureLookup(
                table_name=self.customer_features_table,
                lookup_key="customer_id",
                feature_names=["job_category", "education_level", "balance_positive", "age_group"],
            ),
            FeatureLookup(
                table_name=self.campaign_features_table,
                lookup_key="campaign_id",
                feature_names=["month_sin", "month_cos", "campaign_intensity", "has_previous"],
            ),
        ]

        # Create training set
        base_features = self.num_features + self.cat_features + [self.target, "customer_id", "campaign_id"]

        self.training_set = self.fe_client.create_training_set(
            df=df_with_ids.select(*base_features),
            feature_lookups=feature_lookups,
            label=self.target,
            exclude_columns=["customer_id", "campaign_id"],
        )

        # Load as pandas
        self.training_df = self.training_set.load_df().toPandas()

        # Convert target
        if self.training_df[self.target].dtype == "object":
            self.training_df[self.target] = (self.training_df[self.target] == "yes").astype(int)

        logger.info(f"✅ Training dataset created: {self.training_df.shape}")

    def train_model(self) -> None:
        """Train model with Feature Store features."""
        logger.info("🚀 Training model...")

        # Prepare features
        feature_cols = [col for col in self.training_df.columns if col != self.target]
        X = self.training_df[feature_cols]
        y = self.training_df[self.target]

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.parameters["random_state"]
        )

        # Create pipeline
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        self.pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", LGBMClassifier(**self.parameters))])

        # Train
        self.pipeline.fit(self.X_train, self.y_train)
        logger.info("✅ Model training completed")

    def register_model(self) -> None:
        """Register model with MLflow and Feature Store."""
        logger.info("📝 Registering model...")

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            # Calculate metrics
            y_pred = self.pipeline.predict(self.X_test)
            y_pred_proba = self.pipeline.predict_proba(self.X_test)[:, 1]

            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            accuracy = accuracy_score(self.y_test, y_pred)

            # Log metrics
            mlflow.log_params(self.parameters)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("accuracy", accuracy)

            # Log model with Feature Store
            self.fe_client.log_model(
                model=self.pipeline,
                artifact_path="model",
                flavor=mlflow.sklearn,
                training_set=self.training_set,
                registered_model_name=self.model_name,
            )

            logger.info(f"✅ Model registered: {self.model_name} (AUC: {roc_auc:.4f})")

        # Set alias using Unity Catalog - simplified approach
        try:
            client = MlflowClient()
            # Get all versions and take the latest
            all_versions = client.search_model_versions(f"name='{self.model_name}'")
            if all_versions:
                latest_version = max(int(v.version) for v in all_versions)
                client.set_registered_model_alias(self.model_name, "latest", str(latest_version))
                logger.info(f"✅ Alias 'latest' set for version {latest_version}")
        except Exception as e:
            logger.warning(f"⚠️ Could not set alias: {e}")
            logger.info("✅ Model registration completed without alias")

    def predict(self, input_data: DataFrame) -> dict:
        """Make predictions using registered model."""
        logger.info("🔮 Making predictions...")

        # Use pyfunc instead of sklearn for Feature Store models
        model_uri = f"models:/{self.model_name}/1"
        model = mlflow.pyfunc.load_model(model_uri)

        if hasattr(input_data, "toPandas"):
            input_data = input_data.toPandas()

        # Use predict instead of predict_proba for pyfunc
        predictions_result = model.predict(input_data)

        # Handle different output formats
        if isinstance(predictions_result, np.ndarray):
            if predictions_result.ndim == 1:
                # Binary predictions
                predictions = predictions_result.astype(int).tolist()
                probabilities = predictions_result.tolist()  # Use same values as probabilities
            else:
                # Probability matrix
                predictions = (predictions_result[:, 1] > 0.5).astype(int).tolist()
                probabilities = predictions_result[:, 1].tolist()
        else:
            # Fallback
            predictions = predictions_result.tolist()
            probabilities = predictions_result.tolist()

        return {"predictions": predictions, "probabilities": probabilities}

    # Convenience methods for easy usage
    def run_full_pipeline(self) -> None:
        """Run the complete Feature Store pipeline."""
        logger.info("🚀 Running complete Feature Store pipeline...")

        self.load_data()
        self.create_feature_store_tables()
        self.create_training_dataset()
        self.train_model()
        self.register_model()

        logger.info("🎉 Complete pipeline finished!")

    def get_model_performance(self) -> dict:
        """Get model performance metrics."""
        if hasattr(self, "X_test") and hasattr(self, "y_test"):
            y_pred = self.pipeline.predict(self.X_test)
            y_pred_proba = self.pipeline.predict_proba(self.X_test)[:, 1]

            return {
                "roc_auc": roc_auc_score(self.y_test, y_pred_proba),
                "accuracy": accuracy_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred),
                "recall": recall_score(self.y_test, y_pred),
                "f1": f1_score(self.y_test, y_pred),
            }
        return {}
