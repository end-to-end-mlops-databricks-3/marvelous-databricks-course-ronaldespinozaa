# Databricks notebook source
# MAGIC %md
# MAGIC # Logging and Registering Models with MLflow
# MAGIC
# MAGIC Este notebook demuestra diferentes enfoques para registrar modelos con MLflow
# MAGIC usando Databricks Asset Bundles y Unity Catalog.

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

# Importar versión del paquete si está disponible
try:
    from bank_marketing import __version__
except ImportError:
    __version__ = "0.1.0"  # Versión por defecto

from mlflow.utils.environment import _mlflow_conda_env

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración de MLflow para Databricks Asset Bundles

# COMMAND ----------

# Configurar MLflow para Unity Catalog (esto es automático en Databricks)
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    print("✅ Ejecutando en Databricks - configuración automática")
    # En Databricks, MLflow ya está configurado para Unity Catalog
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
else:
    print("⚠️ Ejecutando localmente")
    # Para desarrollo local con DAB
    try:
        from databricks.connect import DatabricksSession

        spark = DatabricksSession.builder.getOrCreate()
        print("✅ Conectado usando Databricks Connect")
    except ImportError:
        print("❌ Databricks Connect no disponible, usando Spark local")
        spark = SparkSession.builder.getOrCreate()

# Cargar configuración
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Inicializar Spark y cargar datos

# COMMAND ----------

# Obtener SparkSession (en Databricks ya está disponible como 'spark')
try:
    # En Databricks, 'spark' ya está definido
    spark = spark
    print("✅ Usando SparkSession de Databricks")
except NameError:
    # Para ejecución local
    spark = SparkSession.builder.getOrCreate()
    print("✅ SparkSession local creado")

# Cargar datos de entrenamiento con manejo de errores
try:
    # Intentar cargar desde Unity Catalog
    train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_processed").toPandas()
    print("✅ Datos cargados desde Unity Catalog")
except Exception as e:
    print(f"⚠️ No se pudo cargar desde Unity Catalog: {e}")

    # Crear datos sintéticos para demostración
    import numpy as np

    np.random.seed(42)

    # Crear datos que coincidan con tu configuración
    n_samples = 1000
    synthetic_data = {}

    # Crear características numéricas
    for col in config.num_features:
        if col == "age":
            synthetic_data[col] = np.random.randint(18, 70, n_samples)
        elif col == "balance":
            synthetic_data[col] = np.random.normal(1000, 500, n_samples)
        else:
            synthetic_data[col] = np.random.normal(0, 1, n_samples)

    # Crear características categóricas
    for col in config.cat_features:
        if col == "job":
            synthetic_data[col] = np.random.choice(["admin", "blue-collar", "management", "technician"], n_samples)
        elif col == "marital":
            synthetic_data[col] = np.random.choice(["married", "single", "divorced"], n_samples)
        else:
            synthetic_data[col] = np.random.choice(["A", "B", "C"], n_samples)

    # Crear target
    synthetic_data[config.target] = np.random.choice([0, 1], n_samples)

    train_set = pd.DataFrame(synthetic_data)
    print("✅ Datos sintéticos creados")

# Separar features y target
X_train = train_set[config.num_features + config.cat_features]
y_train = train_set[config.target]

print(f"📊 Forma de datos: {X_train.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Crear y entrenar un pipeline básico

# COMMAND ----------

# Definir el pipeline con preprocesamiento para clasificación
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

# Combinar transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, config.num_features),
        ("cat", categorical_transformer, config.cat_features),
    ]
)

# Crear el pipeline completo con clasificador
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**config.parameters))])

# Entrenar el modelo
pipeline.fit(X_train, y_train)
print("✅ Modelo entrenado correctamente")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Loguear el modelo con MLflow usando Unity Catalog

# COMMAND ----------

# Configurar experimento usando la estructura de DAB
experiment_name = f"/{config.schema_name}/bank-marketing-demo"
mlflow.set_experiment(experiment_name)

# Iniciar run y loguear modelo
with mlflow.start_run(
    run_name="dab-demo-model",
    tags={"git_sha": "dab123", "branch": "main", "bundle": "marvelous-databricks-course-ronaldespinozaa"},
    description="Modelo demo usando Databricks Asset Bundle",
) as run:
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM con preprocesamiento - DAB")
    mlflow.log_params(config.parameters)

    # Calcular y loguear métricas
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

    print(f"📊 Métricas - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

    # Inferir firma del modelo
    signature = infer_signature(model_input=X_train, model_output=y_pred_proba)

    # Loguear el modelo
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)

    print(f"✅ Modelo logueado con run_id: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Registrar modelo en Unity Catalog (formato correcto para DAB)

# COMMAND ----------

# Definir nombre del modelo usando la estructura correcta para Unity Catalog
model_name = f"{config.catalog_name}.{config.schema_name}.bank_marketing_model_demo"

try:
    # Registrar modelo
    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/lightgbm-pipeline-model",
        name=model_name,
        tags={"bundle": "marvelous-databricks-course-ronaldespinozaa", "git_sha": "dab123"},
    )

    print(f"✅ Modelo registrado: {model_name} versión {model_version.version}")

    # Configurar alias usando MLflow Client
    client = MlflowClient()

    # Configurar alias "dab-latest"
    client.set_registered_model_alias(name=model_name, alias="dab-latest", version=model_version.version)
    print(f"✅ Alias 'dab-latest' configurado para versión {model_version.version}")

except Exception as e:
    print(f"⚠️ Error al registrar modelo: {e}")
    print("Esto puede suceder si no tienes permisos en Unity Catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Cargar modelo y hacer predicciones

# COMMAND ----------

try:
    # Cargar modelo usando el alias
    model_uri = f"models:/{model_name}@dab-latest"
    sklearn_pipeline = mlflow.sklearn.load_model(model_uri)

    # Hacer predicciones en una muestra
    sample_data = X_train.iloc[:5]
    predictions = sklearn_pipeline.predict(sample_data)
    probabilities = sklearn_pipeline.predict_proba(sample_data)[:, 1]

    # Mostrar resultados
    result_df = pd.DataFrame(
        {"prediction": predictions, "probability": probabilities, "actual": y_train.iloc[:5].values}
    )

    display(result_df)
    print("✅ Predicciones realizadas exitosamente")

except Exception as e:
    print(f"⚠️ Error al cargar modelo: {e}")
    print("Usando modelo en memoria para demostración...")

    # Fallback a modelo en memoria
    sample_data = X_train.iloc[:5]
    predictions = pipeline.predict(sample_data)
    probabilities = pipeline.predict_proba(sample_data)[:, 1]

    result_df = pd.DataFrame(
        {"prediction": predictions, "probability": probabilities, "actual": y_train.iloc[:5].values}
    )

    display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Modelo personalizado con pyfunc y DAB

# COMMAND ----------


# Clase wrapper para modelo personalizado
class BankMarketingModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper personalizado para el modelo de Bank Marketing con DAB."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            # Obtener predicciones
            predictions = self.model.predict(model_input)
            probabilities = self.model.predict_proba(model_input)[:, 1]
