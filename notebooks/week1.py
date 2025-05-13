# Databricks notebook source
# %pip install -e ..
# %restart_python

# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------
# Iniciar instalaciones 

from loguru import logger
import yaml
from pyspark.sql import SparkSession
import pandas as pd

from bank_marketing.config import ProjectConfig
from bank_marketing.data_processor import DataProcessor
from marvelous.logging import setup_logging
from marvelous.timer import Timer

# Cargar configuración desde YAML
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

# Setup del logging
setup_logging(log_file="logs/marvelous-1.log")

logger.info("✅ Configuración cargada:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Iniciar Spark y cargar CSV desde Unity Catalog
spark = SparkSession.builder.getOrCreate()

filepath = "../data/data.csv"

# Load the data
df = pd.read_csv(filepath)

logger.info(f"📥 Datos cargados desde: {filepath} - Shape: {df.shape}")

# COMMAND ----------
# Preprocesamiento
with Timer() as preprocess_timer:
    data_processor = DataProcessor(df, config, spark)
    data_processor.preprocess()
logger.info(f"⚙️ Preprocesamiento completado en: {preprocess_timer}")

# COMMAND ----------
# División train/test
X_train, X_test = data_processor.split_data()
logger.info(f"📊 Train shape: {X_train.shape} | Test shape: {X_test.shape}")

# COMMAND ----------
# Guardar en Unity Catalog

# Guardar en Unity Catalog

logger.info("💾 Guardando train/test en Unity Catalog")
data_processor.save_to_catalog(X_train, X_test)

# Activar Change Data Feed
logger.info("🔁 Activando Change Data Feed")
data_processor.enable_change_data_feed()
# COMMAND ----------
