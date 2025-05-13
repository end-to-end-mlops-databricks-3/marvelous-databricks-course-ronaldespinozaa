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
# NUEVO: Importación del VolumeManager
from infrastructure.volume_manager import VolumeManager  
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
# NUEVO: Creación y verificación del volumen

# Verificar que el volumen existe
volume_manager = VolumeManager(spark, config)
volume_manager.ensure_volume_exists()

logger.info(f"📦 Volumen configurado: {volume_manager.volume_path}")

# COMMAND ----------
# NUEVO: Guardar en Volumen

logger.info("💾 Guardando datos en volumen (raw + processed)")
with Timer() as volume_timer:
    data_processor.save_to_volume(X_train, X_test)
logger.info(f"⏱️ Datos guardados en volumen en: {volume_timer}")

# COMMAND ----------
# Guardar en Unity Catalog (mantenemos para compatibilidad)
try:
    logger.info("💾 Guardando train/test en Unity Catalog (raw + processed)")
    data_processor.save_to_catalog(X_train, X_test)
    
    # Activar Change Data Feed para todas las tablas
    if hasattr(data_processor, 'enable_change_data_feed'):
        logger.info("🔁 Activando Change Data Feed para tablas")
        data_processor.enable_change_data_feed()
    else:
        logger.warning("⚠️ Método enable_change_data_feed no encontrado")
        
    logger.info("✅ Datos disponibles tanto en formato raw como procesado")
except Exception as e:
    logger.warning(f"⚠️ No se pudo guardar en Unity Catalog: {e}")
    logger.info("ℹ️ Los datos están disponibles en el volumen")

# COMMAND ----------
# NUEVO: Activar Change Data Feed para volúmenes (opcional)

try:
    logger.info("🔁 Activando Change Data Feed para volúmenes")
    data_processor.enable_volume_change_data_feed()
except Exception as e:
    logger.warning(f"⚠️ No se pudo activar Change Data Feed para volúmenes: {e}")

# COMMAND ----------