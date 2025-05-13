import argparse
import os
print(os.getcwd())
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from bank_marketing.config import ProjectConfig
from bank_marketing.data_processor import DataProcessor
from marvelous.logging import setup_logging
from marvelous.timer import Timer


def main(env: str):
    # Ruta robusta al archivo YAML de configuración
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "../project_config.yml")

    # Carga de configuración
    config = ProjectConfig.from_yaml(config_path=config_path, env=env)

    # Setup de logging (guardar en ruta basada en el catálogo y esquema)
    log_path = f"/Volumes/{config.catalog_name}/{config.schema_name}/logs/marvelous-preprocess.log"
    setup_logging(log_file=log_path)

    logger.info("✅ Configuración cargada correctamente")
    logger.info(yaml.dump(config, default_flow_style=False))

    # Inicializa Spark
    spark = SparkSession.builder.getOrCreate()

    # Ruta del dataset
    data_path = f"/Volumes/{config.catalog_name}/{config.schema_name}/data/data.csv"
    
    # Validación de existencia del dataset
    if not spark._jvm.org.apache.hadoop.fs.FileSystem.get(
        spark._jsc.hadoopConfiguration()
    ).exists(spark._jvm.org.apache.hadoop.fs.Path(data_path)):
        logger.error(f"❌ El archivo no existe en: {data_path}")
        return

    # Carga de datos desde Unity Catalog
    logger.info("📥 Cargando datos desde Unity Catalog...")
    df = spark.read.csv(data_path, header=True, inferSchema=True).toPandas()

    # Preprocesamiento
    logger.info("⚙️ Iniciando preprocesamiento de datos...")
    with Timer() as preprocess_timer:
        data_processor = DataProcessor(df, config, spark)
        data_processor.preprocess()
    logger.info(f"⏱️ Tiempo de preprocesamiento: {preprocess_timer}")

    # Split de datos
    X_train, X_test = data_processor.split_data()
    logger.info(f"📊 Shape del training set: {X_train.shape}")
    logger.info(f"📊 Shape del test set: {X_test.shape}")

    # Guardado en Unity Catalog
    logger.info("💾 Guardando datos en Unity Catalog...")
    data_processor.save_to_catalog(X_train, X_test)
    logger.info("✅ Proceso completado correctamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesamiento de datos de bank marketing")
    parser.add_argument("--env", type=str, default="dev", help="Entorno: dev, acc, prd")
    args = parser.parse_args()

    main(env=args.env)
