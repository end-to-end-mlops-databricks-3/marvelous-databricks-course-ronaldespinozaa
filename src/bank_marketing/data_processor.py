import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split
from infrastructure.volume_manager import VolumeManager

from bank_marketing.config import ProjectConfig


class DataProcessor:
    """Preprocesses, splits, and stores the Bank Marketing dataset."""

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df
        self.config = config
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the raw dataset.

        Converts column types, handles missing values, encodes categorical variables,
        and selects only relevant features.
        """
        target_column = self.config.target
        print(f"Columna objetivo definida en config: {target_column}")

        if target_column not in self.df.columns:
            raise KeyError(f"La columna '{target_column}' no está presente en el DataFrame.")

        # Convertir columnas numéricas a tipo numérico (con coerción de errores)
        for col in self.config.num_features:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Convertir columnas categóricas a tipo category
        for col in self.config.cat_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")

        # Reemplazar 'unknown' por pd.NA
        self.df.replace("unknown", pd.NA, inplace=True)

        # Agregar categoría 'missing' a columnas categóricas y rellenar NaNs
        for col in self.config.cat_features:
            if col in self.df.columns:
                if "missing" not in self.df[col].cat.categories:
                    self.df[col] = self.df[col].cat.add_categories("missing")
                self.df[col] = self.df[col].fillna("missing")

        # Rellenar NaNs en columnas numéricas (ejemplo: con 0, podrías usar la media si prefieres)
        for col in self.config.num_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)

        # Convertir la columna objetivo a binaria
        if target_column in self.df.columns:
            self.df[target_column] = self.df[target_column].map({"yes": 1, "no": 0}).fillna(0).astype(int)

        # Asegurarse de que las columnas categóricas tienen el tipo correcto
        for col in self.config.cat_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")

        # Filtrar las columnas relevantes
        features = self.config.cat_features + self.config.num_features + [target_column]
        print(f"Características a mantener: {features}")

        # Verificar existencia de todas las columnas necesarias
        missing_columns = [col for col in features if col not in self.df.columns]
        if missing_columns:
            raise KeyError(f"Las siguientes columnas no están en el DataFrame: {missing_columns}")

        # Dejar solo las columnas deseadas
        self.df = self.df[features]

        
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data into training and testing sets."""
        train_df, test_df = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_df, test_df

    def save_to_catalog(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Save processed data into Delta tables in the Unity Catalog with MLOps best practices."""
        # 1. Guardar versión RAW (target como string) para auditoría y referencia
        train_raw = train_df.copy()
        test_raw = test_df.copy()
        
        # Convertir target a string si es numérico
        target_column = self.config.target
        if target_column in train_raw.columns and pd.api.types.is_numeric_dtype(train_raw[target_column]):
            train_raw[target_column] = train_raw[target_column].map({1: "yes", 0: "no"})
            test_raw[target_column] = test_raw[target_column].map({1: "yes", 0: "no"})
        
        # Crear Spark DataFrames
        train_raw_sdf = self.spark.createDataFrame(train_raw).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        test_raw_sdf = self.spark.createDataFrame(test_raw).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        
        # Eliminar tablas raw existentes
        self.spark.sql(f"DROP TABLE IF EXISTS {self.config.catalog_name}.{self.config.schema_name}.train_raw")
        self.spark.sql(f"DROP TABLE IF EXISTS {self.config.catalog_name}.{self.config.schema_name}.test_raw")
        
        # Guardar versión raw
        train_raw_sdf.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_raw"
        )
        test_raw_sdf.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_raw"
        )
        
        # 2. Guardar versión PROCESADA (target como numérico) para entrenamiento
        # Eliminar tablas procesadas existentes
        self.spark.sql(f"DROP TABLE IF EXISTS {self.config.catalog_name}.{self.config.schema_name}.train_processed")
        self.spark.sql(f"DROP TABLE IF EXISTS {self.config.catalog_name}.{self.config.schema_name}.test_processed")
        
        # Crear Spark DataFrames para datos procesados
        train_processed_sdf = self.spark.createDataFrame(train_df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        test_processed_sdf = self.spark.createDataFrame(test_df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        
        # Guardar versión procesada con schema overwrite para manejar cambios de tipo
        train_processed_sdf.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_processed"
        )
        test_processed_sdf.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_processed"
        )
        
        print(f"Datos guardados en Unity Catalog:")
        print(f"  - Datos raw (target como string): {self.config.catalog_name}.{self.config.schema_name}.train_raw")
        print(f"  - Datos procesados (target numérico): {self.config.catalog_name}.{self.config.schema_name}.train_processed")

    def enable_change_data_feed(self) -> None:
        """Enable Delta Lake Change Data Feed on all catalog tables."""
        # Tablas raw
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_raw "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_raw "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
        
        # Tablas procesadas
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_processed "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_processed "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

    def save_to_volume(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Save data to volumes in both raw (string target) and processed (numeric target) formats."""
        
        # Inicializar volumen
        volume_manager = VolumeManager(self.spark, self.config)
        volume_manager.ensure_volume_exists()
        
        # Obtener nombre de columna objetivo
        target_column = self.config.target
        
        # 1. Guardar datos raw (con target como string)
        train_raw = train_df.copy()
        test_raw = test_df.copy()
        
        # Convertir target de nuevo a string si es numérico
        if target_column in train_raw.columns and pd.api.types.is_numeric_dtype(train_raw[target_column]):
            train_raw[target_column] = train_raw[target_column].map({1: "yes", 0: "no"})
            test_raw[target_column] = test_raw[target_column].map({1: "yes", 0: "no"})
        
        # Crear Spark DataFrames
        train_raw_sdf = self.spark.createDataFrame(train_raw)
        test_raw_sdf = self.spark.createDataFrame(test_raw)
        
        # Añadir timestamp
        train_raw_sdf = train_raw_sdf.withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        test_raw_sdf = test_raw_sdf.withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        
        # Guardar versión raw CON OPCIÓN overwriteSchema
        raw_train_path = volume_manager.get_path("raw", "train")
        raw_test_path = volume_manager.get_path("raw", "test")
        
        train_raw_sdf.write.format("delta")\
            .option("overwriteSchema", "true")\
            .mode("overwrite")\
            .save(raw_train_path)
        
        test_raw_sdf.write.format("delta")\
            .option("overwriteSchema", "true")\
            .mode("overwrite")\
            .save(raw_test_path)
        
        # 2. Guardar versión procesada (con target numérico)
        train_processed_sdf = self.spark.createDataFrame(train_df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        test_processed_sdf = self.spark.createDataFrame(test_df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        
        # Guardar versión procesada CON OPCIÓN overwriteSchema
        processed_train_path = volume_manager.get_path("processed", "train")
        processed_test_path = volume_manager.get_path("processed", "test")
        
        train_processed_sdf.write.format("delta")\
            .option("overwriteSchema", "true")\
            .mode("overwrite")\
            .save(processed_train_path)
        
        test_processed_sdf.write.format("delta")\
            .option("overwriteSchema", "true")\
            .mode("overwrite")\
            .save(processed_test_path)
        
        print(f"Datos guardados en volumen: {volume_manager.volume_path}")
        print(f"  - Datos raw: {volume_manager.get_path('raw')}")
        print(f"  - Datos procesados: {volume_manager.get_path('processed')}")

    def enable_volume_change_data_feed(self) -> None:
        """Enable Delta Lake Change Data Feed on the saved volume tables."""
        
        volume_manager = VolumeManager(self.spark, self.config)
        
        # Obtener rutas para los datos
        raw_train_path = volume_manager.get_path("raw", "train")
        raw_test_path = volume_manager.get_path("raw", "test")
        processed_train_path = volume_manager.get_path("processed", "train")
        processed_test_path = volume_manager.get_path("processed", "test")
        
        # Habilitar Change Data Feed para todas las tablas
        for path in [raw_train_path, raw_test_path, processed_train_path, processed_test_path]:
            self.spark.sql(f"ALTER TABLE delta.`{path}` SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
        
        print("Change Data Feed activado para todas las tablas en volumen")