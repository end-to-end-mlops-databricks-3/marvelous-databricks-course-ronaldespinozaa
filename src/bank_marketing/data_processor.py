import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

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
        """Save processed data into Delta tables in the Unity Catalog."""
        # Crear DataFrame de Spark a partir de DataFrame de Pandas para poder usar Delta Lake
        train_sdf = self.spark.createDataFrame(train_df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        test_sdf = self.spark.createDataFrame(test_df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        # Guardar como tablas Delta
        train_sdf.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )
        test_sdf.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Delta Lake Change Data Feed on the saved tables."""
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
