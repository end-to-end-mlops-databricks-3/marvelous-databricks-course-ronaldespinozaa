"""Unit tests for DataProcessor."""

import pandas as pd
import pytest
from conftest import CATALOG_DIR
from delta.tables import DeltaTable
from pyspark.sql import SparkSession

from bank_marketing.config import ProjectConfig
from bank_marketing.data_processor import DataProcessor


def test_data_ingestion(sample_bank_data: pd.DataFrame) -> None:
    """Test the data ingestion process by checking the shape of the sample data.

    Asserts that the sample data has at least one row and one column.

    :param sample_bank_data: The sample data to be tested
    """
    assert sample_bank_data.shape[0] > 0
    assert sample_bank_data.shape[1] > 0


def test_dataprocessor_init(
    sample_bank_data: pd.DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession,
) -> None:
    """Test the initialization of DataProcessor.

    :param sample_bank_data: Sample DataFrame for testing
    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(pandas_df=sample_bank_data, config=config, spark=spark_session)
    assert isinstance(processor.df, pd.DataFrame)
    assert processor.df.equals(sample_bank_data)

    assert isinstance(processor.config, ProjectConfig)
    assert isinstance(processor.spark, SparkSession)


def test_preprocess_data_types(
    data_processor_input: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test data type transformations performed by the DataProcessor.

    Verifies that numeric columns are converted to numeric types and categorical columns to categories.

    :param data_processor_input: Input DataFrame with prepared test data
    :param config: Configuration object for the project
    :param spark_session: SparkSession object
    """
    processor = DataProcessor(pandas_df=data_processor_input, config=config, spark=spark_session)
    processor.preprocess()

    # Verificar que las columnas numéricas son de tipo numérico
    for col in config.num_features:
        assert pd.api.types.is_numeric_dtype(processor.df[col]), f"Column {col} should be numeric"

    # Verificar que las columnas categóricas son de tipo category
    for col in config.cat_features:
        assert pd.api.types.is_categorical_dtype(processor.df[col]), f"Column {col} should be category"


def test_missing_value_handling(
    data_processor_input: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test missing value handling in the DataProcessor.

    Verifies that 'unknown' values are handled properly and missing values are filled.

    :param data_processor_input: Input DataFrame with prepared test data
    :param config: Configuration object for the project
    :param spark_session: SparkSession object
    """
    # Asegurar que hay algunos valores 'unknown' para probar
    for col in config.cat_features[:2]:  # Usar solo algunas columnas
        if len(data_processor_input) > 2:
            data_processor_input.loc[1, col] = "unknown"

    # Asegurar que hay algunos valores NaN para probar
    for col in config.num_features[:2]:  # Usar solo algunas columnas
        if len(data_processor_input) > 2:
            data_processor_input.loc[2, col] = None

    processor = DataProcessor(pandas_df=data_processor_input, config=config, spark=spark_session)
    processor.preprocess()

    # Verificar que no hay valores faltantes en columnas numéricas
    for col in config.num_features:
        assert processor.df[col].isna().sum() == 0, f"Column {col} should not have NaN values"

    # Verificar que no hay valores 'unknown' en columnas categóricas
    for col in config.cat_features:
        assert "unknown" not in processor.df[col].values, f"Column {col} should not have 'unknown' values"
        assert processor.df[col].isna().sum() == 0, f"Column {col} should not have NaN values"


def test_column_selection(
    data_processor_input: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test column selection in the DataProcessor.

    Verifies that the processed DataFrame contains only the relevant features and target.

    :param data_processor_input: Input DataFrame with prepared test data
    :param config: Configuration object for the project
    :param spark_session: SparkSession object
    """
    processor = DataProcessor(pandas_df=data_processor_input, config=config, spark=spark_session)
    processor.preprocess()

    expected_columns = config.num_features + config.cat_features + [config.target]
    assert set(processor.df.columns) == set(expected_columns), "Columns in processed data don't match expected columns"


def test_target_encoding(
    data_processor_input: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test that the target column is properly encoded.

    Verifies that the target column is converted to binary values.

    :param data_processor_input: Input DataFrame with prepared test data
    :param config: Configuration object for the project
    :param spark_session: SparkSession object
    """
    # Asegurar que el target tiene valores string 'yes'/'no'
    if config.target in data_processor_input.columns:
        data_processor_input[config.target] = data_processor_input[config.target].map(
            lambda x: "yes"
            if isinstance(x, int | float) and x > 0
            else "no"  # Corrected: Used `int | float` instead of `(int, float)`
        )

    processor = DataProcessor(pandas_df=data_processor_input, config=config, spark=spark_session)
    processor.preprocess()

    # Verificar que el target es numérico y solo contiene 0 y 1
    assert pd.api.types.is_numeric_dtype(processor.df[config.target]), "Target should be numeric"
    assert set(processor.df[config.target].unique()).issubset({0, 1}), "Target should only contain 0 and 1"


def test_split_data_default_params(
    data_processor_input: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test the default parameters of the split_data method in DataProcessor.

    Verifies that the split_data method correctly splits the input DataFrame
    into train and test sets using default parameters.

    :param data_processor_input: Input DataFrame to be split
    :param config: Configuration object for the project
    :param spark_session: SparkSession object
    """
    processor = DataProcessor(pandas_df=data_processor_input, config=config, spark=spark_session)
    processor.preprocess()
    train, test = processor.split_data()

    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(test) == len(processor.df)
    assert set(train.columns) == set(test.columns) == set(processor.df.columns)

    # Guardar datos para otras pruebas
    train.to_csv((CATALOG_DIR / "train_processed.csv").as_posix(), index=False)
    test.to_csv((CATALOG_DIR / "test_processed.csv").as_posix(), index=False)


def test_split_data_custom_params(
    data_processor_input: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test custom parameters for the split_data method.

    Verifies that different test_size and random_state parameters produce the expected splits.

    :param data_processor_input: Input DataFrame to be split
    :param config: Configuration object for the project
    :param spark_session: SparkSession object
    """
    processor = DataProcessor(pandas_df=data_processor_input, config=config, spark=spark_session)
    processor.preprocess()

    # Probar con un test_size más grande
    test_size = 0.3
    train, test = processor.split_data(test_size=test_size, random_state=99)

    # Verificar proporciones (con cierta tolerancia debido a redondeo)
    expected_test_ratio = test_size
    actual_test_ratio = len(test) / (len(train) + len(test))
    assert abs(actual_test_ratio - expected_test_ratio) < 0.1, (
        f"Expected test ratio {expected_test_ratio}, got {actual_test_ratio}"
    )


def test_preprocess_empty_dataframe(config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test the preprocess method with an empty DataFrame.

    Verifies that the preprocess method correctly handles an empty DataFrame
    and raises KeyError.

    :param config: Configuration object for the project
    :param spark_session: SparkSession object
    """
    processor = DataProcessor(pandas_df=pd.DataFrame([]), config=config, spark=spark_session)
    with pytest.raises(KeyError):
        processor.preprocess()


@pytest.mark.skip(reason="depends on delta tables on Databricks")
def test_save_to_catalog_succesfull(
    data_processor_input: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test the successful saving of data to the catalog.

    Processes input data, splits it, and saves to the catalog. Verifies that tables exist.

    :param data_processor_input: The input data to be processed and saved
    :param config: Configuration object for the project
    :param spark_session: SparkSession object for interacting with Spark
    """
    processor = DataProcessor(pandas_df=data_processor_input, config=config, spark=spark_session)
    processor.preprocess()
    train_set, test_set = processor.split_data()
    processor.save_to_catalog(train_set, test_set)
    processor.enable_change_data_feed()

    # Assert
    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.train_processed")
    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.test_processed")


@pytest.mark.skip(reason="depends on delta tables on Databricks")
@pytest.mark.order(after=test_save_to_catalog_succesfull)
def test_delta_table_property_of_enableChangeDataFeed_check(config: ProjectConfig, spark_session: SparkSession) -> None:
    """Check if Change Data Feed is enabled for train and test sets.

    Verifies that the 'delta.enableChangeDataFeed' property is set to True for both
    the train and test set Delta tables.

    :param config: Project configuration object
    :param spark_session: SparkSession object
    """
    train_set_path = f"{config.catalog_name}.{config.schema_name}.train_processed"
    test_set_path = f"{config.catalog_name}.{config.schema_name}.test_processed"
    tables = [train_set_path, test_set_path]
    for table in tables:
        delta_table = DeltaTable.forName(spark_session, table)
        properties = delta_table.detail().select("properties").collect()[0][0]
        cdf_enabled = properties.get("delta.enableChangeDataFeed")
        assert bool(cdf_enabled) is True


@pytest.mark.skip(reason="depends on volume access on Databricks")
def test_save_to_volume(data_processor_input: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test saving data to Databricks Volume.

    Verifies that data is correctly saved to raw and processed paths in the Volume.

    :param data_processor_input: Input data for processing
    :param config: Configuration object
    :param spark_session: SparkSession for Spark operations
    """
    # Configurar processor
    processor = DataProcessor(pandas_df=data_processor_input, config=config, spark=spark_session)
    processor.preprocess()
    train, test = processor.split_data()

    # Guardar en volumen
    processor.save_to_volume(train, test)

    # Verificar que los archivos existen
    volume_path = f"/Volumes/{config.catalog_name}/{config.schema_name}/{config.volume_name}"
    paths_to_check = [
        f"{volume_path}/raw/train",
        f"{volume_path}/raw/test",
        f"{volume_path}/processed/train",
        f"{volume_path}/processed/test",
    ]

    for path in paths_to_check:
        # Usar API de Hadoop para verificar existencia
        exists = spark_session._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark_session._jsc.hadoopConfiguration()
        ).exists(spark_session._jvm.org.apache.hadoop.fs.Path(path))

        assert exists, f"Path {path} should exist"
