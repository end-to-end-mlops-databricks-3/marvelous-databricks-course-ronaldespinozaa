"""Dataloader fixture."""

import numpy as np
import pandas as pd
import pytest
from loguru import logger
from pyspark.sql import SparkSession

from bank_marketing import PROJECT_DIR
from bank_marketing.config import ProjectConfig, Tags
from tests.unit_tests.spark_config import spark_config


@pytest.fixture(scope="session")
def spark_session() -> SparkSession:
    """Create and return a SparkSession for testing.

    This fixture creates a SparkSession with the specified configuration and returns it for use in tests.
    """
    # Configuración para SparkSession
    spark = (
        SparkSession.builder.master(spark_config.master)
        .appName(spark_config.app_name)
        .config("spark.executor.cores", spark_config.spark_executor_cores)
        .config("spark.executor.instances", spark_config.spark_executor_instances)
        .config("spark.sql.shuffle.partitions", spark_config.spark_sql_shuffle_partitions)
        .config("spark.driver.bindAddress", spark_config.spark_driver_bindAddress)
        .getOrCreate()
    )

    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    """Load and return the project configuration.

    This fixture reads the project configuration from a YAML file and returns a ProjectConfig object.

    :return: The loaded project configuration
    """
    config_file_path = (PROJECT_DIR / "project_config.yml").resolve()
    logger.info(f"Current config file path: {config_file_path.as_posix()}")

    # Cargar configuración con entorno 'dev'
    config = ProjectConfig.from_yaml(config_file_path.as_posix(), env="dev")
    return config


@pytest.fixture(scope="function")
def sample_bank_data(config: ProjectConfig, spark_session: SparkSession) -> pd.DataFrame:
    """Create a sample DataFrame for bank marketing data.

    This fixture reads a CSV file if it exists or creates synthetic data for testing.

    :return: A Pandas DataFrame containing bank marketing data.
    """
    file_path = PROJECT_DIR / "tests" / "test_data" / "sample.csv"

    try:
        if file_path.exists():
            logger.info(f"Loading sample data from {file_path}")
            sample = pd.read_csv(file_path.as_posix())

            # Verificar si el archivo parece contener datos de marketing bancario
            expected_bank_columns = ["age", "job", "marital", "balance"]
            bank_columns_present = all(col in sample.columns for col in expected_bank_columns)

            if not bank_columns_present:
                logger.warning("Sample file doesn't appear to contain bank marketing data, creating synthetic data")
                sample = _create_synthetic_bank_data()
        else:
            # Crear datos sintéticos si el archivo no existe
            logger.info("Sample file not found, creating synthetic bank marketing data")
            sample = _create_synthetic_bank_data()

            # Guardar datos para uso futuro
            test_data_dir = PROJECT_DIR / "tests" / "test_data"
            if not test_data_dir.exists():
                test_data_dir.mkdir(parents=True, exist_ok=True)
            sample.to_csv(file_path.as_posix(), index=False)
            logger.info(f"Created and saved synthetic data to {file_path}")

    except Exception as e:
        logger.error(f"Error loading/creating sample data: {e}")
        # Crear un DataFrame mínimo en caso de error
        sample = _create_synthetic_bank_data(num_samples=2)

    # Verificar que todas las columnas configuradas estén presentes
    expected_columns = config.num_features + config.cat_features + [config.target]
    missing_columns = [col for col in expected_columns if col not in sample.columns]

    if missing_columns:
        logger.warning(f"Missing columns in sample data: {missing_columns}")
        # Añadir columnas faltantes con valores predeterminados
        for col in missing_columns:
            if col in config.num_features:
                sample[col] = 0
            elif col in config.cat_features:
                sample[col] = "unknown"
            elif col == config.target:
                sample[col] = "no"

    return sample


def _create_synthetic_bank_data(num_samples: int = 10) -> pd.DataFrame:
    """Crear datos sintéticos para pruebas de Bank Marketing.

    Esta función auxiliar crea un DataFrame con datos sintéticos para pruebas.

    Args:
        num_samples: Número de muestras a generar (default: 10)

    Returns:
        DataFrame con datos sintéticos

    """
    # Crear datos sintéticos para pruebas
    np.random.seed(42)  # Para reproducibilidad

    data = {
        "age": np.random.randint(18, 70, num_samples),
        "job": np.random.choice(["admin", "blue-collar", "management", "technician", "retired"], num_samples),
        "marital": np.random.choice(["married", "single", "divorced"], num_samples),
        "education": np.random.choice(["primary", "secondary", "tertiary"], num_samples),
        "default": np.random.choice(["yes", "no"], num_samples),
        "balance": np.random.normal(1000, 500, num_samples).astype(int),
        "housing": np.random.choice(["yes", "no"], num_samples),
        "loan": np.random.choice(["yes", "no"], num_samples),
        "contact": np.random.choice(["cellular", "telephone"], num_samples),
        "day": np.random.randint(1, 31, num_samples),
        "month": np.random.choice(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct"], num_samples),
        "duration": np.random.randint(100, 1000, num_samples),
        "campaign": np.random.randint(1, 10, num_samples),
        "pdays": np.random.choice([999] + list(range(1, 30)), num_samples),
        "previous": np.random.randint(0, 5, num_samples),
        "poutcome": np.random.choice(["unknown", "success", "failure"], num_samples),
        "Target": np.random.choice(["yes", "no"], num_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture(scope="function")
def data_processor_input(sample_bank_data: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Prepare input data specifically for testing DataProcessor.

    This fixture ensures that the input data is in the correct format for the DataProcessor class,
    with proper column types and handling of missing values.

    :param sample_bank_data: Raw sample data
    :param config: Project configuration with feature definitions
    :return: Prepared DataFrame ready for DataProcessor
    """
    # Hacer una copia para no modificar el original
    df = sample_bank_data.copy()

    # Asegurar que las columnas numéricas sean de tipo numérico
    for col in config.num_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Asegurar que las columnas categóricas sean de tipo object o category
    for col in config.cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Asegurar que el target sea de tipo string
    if config.target in df.columns:
        df[config.target] = df[config.target].astype(str)

    # Simular valores "unknown" para probar el manejo de valores faltantes
    for col in config.cat_features[:2]:  # Solo algunas columnas para no exagerar
        if col in df.columns and len(df) > 3:
            df.loc[2, col] = "unknown"

    return df


@pytest.fixture(scope="session")
def tags() -> Tags:
    """Create and return a Tags instance for the test session.

    This fixture provides a Tags object with predefined values for git_sha, branch, and job_run_id.
    """
    return Tags(git_sha="wxyz", branch="test", job_run_id="9")
