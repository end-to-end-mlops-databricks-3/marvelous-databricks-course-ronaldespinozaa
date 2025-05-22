"""Custom model fixture."""

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from loguru import logger
from pyspark.sql import SparkSession

from bank_marketing import PROJECT_DIR
from bank_marketing.config import ProjectConfig, Tags
from bank_marketing.models.custom_model import CustomModel

# Define directorios importantes para las pruebas
# (Normalmente estarían en conftest.py, pero los incluimos aquí para mayor simplicidad)
MLRUNS_DIR = PROJECT_DIR / "tests" / "mlruns"
CATALOG_DIR = PROJECT_DIR / "tests" / "catalog"

# Crear directorios necesarios
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
CATALOG_DIR.mkdir(parents=True, exist_ok=True)

whl_file_name: str | None = None  # Variable global para almacenar el nombre del archivo .whl


@pytest.fixture(scope="session", autouse=True)
def create_mlruns_directory() -> None:
    """Create or recreate the MLFlow tracking directory.

    This fixture ensures that the MLFlow tracking directory is clean and ready for use
    before each test session.
    """
    if MLRUNS_DIR.exists():
        shutil.rmtree(MLRUNS_DIR)
        MLRUNS_DIR.mkdir()
        logger.info(f"Created {MLRUNS_DIR} directory for MLFlow tracking")
    else:
        logger.info(f"MLFlow tracking directory {MLRUNS_DIR} does not exist")


@pytest.fixture(scope="session", autouse=True)
def build_whl_file() -> None:
    """Session-scoped fixture to build a .whl file for the project.

    This fixture ensures that the project's distribution directory is cleaned,
    the build process is executed, and the resulting .whl file is identified.
    The fixture runs automatically once per test session.

    :raises RuntimeError: If an unexpected error occurs during the build process.
    :raises FileNotFoundError: If the dist directory or .whl file is not found.
    """
    global whl_file_name
    dist_directory_path = PROJECT_DIR / "dist"
    original_directory = Path.cwd()  # Save the current working directory

    try:
        # Clean up the dist directory if it exists
        if dist_directory_path.exists():
            shutil.rmtree(dist_directory_path)

        # Change to project directory and execute 'uv build'
        subprocess.run(["uv", "build"], check=True, text=True, capture_output=True)

        # Ensure the dist directory exists after the build
        if not dist_directory_path.exists():
            raise FileNotFoundError(f"Dist directory does not exist: {dist_directory_path}")

        # Get list of files in the dist directory
        files = [entry.name for entry in dist_directory_path.iterdir() if entry.is_file()]

        # Find the first .whl file
        whl_file = next((file for file in files if file.endswith(".whl")), None)
        if not whl_file:
            raise FileNotFoundError("No .whl file found in the dist directory.")

        # Set the global variable with the .whl file name
        whl_file_name = whl_file

    except Exception as err:
        raise RuntimeError(f"Unexpected error occurred: {err}") from err

    finally:
        # Restore the original working directory
        os.chdir(original_directory)


@pytest.fixture(scope="function")
def mock_custom_model(config: ProjectConfig, tags: Tags, spark_session: SparkSession) -> CustomModel:
    """Fixture that provides a CustomModel instance with mocked Spark interactions.

     Initializes the model with test data and mocks Spark DataFrame conversions to pandas.

    :param config: Project configuration parameters
    :param tags: Tagging metadata for model tracking
    :param spark_session: Spark session instance for testing environment
    :return: Configured CustomModel instance with mocked Spark interactions
    """
    instance = CustomModel(
        config=config,
        tags=tags,
        spark=spark_session,
        code_paths=[f"{PROJECT_DIR.as_posix()}/dist/{whl_file_name}"] if whl_file_name else [],
    )

    # Intentar cargar datos de prueba
    try:
        train_data = pd.read_csv((CATALOG_DIR / "train_processed.csv").as_posix())
        # Important Note: Replace NaN with None in Pandas
        train_data = train_data.where(train_data.notna(), None)  # noqa

        test_data = pd.read_csv((CATALOG_DIR / "test_processed.csv").as_posix())
        test_data = test_data.where(test_data.notna(), None)  # noqa
    except FileNotFoundError:
        # Si los archivos no existen, crear datos sintéticos
        logger.warning("Test data files not found, creating synthetic data")

        # Reutilizar la función para crear datos sintéticos del otro fixture
        # (Normalmente importaríamos esta función, pero como estamos limitados a estos archivos,
        # vamos a duplicar la implementación)

        # Crear datos sintéticos
        np.random.seed(42)  # Para reproducibilidad
        num_samples = 10

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
            "month": np.random.choice(
                ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct"], num_samples
            ),
            "duration": np.random.randint(100, 1000, num_samples),
            "campaign": np.random.randint(1, 10, num_samples),
            "pdays": np.random.choice([999] + list(range(1, 30)), num_samples),
            "previous": np.random.randint(0, 5, num_samples),
            "poutcome": np.random.choice(["unknown", "success", "failure"], num_samples),
            "Target": np.random.choice([0, 1], num_samples),
        }

        synthetic_data = pd.DataFrame(data)

        # Dividir en train y test
        train_data = synthetic_data.iloc[:8]
        test_data = synthetic_data.iloc[8:]

        # Guardar para uso futuro
        os.makedirs(CATALOG_DIR, exist_ok=True)
        train_data.to_csv((CATALOG_DIR / "train_processed.csv").as_posix(), index=False)
        test_data.to_csv((CATALOG_DIR / "test_processed.csv").as_posix(), index=False)

    ## Mock Spark interactions
    # Mock Spark DataFrame with toPandas() method
    mock_spark_df_train = MagicMock()
    mock_spark_df_train.toPandas.return_value = train_data
    mock_spark_df_test = MagicMock()
    mock_spark_df_test.toPandas.return_value = test_data

    # Mock spark.table method
    mock_spark = MagicMock()
    mock_spark.table.side_effect = [mock_spark_df_train, mock_spark_df_test]

    # Reemplazar la sesión Spark en la instancia
    instance.spark = mock_spark

    # Pre-configurar algunos atributos para simplificar las pruebas
    instance.train_set_spark = mock_spark_df_train
    instance.train_set = train_data
    instance.test_set = test_data

    # Mocks adicionales para consultas SQL
    mock_sql_result = MagicMock()
    mock_spark.sql.return_value = mock_sql_result

    # Mocks para createDataFrame
    mock_df = MagicMock()
    mock_spark.createDataFrame.return_value = mock_df

    # Configurar un resultado para DESCRIBE HISTORY
    mock_history_row = MagicMock()
    mock_history_row.__getitem__.return_value = "1"  # version
    mock_history_collect = MagicMock()
    mock_history_collect.collect.return_value = [mock_history_row]
    mock_spark.sql.return_value = mock_history_collect

    return instance


@pytest.fixture(scope="session", autouse=True)
def prepare_test_files() -> None:
    """Prepare test files for bank marketing by converting house price data or creating new files.

    This fixture checks if the existing files contain bank marketing data.
    If not, it creates new files with appropriate bank marketing data structure.
    """
    # Comprobar si los archivos originales existen
    train_source = CATALOG_DIR / "train_set.csv"
    test_source = CATALOG_DIR / "test_set.csv"

    # Archivos destino para nuestras pruebas
    train_dest = CATALOG_DIR / "train_processed.csv"
    test_dest = CATALOG_DIR / "test_processed.csv"

    try:
        # Verificar si los archivos existentes contienen datos de bank marketing
        if train_source.exists():
            df = pd.read_csv(train_source)
            # Verificar si tenemos columnas típicas de bank marketing
            has_bank_columns = all(col in df.columns for col in ["age", "job", "marital", "balance"])

            if not has_bank_columns:
                logger.info("Existing files don't contain bank marketing data, creating new files")
                _create_bank_marketing_files(train_dest, test_dest)
            else:
                # Si ya son datos de bank marketing, solo copiar si es necesario
                if not train_dest.exists():
                    shutil.copy2(train_source, train_dest)
                    logger.info(f"Copied {train_source} to {train_dest}")

                if not test_dest.exists() and test_source.exists():
                    shutil.copy2(test_source, test_dest)
                    logger.info(f"Copied {test_source} to {test_dest}")
        else:
            # Si no hay archivos originales, crear nuevos
            logger.info("Source files not found, creating bank marketing files")
            _create_bank_marketing_files(train_dest, test_dest)

    except Exception as e:
        logger.error(f"Error preparing test files: {e}")
        logger.warning("Tests may fail due to missing or incorrect test data")


def _create_bank_marketing_files(train_path: Path, test_path: Path) -> None:
    """Create bank marketing data files for testing.

    Args:
        train_path: Path where to save the training data
        test_path: Path where to save the test data

    """
    import numpy as np

    # Crear datos sintéticos
    np.random.seed(42)  # Para reproducibilidad
    num_samples = 100
    test_samples = 20

    # Generar datos para bank marketing
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

    # Crear DataFrames
    all_data = pd.DataFrame(data)

    # Dividir en train y test
    train_data = all_data.iloc[:-test_samples]
    test_data = all_data.iloc[-test_samples:]

    # Guardar a archivos
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    logger.info(f"Created bank marketing data: {train_path} and {test_path}")
