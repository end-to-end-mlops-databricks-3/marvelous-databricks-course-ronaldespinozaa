"""Fixtures que usan los archivos de datos reales del proyecto."""

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from bank_marketing import PROJECT_DIR


@pytest.fixture
def sample_bank_data_file() -> pd.DataFrame:
    """Carga datos del archivo sample_bank_data.csv."""
    file_path = PROJECT_DIR / "tests" / "test_data" / "sample_bank_data.csv"

    if file_path.exists():
        return pd.read_csv(file_path)
    else:
        # Fallback con datos mínimos si no existe el archivo
        return pd.DataFrame(
            {
                "age": [40, 47, 25],
                "job": ["blue-collar", "services", "student"],
                "Target": ["no", "no", "no"],
            }
        )


@pytest.fixture
def train_set_data() -> pd.DataFrame:
    """Carga datos del archivo train_set.csv del catálogo."""
    file_path = PROJECT_DIR / "tests" / "catalog" / "train_set.csv"

    if file_path.exists():
        return pd.read_csv(file_path)
    else:
        # Fallback si no existe
        return pd.DataFrame({"age": [30, 40, 50], "balance": [1000, -200, 500], "Target": [1, 0, 1]})


@pytest.fixture
def test_set_data() -> pd.DataFrame:
    """Carga datos del archivo test_set.csv del catálogo."""
    file_path = PROJECT_DIR / "tests" / "catalog" / "test_set.csv"

    if file_path.exists():
        return pd.read_csv(file_path)
    else:
        # Fallback si no existe
        return pd.DataFrame({"age": [35, 45], "balance": [800, -100], "Target": [0, 1]})


@pytest.fixture
def real_bank_columns() -> list[str]:
    """Columnas reales del dataset bancario."""
    return [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "Target",
    ]


@pytest.fixture
def real_categorical_values() -> dict[str, list[str]]:
    """Valores categóricos reales del dataset."""
    return {
        "job": [
            "blue-collar",
            "services",
            "student",
            "management",
            "admin.",
            "technician",
            "retired",
            "unemployed",
            "self-employed",
            "entrepreneur",
            "housemaid",
            "unknown",
        ],
        "marital": ["married", "single", "divorced"],
        "education": ["primary", "secondary", "tertiary", "unknown"],
        "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        "poutcome": ["success", "failure", "other", "unknown"],
    }


@pytest.fixture
def config_for_real_data() -> Mock:
    """Configuración que coincide con datos reales del banco."""
    config = Mock()
    config.catalog_name = "test_catalog"
    config.schema_name = "test_schema"
    config.target = "Target"
    config.num_features = ["age", "balance", "duration", "campaign", "pdays", "previous", "day"]
    config.cat_features = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
    config.parameters = {
        "random_state": 42,
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "verbose": -1,
    }
    return config


@pytest.fixture
def small_real_dataset() -> pd.DataFrame:
    """Dataset pequeño con estructura real para tests rápidos."""
    return pd.DataFrame(
        {
            "age": [40, 47, 25, 42, 56, 28, 24, 37, 30, 38],
            "job": [
                "blue-collar",
                "services",
                "student",
                "management",
                "management",
                "blue-collar",
                "management",
                "admin.",
                "blue-collar",
                "technician",
            ],
            "marital": [
                "married",
                "single",
                "single",
                "married",
                "married",
                "married",
                "single",
                "single",
                "single",
                "single",
            ],
            "education": [
                "secondary",
                "secondary",
                "tertiary",
                "tertiary",
                "tertiary",
                "secondary",
                "tertiary",
                "secondary",
                "secondary",
                "secondary",
            ],
            "default": ["no"] * 10,
            "balance": [580, 3644, 538, 1773, 217, 1134, 1085, 127, 3, 258],
            "housing": ["yes", "no", "yes", "no", "no", "no", "no", "no", "yes", "no"],
            "loan": ["no", "no", "no", "no", "yes", "no", "yes", "no", "no", "yes"],
            "contact": [
                "unknown",
                "unknown",
                "cellular",
                "cellular",
                "cellular",
                "cellular",
                "cellular",
                "cellular",
                "cellular",
                "unknown",
            ],
            "day": [16, 9, 20, 9, 21, 9, 7, 23, 25, 20],
            "month": ["may", "jun", "apr", "apr", "jul", "feb", "may", "mar", "jul", "jun"],
            "duration": [192, 83, 226, 311, 121, 130, 95, 83, 51, 587],
            "campaign": [1, 2, 1, 1, 2, 3, 6, 4, 1, 2],
            "pdays": [-1, -1, -1, 336, -1, -1, -1, -1, -1, -1],
            "previous": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "poutcome": ["unknown"] * 8 + ["unknown", "unknown"],
            "Target": ["no", "no", "no", "no", "no", "no", "no", "no", "no", "no"],
        }
    )


@pytest.fixture
def balanced_real_dataset() -> pd.DataFrame:
    """Dataset pequeño con casos yes/no balanceados."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 30, 40],
            "job": ["student", "technician", "management", "retired", "admin.", "services"],
            "marital": ["single", "married", "married", "married", "single", "divorced"],
            "education": ["tertiary", "secondary", "tertiary", "secondary", "tertiary", "secondary"],
            "balance": [500, -200, 1500, 800, 0, 300],
            "duration": [200, 150, 300, 180, 120, 250],
            "campaign": [1, 2, 1, 3, 2, 1],
            "previous": [0, 1, 0, 2, 0, 1],
            "month": ["apr", "may", "jun", "jul", "aug", "sep"],
            "contact": ["cellular", "telephone", "cellular", "cellular", "telephone", "cellular"],
            "Target": ["yes", "no", "yes", "no", "yes", "no"],  # Balanceado
        }
    )


def get_test_data_path() -> Path:
    """Retorna el path al directorio de datos de test."""
    return PROJECT_DIR / "tests" / "test_data"


def get_catalog_path() -> Path:
    """Retorna el path al directorio del catálogo."""
    return PROJECT_DIR / "tests" / "catalog"
