"""Conftest module."""

import platform

from bank_marketing import PROJECT_DIR

# Define direcciones importantes para las pruebas
MLRUNS_DIR = PROJECT_DIR / "tests" / "mlruns"
CATALOG_DIR = PROJECT_DIR / "tests" / "catalog"
CATALOG_DIR.mkdir(parents=True, exist_ok=True)  # noqa

# Hacer que la TRACKING_URI sea compatible tanto para macOS como para Windows
if platform.system() == "Windows":
    TRACKING_URI = f"file:///{MLRUNS_DIR.as_posix()}"
else:
    TRACKING_URI = f"file://{MLRUNS_DIR.as_posix()}"

# Incluir todos los fixtures necesarios
pytest_plugins = [
    "tests.fixtures.data_fixture",
    "tests.fixtures.basic_model_fixture",
    "tests.fixtures.custom_model_fixture",
    "tests.fixtures.pipeline_fixture",
]
