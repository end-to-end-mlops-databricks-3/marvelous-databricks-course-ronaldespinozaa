# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Experiment Tracking - Demostración
# MAGIC
# MAGIC Este notebook demuestra el uso de MLflow para seguimiento de experimentos con datos sintéticos.

# COMMAND ----------

import json
import mlflow
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from marvelous.common import is_databricks
from dotenv import load_dotenv


# COMMAND ----------
mlflow.get_tracking_uri()

# COMMAND ----------
if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")

mlflow.get_tracking_uri()
# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración de MLflow

# COMMAND ----------

# Obtener la URI de tracking actual
print(f"URI de tracking actual: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Creación y configuración de experimentos

# COMMAND ----------

# Crear un experimento de demostración
experiment_name = "/Shared/bank-marketing-tracking-demo"
experiment = mlflow.set_experiment(experiment_name)

# Establecer tags para el experimento
mlflow.set_experiment_tags(
    {"project": "bank_marketing_demo", "domain": "financial_services", "purpose": "experiment_tracking_demo"}
)

print(f"Experimento configurado: {experiment.name}")
print(f"ID del experimento: {experiment.experiment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run básico con parámetros y métricas

# COMMAND ----------

# Ejemplo de run básico
with mlflow.start_run(
    run_name="demo-basic-tracking",
    tags={"git_sha": "demo123", "branch": "experiment-tracking", "model_type": "demo"},
    description="Demostración básica de tracking con MLflow",
) as run:
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

    # Registrar parámetros (configuración del modelo)
    mlflow.log_params({"learning_rate": 0.05, "n_estimators": 100, "max_depth": 5, "model_type": "lightgbm"})

    # Simular métricas de entrenamiento
    metrics = {"accuracy": 0.85, "precision": 0.82, "recall": 0.78, "f1_score": 0.80, "auc": 0.88}

    # Registrar métricas
    mlflow.log_metrics(metrics)

    print("✅ Parámetros y métricas registrados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Logging de métricas dinámicas (épocas de entrenamiento)

# COMMAND ----------

# Simular entrenamiento con múltiples épocas
with mlflow.start_run(run_name="demo-dynamic-metrics") as run:
    # Simular mejora de métricas durante el entrenamiento
    for epoch in range(10):
        # Simular métricas que mejoran con el tiempo
        train_loss = 1.0 - (epoch * 0.08) + np.random.normal(0, 0.02)
        val_loss = 1.1 - (epoch * 0.07) + np.random.normal(0, 0.03)
        accuracy = 0.5 + (epoch * 0.04) + np.random.normal(0, 0.01)

        # Registrar métricas con step para crear gráficos temporales
        mlflow.log_metrics(
            {
                "train_loss": max(0.1, train_loss),
                "val_loss": max(0.15, val_loss),
                "accuracy": min(0.95, max(0.5, accuracy)),
            },
            step=epoch,
        )

    print("✅ Métricas dinámicas registradas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Logging de artefactos y visualizaciones

# COMMAND ----------

with mlflow.start_run(run_name="demo-artifacts") as run:
    # 1. Registrar texto simple
    mlflow.log_text("Experimento de demostración para Bank Marketing", "experiment_notes.txt")

    # 2. Registrar diccionario como JSON
    model_config = {
        "architecture": "gradient_boosting",
        "features": ["age", "job", "balance", "housing"],
        "target": "subscription",
        "preprocessing": {"scaling": "standard", "encoding": "onehot"},
    }
    mlflow.log_dict(model_config, "model_config.json")

    # 3. Crear y registrar un gráfico
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico de importancia de características (simulado)
    features = ["age", "balance", "duration", "campaign", "housing"]
    importance = [0.25, 0.30, 0.20, 0.15, 0.10]

    ax1.barh(features, importance)
    ax1.set_xlabel("Importancia")
    ax1.set_title("Importancia de Características")

    # Gráfico de distribución de target (simulado)
    labels = ["No suscribe", "Sí suscribe"]
    sizes = [70, 30]

    ax2.pie(sizes, labels=labels, autopct="%1.1f%%")
    ax2.set_title("Distribución del Target")

    plt.tight_layout()
    mlflow.log_figure(fig, "model_analysis.png")
    plt.close()

    # 4. Crear y registrar múltiples imágenes
    for i in range(3):
        # Simular gráficos de convergencia
        epochs = range(20)
        loss = [1.0 * (0.9**epoch) + np.random.normal(0, 0.02) for epoch in epochs]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss, label=f"Experiment {i + 1}")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.title(f"Convergencia del Modelo - Experimento {i + 1}")
        plt.legend()
        plt.grid(True)

        # Guardar como archivo temporal
        temp_path = f"convergence_exp_{i}.png"
        plt.savefig(temp_path)
        plt.close()

        # Registrar en MLflow
        mlflow.log_artifact(temp_path, "convergence_plots")

        # Limpiar archivo temporal
        os.remove(temp_path)

    print("✅ Artefactos y visualizaciones registrados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Runs anidados para ajuste de hiperparámetros

# COMMAND ----------

# Demostración de runs anidados
with mlflow.start_run(run_name="hyperparameter_tuning_demo") as parent_run:
    print(f"Run principal iniciado: {parent_run.info.run_id}")

    # Parámetros del experimento
    mlflow.log_params(
        {"dataset": "bank_marketing_demo", "target": "subscription", "cv_folds": 5, "search_type": "grid_search"}
    )

    best_score = 0
    best_params = {}

    # Grid search simulado
    for learning_rate in [0.01, 0.05, 0.1]:
        for max_depth in [3, 5, 7]:
            for n_estimators in [50, 100]:
                # Crear run hijo
                with mlflow.start_run(
                    run_name=f"lr_{learning_rate}_depth_{max_depth}_est_{n_estimators}", nested=True
                ) as child_run:
                    # Registrar parámetros específicos
                    params = {"learning_rate": learning_rate, "max_depth": max_depth, "n_estimators": n_estimators}
                    mlflow.log_params(params)

                    # Simular entrenamiento y evaluación
                    base_score = 0.75
                    lr_bonus = learning_rate * 2
                    depth_bonus = max_depth / 20
                    est_bonus = n_estimators / 1000

                    # Añadir algo de ruido
                    noise = np.random.normal(0, 0.02)
                    score = min(0.95, base_score + lr_bonus + depth_bonus + est_bonus + noise)

                    # Simular tiempo de entrenamiento
                    training_time = 10 + (n_estimators / 10) + np.random.normal(0, 2)

                    # Registrar métricas
                    mlflow.log_metrics({"cv_score": score, "training_time": max(1, training_time)})

                    # Tracking del mejor modelo
                    if score > best_score:
                        best_score = score
                        best_params = params

                    print(
                        f"    Configuración: LR={learning_rate}, Depth={max_depth}, Est={n_estimators}, Score={score:.4f}"
                    )

    # Registrar el mejor resultado en el run padre
    mlflow.log_metrics({"best_cv_score": best_score})
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

    print(f"\n✅ Mejor configuración: {best_params} con score: {best_score:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Búsqueda y análisis de experimentos

# COMMAND ----------

# Buscar runs con filtros
from time import time

time_hour_ago = int(time() - 3600) * 1000

# IMPORTANT: Replace 'your_experiment_name' with the actual name of your MLflow experiment

try:
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["start_time DESC"],
        filter_string="status='FINISHED' AND "
        f"start_time>{time_hour_ago} AND "
        "tags.mlflow.runName LIKE '%demo%'",  # Changed to use tags.mlflow.runName
    )

    print(f"Runs encontrados en la última hora: {len(runs)}")

    if not runs.empty:
        # Define the relevant columns you want to display
        relevant_columns = [
            "tags.mlflow.runName",
            "start_time",
            "metrics.best_cv_score",
            "params.model_type",
            "run_id",  # Adding run_id as it's always good for identification
        ]

        # Filter the DataFrame to only include the relevant columns, if they exist
        # Also, ensure 'start_time' is converted to a readable format if present
        display_runs = pd.DataFrame()
        if not runs.empty:
            # Select columns that exist in the DataFrame
            existing_relevant_columns = [col for col in relevant_columns if col in runs.columns]
            display_runs = runs[existing_relevant_columns].copy()

            # Convert 'start_time' to datetime if it exists
            if "start_time" in display_runs.columns:
                display_runs["start_time"] = pd.to_datetime(display_runs["start_time"], unit="ms")

        print("\nResumen de runs con columnas relevantes:")
        print(display_runs.head())

        # Show metrics of the best runs, using 'metrics.best_cv_score'
        if "metrics.best_cv_score" in runs.columns:
            best_runs = runs.nlargest(3, "metrics.best_cv_score")
            print("\nTop 3 runs por 'Best CV Score':")
            for idx, run in best_runs.iterrows():
                run_name = run["tags.mlflow.runName"] if "tags.mlflow.runName" in run else "N/A"
                print(f"- {run_name}: Best CV Score = {run['metrics.best_cv_score']:.4f}")
        else:
            print("\nLa métrica 'metrics.best_cv_score' no está disponible en los runs.")

    else:
        print("No se encontraron runs que coincidan con los criterios de filtro.")

except Exception as e:
    print(f"An error occurred: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Recuperación de artefactos

# COMMAND ----------

# Cargar artefactos del último run
if not runs.empty:
    latest_run = runs.iloc[0]
    artifact_uri = latest_run["artifact_uri"]

    try:
        # Intentar cargar configuración del modelo
        config_data = mlflow.artifacts.load_dict(f"{artifact_uri}/model_config.json")
        print("📊 Configuración del modelo cargada:")
        for key, value in config_data.items():
            print(f"   - {key}: {value}")
    except Exception as e:
        print(f"⚠️ No se pudo cargar la configuración: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Resumen y mejores prácticas

# COMMAND ----------

print("""
✅ MLflow Experiment Tracking Demo Completado

Funcionalidades demostradas:
1. ✅ Configuración de experimentos y tracking
2. ✅ Logging de parámetros, métricas y artefactos
3. ✅ Métricas dinámicas (por época/iteración)
4. ✅ Visualizaciones y gráficos
5. ✅ Runs anidados para hiperparámetros
6. ✅ Búsqueda y filtrado de experimentos
7. ✅ Recuperación de artefactos

💡 Mejores prácticas aprendidas:
- Usar nombres descriptivos para runs
- Incluir tags para organización
- Registrar tanto parámetros como métricas
- Usar runs anidados para búsquedas sistemáticas
- Guardar visualizaciones como artefactos
- Mantener experimentos organizados por proyecto

🚀 Próximos pasos:
- Integrar con pipeline de datos real
- Automatizar el tracking en scripts de producción
- Implementar comparación automática de modelos
- Configurar alertas para modelos con bajo rendimiento
""")
