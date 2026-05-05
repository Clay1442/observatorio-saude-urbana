"""Treina um classificador inicial de categorias de chuva para Fortaleza."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.weather_features import (
    EXPECTED_MODEL_FEATURES,
    add_derived_weather_features,
    fill_missing_numeric_values,
    standardize_weather_columns,
)

CLIMATE_PATH = PROJECT_ROOT / "dados_formatados" / "FORTALEZA_DADOS_CLIMATICOS.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_chuva.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "random_forest_chuva_metrics.json"


def load_climate_data(path: Path = CLIMATE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    return standardize_weather_columns(df)


def create_rain_category(precipitation_mm: pd.Series) -> pd.Series:
    """0: sem chuva; 1: chuva fraca; 2: chuva moderada/forte."""
    return pd.cut(
        precipitation_mm,
        bins=[-np.inf, 0, 5, np.inf],
        labels=[0, 1, 2],
        include_lowest=True,
    ).astype(int)


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if "chuva" not in df.columns:
        raise ValueError("Coluna de precipitacao CHUVA/chuva nao encontrada.")

    df = add_derived_weather_features(df)

    available_features = [column for column in EXPECTED_MODEL_FEATURES if column in df.columns]
    missing_features = [column for column in EXPECTED_MODEL_FEATURES if column not in df.columns]

    if not available_features:
        raise ValueError("Nenhuma variavel meteorologica esperada foi encontrada.")

    model_df = df[available_features + ["chuva"]].copy()
    model_df = model_df.dropna(subset=["chuva"])

    model_df = fill_missing_numeric_values(model_df, available_features)

    y = create_rain_category(model_df["chuva"])
    x = model_df[available_features]
    return x, y, missing_features


def train_model(
    climate_path: Path = CLIMATE_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict:
    df = load_climate_data(climate_path)
    x, y, missing_features = prepare_training_data(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "feature_importance": dict(
            sorted(
                ((column, float(importance)) for column, importance in zip(x.columns, model.feature_importances_, strict=True)),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
        "features_used": list(x.columns),
        "missing_expected_features": missing_features,
        "target_classes": {
            "0": "Sem chuva - 0 mm",
            "1": "Chuva fraca - 0.1 mm ate 5 mm",
            "2": "Chuva moderada/forte - acima de 5 mm",
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(x.columns)}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = train_model()
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Metricas salvas em: {METRICS_PATH}")
    print(f"Acuracia: {result['accuracy']:.4f}")
    if result["missing_expected_features"]:
        print("Variaveis esperadas ausentes:", ", ".join(result["missing_expected_features"]))
