"""Funcoes compartilhadas para preparar variaveis meteorologicas."""

from __future__ import annotations

import numpy as np
import pandas as pd


EXPECTED_MODEL_FEATURES = ["temp", "umid", "press", "lux", "ponto_orvalho", "delta_pressao"]

COLUMN_ALIASES = {
    "chuva": "chuva",
    "precipitacao": "chuva",
    "precipitação": "chuva",
    "rain": "chuva",
    "pressao": "press",
    "pressão": "press",
    "press": "press",
    "temp": "temp",
    "temperatura": "temp",
    "umid": "umid",
    "umidade": "umid",
    "humidity": "umid",
    "lux": "lux",
    "luminosidade": "lux",
    "ponto_orvalho": "ponto_orvalho",
    "dew_point": "ponto_orvalho",
    "delta_p": "delta_pressao",
    "delta_pressao": "delta_pressao",
    "delta_pressão": "delta_pressao",
    "dt_local": "dt_local",
    "data_local": "data_local",
    "hora_local": "hora_local",
    "vento": "vento",
    "rad": "rad",
}


def normalize_column_name(column: str) -> str:
    return COLUMN_ALIASES.get(column.strip().lower(), column.strip().lower())


def standardize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={column: normalize_column_name(column) for column in df.columns})


def calculate_dew_point(temp_celsius: pd.Series, humidity_percent: pd.Series) -> pd.Series:
    """Calcula ponto de orvalho aproximado pela formula de Magnus."""
    a = 17.27
    b = 237.7
    humidity = humidity_percent.clip(lower=1, upper=100)
    alpha = ((a * temp_celsius) / (b + temp_celsius)) + np.log(humidity / 100)
    return (b * alpha) / (a - alpha)


def add_derived_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["temp", "umid", "press", "lux", "delta_pressao", "chuva"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "ponto_orvalho" not in df.columns and {"temp", "umid"}.issubset(df.columns):
        df["ponto_orvalho"] = calculate_dew_point(df["temp"], df["umid"])
    elif "ponto_orvalho" in df.columns:
        df["ponto_orvalho"] = pd.to_numeric(df["ponto_orvalho"], errors="coerce")

    return df


def fill_missing_numeric_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df[column] = df[column].fillna(df[column].median())
    return df
