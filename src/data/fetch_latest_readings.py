"""Busca leituras meteorologicas recentes e prepara entrada para o modelo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.weather_features import (  # noqa: E402
    EXPECTED_MODEL_FEATURES,
    add_derived_weather_features,
    fill_missing_numeric_values,
    standardize_weather_columns,
)


DEFAULT_LOCAL_SOURCE = PROJECT_ROOT / "dados_formatados" / "FORTALEZA_DADOS_CLIMATICOS.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_chuva.pkl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "latest_weather_readings_for_model.csv"


class LatestReadingsError(RuntimeError):
    """Erro controlado no consumo ou preparo das leituras recentes."""


def fetch_from_api(api_url: str, timeout: int = 15) -> pd.DataFrame:
    try:
        response = requests.get(api_url, timeout=timeout)
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise LatestReadingsError("Erro de conexao ao consultar a API.") from exc
    except requests.Timeout as exc:
        raise LatestReadingsError("Tempo limite excedido ao consultar a API.") from exc
    except requests.RequestException as exc:
        raise LatestReadingsError("API fora do ar ou retornando erro HTTP.") from exc

    try:
        payload: Any = response.json()
    except ValueError as exc:
        raise LatestReadingsError("Resposta da API nao esta em formato JSON valido.") from exc

    if isinstance(payload, dict):
        records = payload.get("data") or payload.get("results") or payload.get("leituras")
    else:
        records = payload

    if not records:
        raise LatestReadingsError("Resposta vazia: nenhuma leitura encontrada.")

    if not isinstance(records, list):
        raise LatestReadingsError("Formato inesperado: a resposta deve conter uma lista de leituras.")

    return pd.DataFrame(records)


def fetch_from_local_csv(path: Path, limit: int) -> pd.DataFrame:
    if not path.exists():
        raise LatestReadingsError(f"Arquivo local nao encontrado: {path}")

    try:
        df = pd.read_csv(path, sep=";")
    except Exception as exc:
        raise LatestReadingsError("Erro ao ler o arquivo CSV local.") from exc

    if df.empty:
        raise LatestReadingsError("Arquivo local sem registros.")

    return df.tail(limit).copy()


def load_model_features(model_path: Path = DEFAULT_MODEL_PATH) -> list[str]:
    if not model_path.exists():
        return [feature for feature in EXPECTED_MODEL_FEATURES if feature != "lux"]

    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "features" in artifact:
        return list(artifact["features"])
    raise LatestReadingsError("Artefato de modelo em formato inesperado.")


def prepare_latest_readings(df: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    if df.empty:
        raise LatestReadingsError("Nenhuma leitura recebida para preparacao.")

    df = standardize_weather_columns(df)
    df = add_derived_weather_features(df)

    missing_expected = [feature for feature in EXPECTED_MODEL_FEATURES if feature not in df.columns]
    if missing_expected:
        print(
            "Aviso: campos esperados ausentes nas leituras:",
            ", ".join(missing_expected),
        )

    missing_model_features = [feature for feature in model_features if feature not in df.columns]
    if missing_model_features:
        raise LatestReadingsError(
            "Campos exigidos pelo modelo estao ausentes: "
            + ", ".join(missing_model_features)
        )

    df = fill_missing_numeric_values(df, model_features)
    return df[model_features].copy()


def fetch_latest_readings(
    api_url: str | None = None,
    local_csv: Path = DEFAULT_LOCAL_SOURCE,
    limit: int = 24,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Retorna DataFrame pronto para o modelo, via API ou fallback local."""
    raw_df = fetch_from_api(api_url) if api_url else fetch_from_local_csv(local_csv, limit)
    model_features = load_model_features(model_path)
    return prepare_latest_readings(raw_df, model_features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", help="URL da API de leituras meteorologicas.")
    parser.add_argument(
        "--local-csv",
        type=Path,
        default=DEFAULT_LOCAL_SOURCE,
        help="CSV local usado quando API/banco ainda nao estiver disponivel.",
    )
    parser.add_argument("--limit", type=int, default=24, help="Quantidade de linhas recentes.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    readings = fetch_latest_readings(
        api_url=args.api_url,
        local_csv=args.local_csv,
        limit=args.limit,
        model_path=args.model_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    readings.to_csv(args.output, index=False)
    print(f"Leituras preparadas: {len(readings)}")
    print(f"Arquivo salvo em: {args.output}")
