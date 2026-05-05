"""Tratamento do dataset epidemiologico InfoDengue para Fortaleza."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "dados_infodengue" / "infodengue_fortaleza_2017_2024.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "infodengue_fortaleza_tratado.csv"

FORTALEZA_IBGE = "2304400"
LOCATION_COLUMNS = [
    "municipio_nome",
    "municipio",
    "cidade",
    "nome_municipio",
    "nome do municipio",
    "localidade",
]
IBGE_COLUMNS = ["codigo_ibge", "cod_ibge", "municipio_codigo", "ibge", "geocodigo"]

CASE_COLUMNS = [
    "casos_est",
    "casos_est_min",
    "casos_est_max",
    "casos",
    "casprov",
    "casprov_est",
    "casprov_est_min",
    "casprov_est_max",
    "casconf",
    "notif_accum_year",
]


def filter_fortaleza(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra registros de Fortaleza usando nome de municipio ou codigo IBGE."""
    masks = []

    for column in LOCATION_COLUMNS:
        if column in df.columns:
            masks.append(
                df[column].astype(str).str.strip().str.lower().eq("fortaleza")
            )

    for column in IBGE_COLUMNS:
        if column in df.columns:
            masks.append(df[column].astype(str).str.extract(r"(\d+)")[0].eq(FORTALEZA_IBGE))

    if not masks:
        raise ValueError(
            "Nenhuma coluna de localidade reconhecida foi encontrada para filtrar Fortaleza."
        )

    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = combined_mask | mask

    filtered = df.loc[combined_mask].copy()
    if filtered.empty:
        raise ValueError("O filtro de Fortaleza nao retornou registros.")

    return filtered


def prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Converte data_iniSE para datetime e remove datas invalidas."""
    if "data_iniSE" not in df.columns:
        raise ValueError("Coluna obrigatoria data_iniSE nao encontrada.")

    df = df.copy()
    df["data_iniSE"] = pd.to_datetime(df["data_iniSE"], errors="coerce")
    invalid_dates = df["data_iniSE"].isna().sum()
    if invalid_dates:
        print(f"Aviso: {invalid_dates} linhas com data_iniSE invalida foram removidas.")
        df = df.dropna(subset=["data_iniSE"])

    return df.sort_values("data_iniSE")


def prioritize_estimated_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Prioriza casos_est e trata lacunas existentes antes da serie continua."""
    if "casos_est" not in df.columns:
        if "casos" not in df.columns:
            raise ValueError("Colunas casos_est e casos nao encontradas.")
        df["casos_est"] = pd.to_numeric(df["casos"], errors="coerce")
        print("Aviso: casos_est ausente; a coluna casos foi usada como base.")

    df = df.copy()
    for column in CASE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "casos" in df.columns:
        df["casos_est"] = df["casos_est"].fillna(df["casos"])

    # Interpolacao apenas para lacunas em semanas existentes no arquivo bruto.
    df["casos_est"] = df["casos_est"].interpolate(method="linear", limit_direction="both")
    return df


def complete_weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Garante linha do tempo semanal continua e zera casos em semanas ausentes."""
    df = df.copy()
    df["registro_original"] = True

    numeric_case_columns = [column for column in CASE_COLUMNS if column in df.columns]
    agg_rules = {
        column: "sum" if column in numeric_case_columns else "first"
        for column in df.columns
        if column != "data_iniSE"
    }
    weekly = df.groupby("data_iniSE", as_index=False).agg(agg_rules)

    full_index = pd.date_range(
        weekly["data_iniSE"].min(), weekly["data_iniSE"].max(), freq="W-SUN"
    )
    weekly = weekly.set_index("data_iniSE").reindex(full_index)
    weekly.index.name = "data_iniSE"

    inserted_rows = weekly["registro_original"].isna()
    weekly["registro_original"] = weekly["registro_original"].fillna(False)

    for column in numeric_case_columns:
        weekly.loc[inserted_rows, column] = 0

    if "municipio_nome" in weekly.columns:
        weekly["municipio_nome"] = weekly["municipio_nome"].fillna("Fortaleza")

    weekly = weekly.reset_index()
    weekly["ano"] = weekly["data_iniSE"].dt.year
    weekly["mes"] = weekly["data_iniSE"].dt.month
    weekly["semana_epidemiologica"] = weekly["data_iniSE"].dt.isocalendar().week.astype(int)

    return weekly


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona e organiza colunas finais relevantes para analise."""
    df = df.rename(
        columns={
            "data_iniSE": "data_semana_epidemiologica",
            "municipio_nome": "municipio",
            "casos": "casos_notificados",
        }
    )

    preferred_order = [
        "data_semana_epidemiologica",
        "ano",
        "mes",
        "semana_epidemiologica",
        "SE",
        "municipio",
        "casos_est",
        "casos_notificados",
        "casos_est_min",
        "casos_est_max",
        "casprov",
        "casprov_est",
        "casconf",
        "p_inc100k",
        "p_rt1",
        "Rt",
        "nivel",
        "nivel_inc",
        "receptivo",
        "transmissao",
        "pop",
        "tempmin",
        "tempmed",
        "tempmax",
        "umidmin",
        "umidmed",
        "umidmax",
        "registro_original",
    ]
    existing = [column for column in preferred_order if column in df.columns]
    remaining = [column for column in df.columns if column not in existing]
    return df[existing + remaining]


def process_infodengue(raw_path: Path = RAW_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Executa o pipeline completo e salva o CSV tratado."""
    df = pd.read_csv(raw_path)
    df = filter_fortaleza(df)
    df = prepare_dates(df)
    df = prioritize_estimated_cases(df)
    df = complete_weekly_series(df)
    df = standardize_columns(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    processed = process_infodengue()
    print(f"Arquivo salvo em: {OUTPUT_PATH}")
    print(f"Linhas: {len(processed)}")
    print(
        "Periodo:",
        processed["data_semana_epidemiologica"].min(),
        "a",
        processed["data_semana_epidemiologica"].max(),
    )
