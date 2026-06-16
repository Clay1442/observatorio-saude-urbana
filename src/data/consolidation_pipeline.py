from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
# Fontes oficiais usadas na consolidação mensal.
CLIMATE_PATH = BASE_DIR / "dados_formatados" / "FORTALEZA_DADOS_CLIMATICOS.csv"
DENGUE_PATH = BASE_DIR / "dados_formatados" / "INFODENGUE_DADOS_FORTALEZA.CSV"
SANITATION_PATH = BASE_DIR / "dados_formatados" / "SANEAMENTO_DADOS_FORTALEZA.csv"


def build_consolidated_dataframe() -> pd.DataFrame:
    """
    Consolidação alinhada ao notebook consolidacao_e_correlacao.ipynb:
    1) lê clima, infodengue e saneamento;
    2) faz merge por MES_REFERENCIA;
    3) aplica fillna(0).

    Além disso, converte MES_REFERENCIA para datetime para padronização temporal.
    """
    # Clima vem com separador ';' e decimal com vírgula.
    df_inmet = pd.read_csv(CLIMATE_PATH, sep=";", decimal=",", encoding="utf-8-sig")
    # InfoDengue e saneamento já estão em CSV padrão.
    df_infodengue = pd.read_csv(DENGUE_PATH)
    df_saneamento = pd.read_csv(SANITATION_PATH)

    # Mantém a mesma estratégia do notebook: merge por mês de referência.
    df_final = pd.merge(df_inmet, df_infodengue, on="MES_REFERENCIA", how="left")
    df_final = pd.merge(df_final, df_saneamento, on="MES_REFERENCIA", how="left")
    # Buracos de informação continuam como zero para manter compatibilidade com o fluxo atual.
    df_final = df_final.fillna(0)

    # Padroniza o campo temporal para facilitar ordenação e análises posteriores.
    df_final["MES_REFERENCIA"] = pd.to_datetime(df_final["MES_REFERENCIA"], errors="coerce")
    df_final = df_final.dropna(subset=["MES_REFERENCIA"]).sort_values("MES_REFERENCIA").reset_index(drop=True)
    return df_final
