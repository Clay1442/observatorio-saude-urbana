from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
CLIMATE_PATH = BASE_DIR / "dados_formatados" / "FORTALEZA_DADOS_CLIMATICOS.csv"
DENGUE_PATH = BASE_DIR / "dados_formatados" / "INFODENGUE_DADOS_FORTALEZA.CSV"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "analise_climatica"


def load_data() -> pd.DataFrame:
    # Lê a base climática mensal (separador ; e vírgula decimal).
    climate = pd.read_csv(
        CLIMATE_PATH,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )
    climate = climate.rename(
        columns={
            "MES_REFERENCIA": "mes_referencia",
            "Chuva_Acumulada": "chuva_acumulada",
            "Temp_Media_Mensal": "temp_media",
            "Umid_Media": "umid_media",
            "Pres_Media": "pres_media",
        }
    )
    climate["mes_referencia"] = pd.to_datetime(climate["mes_referencia"], errors="coerce")

    # Lê a base epidemiológica mensal consolidada.
    dengue = pd.read_csv(DENGUE_PATH)
    dengue = dengue.rename(
        columns={
            "MES_REFERENCIA": "mes_referencia",
            "casos_mensais": "casos_dengue",
        }
    )
    dengue["mes_referencia"] = pd.to_datetime(dengue["mes_referencia"], errors="coerce")

    # Junta apenas os meses que existem nas duas bases para evitar desalinhamento temporal.
    merged = climate.merge(dengue[["mes_referencia", "casos_dengue"]], on="mes_referencia", how="inner")
    merged = merged.dropna(subset=["mes_referencia"]).sort_values("mes_referencia").reset_index(drop=True)
    return merged


def validate_input_data(df: pd.DataFrame) -> None:
    # Checagens mínimas para garantir que a análise não rode com dados quebrados.
    expected_cols = ["mes_referencia", "chuva_acumulada", "temp_media", "umid_media", "pres_media", "casos_dengue"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")
    if df.empty:
        raise ValueError("Base consolidada vazia após merge.")
    if not df["mes_referencia"].is_monotonic_increasing:
        raise ValueError("Série temporal não está ordenada por mês.")
    if df[expected_cols[1:]].isna().any().any():
        raise ValueError("Há valores nulos em colunas numéricas críticas.")


def pressure_drop_before_rain(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    # Delta de pressão: valor atual menos valor do mês anterior.
    aux = df.copy()
    aux["pres_anterior"] = aux["pres_media"].shift(1)
    aux["delta_p"] = aux["pres_media"] - aux["pres_anterior"]
    aux["queda_antecede"] = aux["pres_anterior"] - aux["pres_media"]

    rainy = aux[(aux["chuva_acumulada"] > 0) & aux["queda_antecede"].notna()]
    mean_drop = rainy.loc[rainy["queda_antecede"] > 0, "queda_antecede"].mean()
    return rainy, float(mean_drop) if pd.notna(mean_drop) else float("nan")


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["temp_media", "umid_media", "casos_dengue"]
    return df[cols].corr(method="pearson")


def lag_analysis(df: pd.DataFrame, max_lag_months: int = 6) -> pd.DataFrame:
    # Testa atrasos de 0..max_lag e mede correlação com casos de dengue.
    rows: list[dict[str, float | str]] = []
    climate_cols = ["temp_media", "umid_media", "pres_media", "chuva_acumulada"]

    for var in climate_cols:
        for lag_m in range(0, max_lag_months + 1):
            shifted = df[var].shift(lag_m)
            corr = shifted.corr(df["casos_dengue"])
            rows.append(
                {
                    "variavel": var,
                    "lag_meses": lag_m,
                    "lag_semanas_aprox": lag_m * 4.345,
                    "correlacao_com_casos": corr,
                }
            )
    lag_df = pd.DataFrame(rows).dropna()
    return lag_df


def trigger_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    # Define "surto" como meses no quartil superior de casos.
    outbreak_threshold = df["casos_dengue"].quantile(0.75)
    df = df.copy()
    df["surto"] = df["casos_dengue"] >= outbreak_threshold

    trigger = (df["umid_media"] > 80) & (df["temp_media"].between(24, 30))
    base_prob = df["surto"].mean()
    trigger_prob = df.loc[trigger, "surto"].mean() if trigger.any() else np.nan
    support = int(trigger.sum())

    # Busca em grade simples para gatilho com melhor lift e suporte mínimo.
    best = {
        "umid_threshold": np.nan,
        "temp_min": np.nan,
        "temp_max": np.nan,
        "support": 0,
        "trigger_prob": np.nan,
        "lift": np.nan,
    }
    for umid_t in [70, 75, 80, 85]:
        for tmin, tmax in [(22, 28), (24, 30), (26, 32)]:
            cond = (df["umid_media"] > umid_t) & (df["temp_media"].between(tmin, tmax))
            n = int(cond.sum())
            if n < 5:
                continue
            p = df.loc[cond, "surto"].mean()
            lift = p / base_prob if base_prob > 0 else np.nan
            if np.isnan(best["lift"]) or lift > best["lift"]:
                best = {
                    "umid_threshold": umid_t,
                    "temp_min": tmin,
                    "temp_max": tmax,
                    "support": n,
                    "trigger_prob": p,
                    "lift": lift,
                }

    summary = {
        "outbreak_threshold_q75": float(outbreak_threshold),
        "base_surge_probability": float(base_prob),
        "example_trigger_probability": float(trigger_prob) if pd.notna(trigger_prob) else float("nan"),
        "example_trigger_support_months": float(support),
        "example_trigger_lift": float(trigger_prob / base_prob) if base_prob > 0 and pd.notna(trigger_prob) else float("nan"),
    }
    summary.update({f"best_{k}": float(v) if isinstance(v, (np.floating, float, int)) else v for k, v in best.items()})
    return df, summary


def save_plots(df: pd.DataFrame, corr: pd.DataFrame, lag_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Heatmap da matriz de correlação de Pearson.
    plt.figure(figsize=(7, 5))
    mat = corr.values
    plt.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlação de Pearson")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")
    plt.title("Matriz de Correlação (Pearson)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "matriz_correlacao_pearson.png", dpi=150)
    plt.close()

    # Curvas de correlação por defasagem temporal (lag).
    plt.figure(figsize=(9, 5))
    for var, group in lag_df.groupby("variavel"):
        plt.plot(group["lag_semanas_aprox"], group["correlacao_com_casos"], marker="o", label=var)
    plt.axhline(0, color="gray", linewidth=1)
    plt.xlabel("Lag (semanas, aprox.)")
    plt.ylabel("Correlação com casos de dengue")
    plt.title("Análise de Lag Clima x Dengue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "analise_lag_clima_dengue.png", dpi=150)
    plt.close()

    # Série conjunta de pressão e chuva para inspeção temporal.
    plt.figure(figsize=(10, 5))
    plt.plot(df["mes_referencia"], df["pres_media"], label="Pressão média (mB)")
    plt.plot(df["mes_referencia"], df["chuva_acumulada"], label="Chuva acumulada (mm)")
    plt.title("Série Temporal: Pressão e Chuva")
    plt.xlabel("Mês")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "serie_pressao_chuva.png", dpi=150)
    plt.close()

    # Tendências normalizadas (z-score) suavizadas por média móvel.
    plt.figure(figsize=(10, 5))
    rolling = df.set_index("mes_referencia")[["casos_dengue", "chuva_acumulada", "umid_media", "temp_media"]].rolling(3).mean()
    norm = (rolling - rolling.mean()) / rolling.std(ddof=0)
    for col in norm.columns:
        plt.plot(norm.index, norm[col], label=col)
    plt.axhline(0, color="gray", linewidth=1)
    plt.title("Séries Padronizadas (z-score) com Média Móvel de 3 Meses")
    plt.xlabel("Mês")
    plt.ylabel("z-score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "series_padronizadas_tendencia.png", dpi=150)
    plt.close()

    # Dispersões com reta linear para leitura visual de relação.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, xcol, title in [
        (axes[0], "umid_media", "Umidade x Casos"),
        (axes[1], "temp_media", "Temperatura x Casos"),
    ]:
        x = df[xcol].to_numpy()
        y = df["casos_dengue"].to_numpy()
        ax.scatter(x, y, alpha=0.7)
        coeffs = np.polyfit(x, y, deg=1)
        xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
        yy = coeffs[0] * xx + coeffs[1]
        ax.plot(xx, yy, color="red", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel(xcol)
        ax.set_ylabel("casos_dengue")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dispersao_clima_casos.png", dpi=150)
    plt.close()

    # Sazonalidade média por mês do ano.
    seasonal = df.copy()
    seasonal["mes"] = seasonal["mes_referencia"].dt.month
    grouped = seasonal.groupby("mes", as_index=False)[["chuva_acumulada", "umid_media", "temp_media", "casos_dengue"]].mean()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(grouped["mes"], grouped["casos_dengue"], marker="o", color="tab:red", label="casos_dengue")
    ax2.plot(grouped["mes"], grouped["chuva_acumulada"], marker="o", color="tab:blue", label="chuva_acumulada")
    ax1.set_title("Sazonalidade Média Mensal: Casos e Chuva")
    ax1.set_xlabel("Mês do ano")
    ax1.set_ylabel("Casos de dengue", color="tab:red")
    ax2.set_ylabel("Chuva acumulada (mm)", color="tab:blue")
    ax1.set_xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sazonalidade_casos_chuva.png", dpi=150)
    plt.close()

    # Probabilidade de surto em bins de temperatura x umidade.
    outbreak_threshold = df["casos_dengue"].quantile(0.75)
    hbins = [65, 70, 75, 80, 85, 90]
    tbins = [22, 24, 26, 28, 30, 32]
    heat = df.copy()
    heat["surto"] = heat["casos_dengue"] >= outbreak_threshold
    heat["h_bin"] = pd.cut(heat["umid_media"], bins=hbins, include_lowest=True)
    heat["t_bin"] = pd.cut(heat["temp_media"], bins=tbins, include_lowest=True)
    pivot = heat.pivot_table(index="h_bin", columns="t_bin", values="surto", aggfunc="mean")

    plt.figure(figsize=(10, 5))
    mat = pivot.to_numpy(dtype=float)
    im = plt.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, label="Probabilidade de surto")
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Heatmap de Probabilidade de Surto (Temp x Umidade)")
    plt.xlabel("Faixas de temperatura")
    plt.ylabel("Faixas de umidade")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_prob_surto_temp_umidade.png", dpi=150)
    plt.close()


def validate_generated_outputs(corr: pd.DataFrame, lag_df: pd.DataFrame) -> list[str]:
    notes: list[str] = []

    # Valida que a correlação está no intervalo teórico [-1, 1].
    corr_vals = corr.to_numpy(dtype=float)
    if np.nanmin(corr_vals) < -1.0001 or np.nanmax(corr_vals) > 1.0001:
        raise ValueError("Matriz de correlação fora do intervalo [-1, 1].")
    notes.append(
        f"correlacao_ok=min:{np.nanmin(corr_vals):.3f},max:{np.nanmax(corr_vals):.3f}"
    )

    # Valida que cada variável possui série de lag e ao menos um valor útil.
    vars_expected = {"temp_media", "umid_media", "pres_media", "chuva_acumulada"}
    vars_found = set(lag_df["variavel"].unique())
    if vars_found != vars_expected:
        raise ValueError(f"Lag incompleto. esperado={vars_expected}, encontrado={vars_found}")
    if lag_df["correlacao_com_casos"].isna().all():
        raise ValueError("Lag sem correlações válidas.")
    notes.append(f"lag_ok=linhas:{len(lag_df)}")

    # Valida existência e sanidade dos PNGs gerados.
    png_files = sorted(OUTPUT_DIR.glob("*.png"))
    if not png_files:
        raise ValueError("Nenhum gráfico PNG foi gerado.")
    for png in png_files:
        if png.stat().st_size < 10_000:
            raise ValueError(f"Gráfico potencialmente vazio ou corrompido: {png.name}")
        img = plt.imread(png)
        if img.ndim < 2 or img.shape[0] < 100 or img.shape[1] < 100:
            raise ValueError(f"Resolução inesperada para gráfico: {png.name} {img.shape}")
        # Se a variação for quase nula, provavelmente é imagem em branco.
        if float(np.nanstd(img)) < 0.001:
            raise ValueError(f"Gráfico sem variação visual detectável: {png.name}")
    notes.append(f"png_ok=arquivos:{len(png_files)}")

    return notes


def main() -> None:
    df = load_data()
    validate_input_data(df)
    rainy, mean_drop = pressure_drop_before_rain(df)
    corr = correlation_matrix(df)
    lag_df = lag_analysis(df)
    trigger_df, trigger_summary = trigger_analysis(df)

    save_plots(df, corr, lag_df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rainy[["mes_referencia", "chuva_acumulada", "pres_anterior", "pres_media", "delta_p", "queda_antecede"]].to_csv(
        OUTPUT_DIR / "queda_pressao_antes_chuva.csv", index=False
    )
    lag_df.to_csv(OUTPUT_DIR / "lag_clima_dengue.csv", index=False)
    trigger_df.to_csv(OUTPUT_DIR / "base_analise_com_trigger.csv", index=False)
    corr.to_csv(OUTPUT_DIR / "matriz_correlacao_pearson.csv")

    best_lags = (
        lag_df.assign(abs_corr=lambda x: x["correlacao_com_casos"].abs())
        .sort_values(["variavel", "abs_corr"], ascending=[True, False])
        .groupby("variavel", as_index=False)
        .first()
    )

    summary_lines = [
        "Resumo da análise climática e epidemiológica (Fortaleza)",
        f"Total de meses analisados: {len(df)}",
        f"Queda média de pressão antes de meses com chuva: {mean_drop:.3f} mB",
        "",
        "Melhor lag por variável climática (|correlação| máxima):",
    ]
    for _, row in best_lags.iterrows():
        summary_lines.append(
            f"- {row['variavel']}: {row['lag_meses']} mês(es) (~{row['lag_semanas_aprox']:.1f} semanas), "
            f"corr={row['correlacao_com_casos']:.3f}"
        )

    summary_lines.extend(
        [
            "",
            "Gatilhos de surto (surto definido por quartil 75 de casos de dengue):",
            f"- Probabilidade base de surto: {trigger_summary['base_surge_probability']:.3f}",
            f"- Trigger exemplo (umidade > 80 e 24<=temp<=30): {trigger_summary['example_trigger_probability']:.3f} "
            f"(suporte={int(trigger_summary['example_trigger_support_months'])} meses, "
            f"lift={trigger_summary['example_trigger_lift']:.3f})",
            f"- Melhor trigger simples encontrado: umidade > {trigger_summary['best_umid_threshold']:.0f}, "
            f"{trigger_summary['best_temp_min']:.0f}<=temp<={trigger_summary['best_temp_max']:.0f}, "
            f"prob={trigger_summary['best_trigger_prob']:.3f}, lift={trigger_summary['best_lift']:.3f}, "
            f"suporte={int(trigger_summary['best_support'])}",
            "",
            "Observação: a base atual não contém série de Zika separada; a correlação foi calculada para dengue.",
        ]
    )

    (OUTPUT_DIR / "resumo_analise.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    validation_notes = validate_generated_outputs(corr, lag_df)
    with (OUTPUT_DIR / "resumo_analise.txt").open("a", encoding="utf-8") as f:
        f.write("\n\nValidações automáticas:\n")
        for note in validation_notes:
            f.write(f"- {note}\n")

    print("\n".join(summary_lines))
    print("\nValidações automáticas:")
    for note in validation_notes:
        print(f"- {note}")
    print(f"\nArquivos salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
