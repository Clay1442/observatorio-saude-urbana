# Analise de Correlacao entre Variaveis Meteorologicas e Indicadores de Arboviroses em Fortaleza

Projeto academico de Big Data para preparar bases meteorologicas e epidemiologicas de Fortaleza, permitindo analises historicas, correlacoes, padroes temporais e suporte a etapa de Diagnostico e Teorizacao.

## Estrutura

```text
.
├── data/
│   ├── processed/
│   └── external/
├── dados_formatados/
├── dados_infodengue/
├── dados_inmet/
├── docs/
├── models/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   └── utils/
├── requirements.txt
└── README.md
```

## Instalacao

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Se o comando `python` nao estiver disponivel, use o interpretador do ambiente virtual, por exemplo `.venv/bin/python`.

## Tratamento InfoDengue

Entrada:

- `dados_infodengue/infodengue_fortaleza_2017_2024.csv`

Execucao:

```bash
.venv/bin/python src/data/process_infodengue.py
```

Saida:

- `data/processed/infodengue_fortaleza_tratado.csv`

O script filtra Fortaleza por colunas de localidade/codigo IBGE, converte `data_iniSE` para datetime, ordena cronologicamente e garante serie semanal continua. A coluna `casos_est` e priorizada por compensar atraso de notificacao e subnotificacao. Semanas inseridas artificialmente recebem zero nas colunas de casos.

## Modelo inicial de chuva

Entrada:

- `dados_formatados/FORTALEZA_DADOS_CLIMATICOS.csv`

Execucao:

```bash
.venv/bin/python src/models/train_rain_classifier.py
```

Saidas:

- `models/random_forest_chuva.pkl`
- `models/random_forest_chuva_metrics.json`

O alvo categorico e criado a partir de `CHUVA`:

- `0`: Sem chuva, 0 mm
- `1`: Chuva fraca, 0.1 mm ate 5 mm
- `2`: Chuva moderada/forte, acima de 5 mm

Variaveis usadas pelo baseline atual:

- `temp`
- `umid`
- `press`
- `ponto_orvalho`, calculado a partir de temperatura e umidade
- `delta_pressao`

Limitacao: `lux` nao existe no dataset climatico atual. A coluna `RAD` foi mantida como radiacao (`rad`) e nao foi renomeada para `lux` para evitar mistura metodologica.

## Consumo de leituras recentes para o modelo

Execucao com fallback local:

```bash
.venv/bin/python src/data/fetch_latest_readings.py --limit 24
```

Execucao com API, quando estiver disponivel:

```bash
.venv/bin/python src/data/fetch_latest_readings.py --api-url "https://exemplo/api/leituras"
```

Saida:

- `data/processed/latest_weather_readings_for_model.csv`

O script valida campos esperados, padroniza nomes, calcula `ponto_orvalho` quando possivel, trata valores ausentes e retorna apenas as colunas usadas pelo modelo salvo.

## Resultados atuais

O baseline Random Forest compacto foi treinado para manter o artefato versionavel. Metricas principais em `models/random_forest_chuva_metrics.json`:

- Acuracia: `0.8206`
- Variavel mais importante: `umid`
- Campo esperado ausente: `lux`

Como a base e desbalanceada, a acuracia deve ser interpretada junto com matriz de confusao, recall por classe e proximas avaliacoes temporais.

## Proximos passos

- Validar com o grupo se `RAD` pode ou nao representar uma proxy de luminosidade.
- Definir fonte oficial de leituras recentes: API, banco remoto ou rotina de exportacao CSV.
- Avaliar divisao temporal de treino/teste para reduzir vazamento em series historicas.
- Testar modelos e tecnicas para classes desbalanceadas.
- Integrar a base epidemiologica tratada com agregacoes meteorologicas semanais.
