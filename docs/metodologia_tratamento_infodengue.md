# Metodologia de tratamento do dataset InfoDengue

## Origem

O arquivo tratado parte de `dados_infodengue/infodengue_fortaleza_2017_2024.csv`, base historica semanal do InfoDengue usada no projeto para analisar indicadores epidemiologicos de arboviroses em Fortaleza.

## Filtro de localidade

O script `src/data/process_infodengue.py` verifica colunas possiveis de localidade, incluindo:

- `municipio_nome`
- `municipio`
- `cidade`
- `nome_municipio`
- `localidade`
- colunas de codigo IBGE, quando existirem

No arquivo atual, `municipio_nome` contem apenas `Fortaleza`. Ainda assim, o filtro foi mantido no codigo para impedir que registros de outras localidades entrem no dataset final em futuras atualizacoes.

## Tratamento temporal

A coluna `data_iniSE` e convertida com `pd.to_datetime(..., errors="coerce")`. Datas invalidas viram nulas e sao removidas com aviso no terminal.

Depois da conversao, os dados sao ordenados cronologicamente e reindexados com frequencia semanal de domingo (`W-SUN`), coerente com as datas iniciais observadas no arquivo. Caso uma semana esteja ausente, o pipeline insere a linha e marca `registro_original = False`.

No processamento atual, o periodo final possui 417 semanas, de `2017-01-01` a `2024-12-22`, sem lacunas detectadas.

## Priorizacao de casos estimados

A coluna principal para analise e `casos_est`, pois representa casos estimados e ajuda a compensar atrasos de notificacao e subnotificacoes. Essa escolha e mais adequada para comparacoes historicas e identificacao de surtos do que usar apenas notificacoes brutas.

Regras aplicadas:

- se `casos_est` existir, ela e mantida como coluna prioritaria;
- se houver nulos em `casos_est` e `casos` estiver disponivel, `casos` preenche esses nulos pontuais;
- lacunas restantes em semanas existentes podem ser interpoladas linearmente;
- semanas criadas artificialmente recebem zero em colunas de casos/notificacoes, pois representam ausencia de registro naquela semana no arquivo.

## Dataset final

Arquivo gerado:

- `data/processed/infodengue_fortaleza_tratado.csv`

Colunas essenciais:

- `data_semana_epidemiologica`
- `ano`
- `mes`
- `semana_epidemiologica`
- `SE`
- `municipio`
- `casos_est`
- `casos_notificados`
- indicadores epidemiologicos auxiliares, como `p_inc100k`, `Rt`, `nivel`, `nivel_inc`, `receptivo` e `transmissao`
- variaveis climaticas semanais presentes na propria base InfoDengue, como `tempmed` e `umidmed`

## Limitacoes

- O arquivo atual nao possui coluna explicita de doenca ou tipo de arbovirose. A documentacao do card do grupo deve confirmar se a base representa dengue especificamente ou uma composicao de arboviroses.
- A semana epidemiologica foi derivada de calendario ISO para as linhas inseridas. Para estudos oficiais, pode ser necessario validar a regra epidemiologica usada pelo InfoDengue.
- Nao havia lacunas semanais no arquivo atual, mas a rotina foi preparada para preencher lacunas em atualizacoes futuras.

## Proximos passos

- Confirmar com o grupo a interpretacao da doenca/arbovirose da base.
- Criar uma base semanal integrada com chuva, temperatura, umidade e pressao agregadas.
- Avaliar correlacoes com defasagens temporais, por exemplo chuva de 1 a 8 semanas antes dos casos.
