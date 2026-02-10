# Hackathon: Risco de Crédito
![Descrição da Imagem](docs/Capa.png)

# Problema de Negócio
Uma grande empresa do setor de telecomunicações busca expandir sua base de clientes em planos de maior valor (Controle) de forma sustentável. O desafio central é equilibrar o crescimento da carteira com uma gestão de risco eficiente, focando na migração de usuários do segmento Pré-pago, que representam a principal porta de entrada de novos clientes.

O objetivo deste projeto é desenvolver um Modelo de Behavior baseado na Visão Unificada do Cliente (CPF) para estimar a probabilidade de inadimplência durante esse processo de migração.

Objetivos Principais:
- Identificar perfis de baixo risco: Localizar clientes pré-pagos com maior potencial de adimplência em planos Controle.

- Otimizar Ofertas: Direcionar propostas de forma mais eficiente e personalizada.

- Rentabilidade: Reduzir o risco de crédito e aumentar a saúde financeira da carteira de clientes.

---

# Ciência de Dados: Instalação e Execução

Esta seção descreve como preparar os dados e executar os pipelines de modelagem da pasta `data_science/`.

## 1) Estrutura esperada da pasta `data/`

Os scripts de exemplo (`data_science/exemplo_treinamento.py` e `data_science/treinar_histgb.py`) procuram os dados em:

```text
<raiz-do-projeto>/data
```

Dentro de `data/`, organize os dados em **subpastas** com os nomes:

- `score_01`
- `score_02`
- `plus_telco`
- `plus_cadastral`
- `plus_recarga`
- `plus_pagamento`

> Observação: para rodar o fluxo incremental completo com todas as etapas padrão (1 a 6), também é esperado o conjunto da etapa cadastral (`plus_cadastral`).

Cada subpasta pode conter arquivos `.csv` e/ou `.parquet`.

Exemplo de layout:

```text
Hackathon-claro-squad-06/
├── data/
│   ├── score_01/
│   ├── score_02/
│   ├── plus_telco/
│   ├── plus_cadastral/
│   ├── plus_recarga/
│   └── plus_pagamento/
└── data_science/
```

## 2) Modo recomendado: `uv`

Use `uv` como forma principal de instalação e execução.

### 2.1) Instalar dependências

Na raiz do projeto:

```bash
uv sync
```

### 2.2) Rodar o exemplo completo (LR + GB + comparação)

```bash
uv run python data_science/exemplo_treinamento.py
```

Esse script executa:

- carregamento dos dados,
- split treino/OOT,
- treino incremental com Regressão Logística,
- treino incremental com Gradient Boosting,
- comparação entre modelos,
- salvamento dos resultados e gráficos.

### 2.3) Rodar o exemplo focado em HistGradientBoosting

```bash
uv run python data_science/treinar_histgb.py
```

Esse script executa:

- carregamento dos dados,
- split treino/OOT,
- treino incremental com HistGradientBoosting,
- salvamento de resultados,
- exportação de matriz de confusão consolidada,
- exportação do modelo final (`.pkl`).

## 3) Modo alternativo: `pip`

Se você preferir não usar `uv`, rode com `pip` usando o `requirements.txt`.

### 3.1) Criar e ativar ambiente virtual

```bash
python -m venv .venv
```

Ative o ambiente virtual:

- Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

- Windows (cmd):

```bash
.venv\Scripts\activate.bat
```

### 3.2) Instalar e executar

```bash
python -m pip install -r requirements.txt
python data_science/exemplo_treinamento.py
python data_science/treinar_histgb.py
```

## 4) Logs e outputs

Durante a execução, arquivos e diretórios de saída são gerados automaticamente, como:

- `logs/` (arquivos de log)
- `output/` (resultados CSV, gráficos KS, modelo salvo etc.)

Essas pastas são criadas no **diretório de execução** (diretório atual de onde o comando é rodado).
