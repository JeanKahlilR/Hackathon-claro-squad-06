# Modelagem

Este diretório concentra os componentes de modelagem do projeto: preparação de dados,
configurações, métricas e pacotes de modelos/treinamento.

## Visão geral

- `preparacao_dados.py`: pipeline de preparação (remoção de grupo controle e split treino/OOT).
- `config.py`: constantes e mapeamentos de etapas/features.
- `metricas.py`: métricas e avaliação de performance (KS, AUC, Gini, matriz de confusão etc.).
- `modelos/`: wrappers dos modelos e factory de criação.
- `treinamento_incremental/`: orquestração do treino incremental por etapa.

---

## Subpacotes principais

### 1) `treinamento_incremental`

Responsável por executar o fluxo incremental de treino e avaliação por etapa via
classe `TreinamentoIncremental`, incluindo resumo de resultados e geração de gráficos KS.

📘 **Documentação detalhada:**
- `data_science/modelagem/treinamento_incremental/README.md`

### 2) `modelos`

Responsável por encapsular os modelos (ex.: Regressão Logística, Gradient Boosting,
HistGradientBoosting) com contrato comum e criação centralizada via factory.

📘 **Documentação detalhada:**
- `data_science/modelagem/modelos/README.md`

---

## API pública do pacote

Atualmente, o `__init__.py` de `modelagem` expõe:

- `preparar_dados_modelagem`
- `TreinamentoIncremental`

Exemplo:

```python
from modelagem import preparar_dados_modelagem, TreinamentoIncremental
```
