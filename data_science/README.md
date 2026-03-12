# data_science

Este diretório reúne os componentes do pipeline de dados e modelagem do projeto.

## Estrutura

- `modelagem/`: preparação de dados para modelagem, métricas, modelos e treino incremental.
- `processamento_dados/`: carregamento das fontes e criação de sessão Spark.
- `utils/`: utilitários de apoio (ex.: configuração de logger e carregamento de `tabelas_treino_oot`).
- `treinar_histgb.py`: script de execução focado em HistGradientBoosting.
- `treinar_xgboost.py`: script de execução focado em XGBoost.

---

## Scripts de execução

### `treinar_histgb.py`

Executa um pipeline completo de treinamento incremental com `Hist Gradient Boosting`,
com geração de métricas, gráficos e artefatos finais de modelo.

### `treinar_xgboost.py`

Executa um pipeline completo de treinamento incremental com `XGBoost`,
com geração de métricas, gráficos e artefatos finais de modelo.

---

## Local da pasta `data/` (importante)

Os dois scripts (`treinar_histgb.py` e `treinar_xgboost.py`) montam o caminho de dados como:

- diretório pai de `data_science` + `data`

Ou seja, a pasta de dados deve estar em:

```text
<raiz-do-projeto>/data
```

No layout atual do repositório, isso corresponde a:

```text
Hackathon-claro-squad-06/
├── data/
└── data_science/
```

Se a pasta `data/` não estiver nesse local, os scripts não conseguirão carregar as fontes.

---

## Documentação dos submódulos

Para detalhes de uso, consulte os READMEs específicos:

- `data_science/modelagem/README.md`
- `data_science/modelagem/treinamento_incremental/README.md`
- `data_science/modelagem/modelos/README.md`
