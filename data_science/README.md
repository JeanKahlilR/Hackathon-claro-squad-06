# data_science

Este diretório reúne os componentes do pipeline de dados e modelagem do projeto.

## Estrutura

- `modelagem/`: preparação de dados para modelagem, métricas, modelos e treino incremental.
- `processamento_dados/`: carregamento das fontes e criação de sessão Spark.
- `utils/`: utilitários de apoio (ex.: configuração de logger).
- `exemplo_treinamento.py`: script de execução de exemplo (LR + GB + comparação).
- `treinar_histgb.py`: script de execução focado em HistGradientBoosting.

---

## Scripts de execução

### `exemplo_treinamento.py`

Executa um pipeline completo de treinamento incremental com:

1. carregamento de dados,
2. split treino/OOT,
3. treino por etapas para Regressão Logística e Gradient Boosting,
4. comparação de modelos,
5. salvamento de resultados e gráficos.

### `treinar_histgb.py`

Executa pipeline semelhante, mas treinando apenas `Hist Gradient Boosting`,
além de salvar artefatos adicionais (matriz de confusão consolidada e modelo final em `.pkl`).

---

## Local da pasta `data/` (importante)

Os dois scripts (`exemplo_treinamento.py` e `treinar_histgb.py`) montam o caminho de dados como:

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
