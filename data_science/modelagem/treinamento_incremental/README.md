# Treinamento Incremental

Este pacote centraliza o fluxo de treino incremental via a classe **`TreinamentoIncremental`**.

Ele recebe tabelas já separadas em **treino/OOT** e executa, por etapa, o ciclo:

1. preparação de features,
2. treino do modelo,
3. predição no OOT,
4. avaliação (KS, AUC, Gini, etc.),
5. geração opcional de gráfico KS.

---

## Classe principal

```python
from modelagem import TreinamentoIncremental
```

### Construtor

```python
TreinamentoIncremental(
    tables_splited: TablesSplited,
    etapas: Optional[List[int]] = None,
)
```

- `tables_splited`: dicionário no formato:
  - `{"nome_tabela": {"treino": df_treino, "oot": df_oot}}`
  - `df_treino`/`df_oot` podem ser Spark DataFrame ou pandas DataFrame.
- `etapas`: etapas a executar. Se `None`, usa as etapas disponíveis na configuração.

---

## Configurações do `config.py` usadas pelo orquestrador

O arquivo `data_science/modelagem/config.py` possui várias constantes, mas o
`TreinamentoIncremental` importa diretamente estas:

- `BENCHMARK_KS`
- `COLUNAS_EXCLUIR`
- `TARGET_COLUMN`
- `listar_etapas_disponiveis()`
- `obter_nome_etapa()`

### `BENCHMARK_KS`

- **Onde é usado:** logs de ganho incremental por etapa.
- **Impacto:** muda a referência exibida em `Delta vs Benchmark` durante o treino.
- **Não altera o treino**, mas altera a leitura de performance no relatório/log.

### `TARGET_COLUMN`

- **Onde é usado:** validação e montagem de `y_treino`/`y_oot`.
- **Impacto:** define qual coluna é tratada como target no fluxo.
- Se estiver incorreta/inexistente nas tabelas da etapa, o orquestrador lança erro.

### `COLUNAS_EXCLUIR`

- **Onde é usado:** seleção automática de features na etapa (`_preparar_dados_etapa`).
- **Impacto:** tudo que estiver nessa lista é removido das features de treino/OOT.
- Permite controlar, via configuração, quais colunas nunca entram no modelo.

### `listar_etapas_disponiveis()`

- **Onde é usado:** no `__init__`, quando `etapas=None`.
- **Impacto:** define automaticamente quais etapas serão executadas por padrão.
- A função, no config atual, deriva as etapas de `FEATURES_POR_ETAPA`.

### `obter_nome_etapa(etapa)`

- **Onde é usado:** logs e metadados de resultado.
- **Impacto:** nome amigável da etapa no resumo e nos resultados armazenados.

### Exemplo rápido de ajuste no `config.py`

```python
# Exemplo: mudar target e benchmark
TARGET_COLUMN = "MEU_TARGET"
BENCHMARK_KS = 35.0

# Exemplo: forçar remoção de coluna adicional das features
COLUNAS_EXCLUIR = [
    "NUM_CPF", "ID_UNICO", "SAFRA", "MEU_TARGET", "FLAG_INSTALACAO",
    "GRUPO_CONTROLE", "PROD", "flag_mig2", "coluna_que_nao_quero_no_modelo"
]
```

> Observação: outras configurações do `config.py` (ex.: `SAFRA_OOT`,
> `GRUPO_CONTROLE_DIGITOS`, `FEATURES_POR_ETAPA`) são relevantes para o pipeline
> completo, mas não são consumidas diretamente pelo orquestrador incremental,
> exceto indiretamente via funções importadas (`listar_etapas_disponiveis` e
> `obter_nome_etapa`).

---

## Ordem recomendada de uso

### 1) Instanciar o orquestrador

```python
treinamento = TreinamentoIncremental(
    tables_splited=tabelas_treino_oot,
    etapas=[1, 2, 3, 4, 5, 6],
)
```

### 2) Executar treino incremental (1 ou mais modelos)

Use `executar_treinamento_incremental(...)` para cada tipo de modelo desejado.

```python
df_resumo_lr = treinamento.executar_treinamento_incremental(
    tipo_modelo="logistica",
    params_modelo={"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
    threshold=0.5,
    salvar_graficos_ks=True,
    output_dir_graficos="output/ks_por_etapa",
)
```

Também aceita aliases suportados pela factory, por exemplo:
- `"gradient_boosting"`, `"gb"`, `"gbm"`
- `"hist_gradient_boosting"`, `"hgb"`, `"hist_gb"`
- `"xgboost"`, `"xgb"`, `"xg_boost"`

### 3) (Opcional) Comparar LR vs GB

Se quiser comparação consolidada, use `comparar_modelos(...)`.

```python
resultados_comp = treinamento.comparar_modelos(
    df_resumo_lr=df_resumo_lr,
    # df_resumo_gb opcional; se não informar, ele treina GB internamente
)

df_comparacao = resultados_comp["comparacao"]
```

### 4) Salvar resultados tabulares

```python
treinamento.salvar_resultados("output/resultados_incrementais.csv")
```

### 5) (Opcional) Salvar/regerar gráficos KS a partir do estado em memória

Se você já executou treinos e quer gerar gráficos depois:

```python
treinamento.salvar_graficos_ks("output/ks_por_etapa")
```

---

## Métodos públicos

## `executar_treinamento_incremental(...) -> pd.DataFrame`

Executa o treino por etapa para **um tipo de modelo**.

Parâmetros principais:
- `tipo_modelo`: tipo/alias do modelo.
- `params_modelo`: hiperparâmetros.
- `threshold`: threshold de classificação binária para métricas.
- `salvar_graficos_ks`: gera PNG por etapa durante o treino.
- `output_dir_graficos`: pasta de saída dos PNGs.

Retorno:
- `DataFrame` resumo por etapa (KS, Delta_KS, AUC, Gini, etc.).

## `comparar_modelos(...) -> Dict[str, pd.DataFrame]`

Compara regressão logística e gradient boosting por etapa.

Retorna:
- `regressao_logistica`
- `gradient_boosting`
- `comparacao`

## `salvar_resultados(caminho) -> None`

Serializa os resultados acumulados (`self.resultados`) em CSV.

## `salvar_graficos_ks(output_dir_graficos) -> Dict[str, Dict[int, str]]`

Gera gráficos KS com base nos vetores armazenados em memória durante os treinos.

---

## Propriedades úteis

- `treinamento.resultados`: métricas detalhadas por `modelo_etapa`.
- `treinamento.modelos_treinados`: objetos de modelos já treinados.
- `treinamento.graficos_ks`: caminhos dos gráficos gerados por tipo de modelo e etapa.

---

## Exemplo completo (resumido)

```python
from modelagem import preparar_dados_modelagem, TreinamentoIncremental

tabelas_treino_oot, metadata, _ = preparar_dados_modelagem(tabelas)

treinamento = TreinamentoIncremental(tables_splited=tabelas_treino_oot, etapas=[1,2,3,4,5,6])

df_resumo_lr = treinamento.executar_treinamento_incremental(
    tipo_modelo="logistica",
    params_modelo={"C": 1.0, "max_iter": 1000},
)

df_resumo_hgb = treinamento.executar_treinamento_incremental(
    tipo_modelo="hgb",
    params_modelo={"max_iter": 1000, "learning_rate": 0.1},
)

treinamento.salvar_resultados("output/resultados_incrementais.csv")

print(treinamento.graficos_ks)
print(treinamento.modelos_treinados.keys())
```

---

## Observações importantes

- O pacote assume que cada etapa aponta para uma tabela agregada específica (mapeamento em `data_builder.ETAPA_PARA_TABELA`).
- A coluna target deve existir em treino e OOT (`TARGET_COLUMN` via configuração).
- A cada execução, os resultados são acumulados no estado interno da instância.
- Se reexecutar o mesmo `tipo_modelo` nas mesmas etapas, as chaves de `resultados` podem ser sobrescritas.
