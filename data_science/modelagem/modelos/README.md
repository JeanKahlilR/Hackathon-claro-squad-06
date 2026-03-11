# Modelos: Como Adicionar Um Novo Modelo

Este diretorio concentra os wrappers de modelos de ML e a API publica usada pelo restante do modulo.

## Estrutura Atual

- `base.py`: contrato comum (`ModeloBase`)
- `regressao_logistica.py`: implementacao de Regressao Logistica
- `gradient_boosting.py`: implementacao de Gradient Boosting
- `hist_gradient_boosting.py`: implementacao de HistGradientBoosting
- `xgboost.py`: implementacao de XGBoost
- `factory.py`: registry e funcao `criar_modelo_factory(...)`
- `__init__.py`: reexporta a API publica do pacote

### Modelos atualmente disponiveis

- `RegressaoLogistica`
- `GradientBoosting`
- `HistGradientBoosting`
- `XGBoost`

## Passo a Passo Para Adicionar Um Novo Modelo

### 1. Criar o arquivo do modelo

Crie um novo arquivo em `modelagem/modelos/`, por exemplo:

- `random_forest.py`

### 2. Implementar a classe do modelo

No novo arquivo, crie uma classe herdando de `ModeloBase` e implemente:

- `fit(self, X, y)`
- `predict_proba(self, X)`
- `get_feature_importance(self)` (opcional, mas recomendado)

Regras importantes:

- Defina `self.features_treino` durante o `fit`.
- Defina `self.is_fitted = True` ao final do `fit`.
- Em `predict_proba`, valide `self.is_fitted` antes de prever.
- Em inferencia, use as mesmas colunas de treino (`X[self.features_treino]`).

### 3. Registrar o modelo na factory

Edite `factory.py` e adicione aliases no `REGISTRY_MODELOS` apontando para a nova classe.

Exemplo de aliases:

- `"random_forest"`
- `"rf"`

Com isso, `criar_modelo_factory("rf", **params)` passa a funcionar.

Exemplo real (ja implementado para HistGradientBoosting):

- `"hist_gradient_boosting"`
- `"hgb"`
- `"hist_gb"`

Com isso, `criar_modelo_factory("hgb", **params)` passa a funcionar.

Exemplo real (ja implementado para XGBoost):

- `"xgboost"`
- `"xgb"`
- `"xg_boost"`

Com isso, `criar_modelo_factory("xgb", **params)` passa a funcionar.

### 4. Exportar o modelo no `__init__.py`

Edite `__init__.py` para:

- importar a nova classe
- incluir a nova classe em `__all__`

Isso mantem a API publica consistente:

```python
from modelagem.modelos import NovoModelo
```
