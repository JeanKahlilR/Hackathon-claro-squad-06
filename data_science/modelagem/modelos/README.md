# Modelos: Como Adicionar Um Novo Modelo

Este diretório concentra os wrappers de modelos de ML e a API pública usada pelo restante do módulo.

## Estrutura Atual

- `base.py`: contrato comum (`ModeloBase`)
- `regressao_logistica.py`: implementação de Regressão Logística
- `gradient_boosting.py`: implementação de Gradient Boosting
- `factory.py`: registry e função `criar_modelo(...)`
- `__init__.py`: reexporta a API pública do pacote

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
- Em inferência, use as mesmas colunas de treino (`X[self.features_treino]`).

### 3. Registrar o modelo na factory

Edite `factory.py` e adicione aliases no `REGISTRY_MODELOS` apontando para a nova classe.

Exemplo de aliases:

- `"random_forest"`
- `"rf"`

Com isso, `criar_modelo("rf", **params)` passa a funcionar.

### 4. Exportar o modelo no `__init__.py`

Edite `__init__.py` para:

- importar a nova classe
- incluir a nova classe em `__all__`

Isso mantém a API pública consistente:

```python
from modelagem.modelos import NovoModelo
```
