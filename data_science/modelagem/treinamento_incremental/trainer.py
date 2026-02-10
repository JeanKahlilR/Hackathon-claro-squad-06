"""
Funcoes de criacao de modelo e treino/predicao para OOT.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..modelos import ModeloBase, criar_modelo_factory


def criar_modelo(
    tipo_modelo: str,
    etapa: int,
    params_modelo: Optional[Dict] = None,
) -> ModeloBase:
    """
    Cria uma instancia de modelo para uma etapa do fluxo incremental.

    Parameters
    ----------
    tipo_modelo : str
        Tipo do modelo (ex.: "logistica", "lr", "gradient_boosting",
        "gb", "hist_gradient_boosting", "hgb").
    etapa : int
        Numero da etapa usado para identificacao no nome do modelo.
    params_modelo : Optional[Dict], optional
        Hiperparametros do estimador.

    Returns
    -------
    ModeloBase
        Modelo inicializado e pronto para treinamento.

    Raises
    ------
    ValueError
        Quando o tipo informado nao pertence aos tipos suportados.
    """
    params_modelo = params_modelo or {}
    tipo = tipo_modelo.lower()

    nomes_padrao = {
        "logistica": f"Regressao Logistica - Etapa {etapa}",
        "lr": f"Regressao Logistica - Etapa {etapa}",
        "gradient_boosting": f"Gradient Boosting - Etapa {etapa}",
        "gb": f"Gradient Boosting - Etapa {etapa}",
        "gbm": f"Gradient Boosting - Etapa {etapa}",
        "hist_gradient_boosting": f"Hist Gradient Boosting - Etapa {etapa}",
        "hgb": f"Hist Gradient Boosting - Etapa {etapa}",
        "hist_gb": f"Hist Gradient Boosting - Etapa {etapa}",
    }
    nome_default = nomes_padrao.get(tipo)
    if nome_default is None:
        raise ValueError(f"Tipo de modelo '{tipo_modelo}' nao reconhecido")

    params_com_nome = {"nome": nome_default, **params_modelo}
    return criar_modelo_factory(tipo=tipo, **params_com_nome)


def treinar_e_prever_oot(
    modelo: ModeloBase,
    x_treino: pd.DataFrame,
    y_treino: pd.Series,
    x_oot: pd.DataFrame,
) -> np.ndarray:
    """
    Treina o modelo com treino e retorna probabilidades no OOT.

    Parameters
    ----------
    modelo : ModeloBase
        Modelo com interface `fit` e `predict_proba`.
    x_treino : pd.DataFrame
        Features de treino.
    y_treino : pd.Series
        Target de treino.
    x_oot : pd.DataFrame
        Features OOT para inferencia.

    Returns
    -------
    np.ndarray
        Probabilidades previstas para o conjunto OOT.
    """
    # Mantem treino e inferencia no mesmo helper para padronizar o fluxo.
    modelo.fit(x_treino, y_treino)
    return modelo.predict_proba(x_oot)
