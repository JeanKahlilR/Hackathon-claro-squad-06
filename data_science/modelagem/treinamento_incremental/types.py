"""
Tipos compartilhados do fluxo de treinamento incremental.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, TypedDict, Union
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame


class ResultadoEtapa(TypedDict, total=False):
    """
    Estrutura padrao para resultados de avaliacao por etapa.

    Notes
    -----
    `total=False` permite campos opcionais, pois algumas metricas podem
    nao estar disponiveis em todos os cenarios.
    """

    etapa: int
    nome_etapa: str
    num_features: int
    features: List[str]
    ks: float
    delta_ks: float
    ks_anterior: float
    gini: float
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    modelo: str
    y_true_oot: np.ndarray
    y_pred_proba_oot: np.ndarray


TablesSplited = Dict[str, Dict[str, Union[SparkDataFrame, pd.DataFrame]]]
"""Alias para tabelas (Spark ou pandas) separadas em treino/OOT por fonte."""


@dataclass
class EstadoIncremental:
    """
    Estado mutavel compartilhado durante a execucao incremental.

    Attributes
    ----------
    resultados : Dict[str, Dict[str, Any]]
        Resultados consolidados por modelo/etapa.
    modelos_treinados : Dict[str, Any]
        Modelos treinados indexados por chave de etapa.
    graficos_ks : Dict[str, Dict[int, str]]
        Caminhos dos graficos KS por tipo de modelo e etapa.
    """

    resultados: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    modelos_treinados: Dict[str, Any] = field(default_factory=dict)
    graficos_ks: Dict[str, Dict[int, str]] = field(default_factory=dict)
