"""
Funcoes de avaliacao para resultados do treinamento incremental.
"""

from typing import Dict, List, Optional, cast

import numpy as np

from ..config import obter_nome_etapa
from ..metricas import avaliar_modelo_completo
from .types import ResultadoEtapa


def avaliar_resultado_etapa(
    etapa: int,
    nome_modelo: str,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
    features: List[str],
    flag_instalacao: Optional[np.ndarray] = None,
) -> ResultadoEtapa:
    """
    Avalia uma etapa e enriquece o resultado com metadados do fluxo.

    Parameters
    ----------
    etapa : int
        Numero da etapa incremental.
    nome_modelo : str
        Nome base do modelo avaliado.
    y_true : np.ndarray
        Valores reais da variavel alvo no OOT.
    y_pred_proba : np.ndarray
        Probabilidades previstas para classe positiva.
    threshold : float
        Threshold para classificacao binaria.
    features : List[str]
        Features efetivamente usadas na etapa.
    flag_instalacao : Optional[np.ndarray], optional
        Vetor opcional para analise swap-in/swap-out.

    Returns
    -------
    ResultadoEtapa
        Dicionario tipado com metricas e metadados da etapa.
    """
    resultados: Dict = avaliar_modelo_completo(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        threshold=threshold,
        flag_instalacao=flag_instalacao,
        nome_modelo=f"{nome_modelo} - Etapa {etapa}",
    )

    # Complementa metricas com metadados padrao para relatorio incremental.
    resultados["etapa"] = etapa
    resultados["nome_etapa"] = obter_nome_etapa(etapa)
    resultados["num_features"] = len(features)
    resultados["features"] = features
    return cast(ResultadoEtapa, resultados)
