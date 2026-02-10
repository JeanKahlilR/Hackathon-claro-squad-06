from typing import Any

from .base import ModeloBase
from .gradient_boosting import GradientBoosting
from .hist_gradient_boosting import HistGradientBoosting
from .regressao_logistica import RegressaoLogistica

REGISTRY_MODELOS = {
    "logistica": RegressaoLogistica,
    "lr": RegressaoLogistica,
    "logistic": RegressaoLogistica,
    "gradient_boosting": GradientBoosting,
    "gb": GradientBoosting,
    "gbm": GradientBoosting,
    "hist_gradient_boosting": HistGradientBoosting,
    "hgb": HistGradientBoosting,
    "hist_gb": HistGradientBoosting,
}


def criar_modelo_factory(tipo: str, **kwargs: Any) -> ModeloBase:
    """
    Factory function para criar modelos.
    """
    tipo_normalizado = tipo.lower()
    classe_modelo = REGISTRY_MODELOS.get(tipo_normalizado)
    if classe_modelo is None:
        raise ValueError(
            f"Tipo de modelo '{tipo}' nao reconhecido. "
            "Use 'logistica', 'gradient_boosting' ou 'hist_gradient_boosting'"
        )
    return classe_modelo(**kwargs)

