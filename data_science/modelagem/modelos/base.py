from typing import Optional

import numpy as np
import pandas as pd


class ModeloBase:
    """
    Classe base para modelos de ML.
    """

    def __init__(self, nome: str = "Modelo"):
        self.nome = nome
        self.modelo = None
        self.scaler = None
        self.features_treino = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModeloBase":
        raise NotImplementedError("Metodo fit deve ser implementado nas subclasses")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Metodo predict_proba deve ser implementado nas subclasses")

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return None

