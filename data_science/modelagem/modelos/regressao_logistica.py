from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import ModeloBase


class RegressaoLogistica(ModeloBase):
    """
    Wrapper para Regressao Logistica com pre-processamento.
    """

    def __init__(
        self,
        nome: str = "Regressao Logistica",
        max_iter: int = 1000,
        random_state: int = 42,
        class_weight: Optional[str] = "balanced",
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        normalizar: bool = True,
    ):
        super().__init__(nome)

        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.normalizar = normalizar
        self.imputer = None

        logger.info(f"Inicializando {self.nome}")
        logger.info(f"  Parametros: C={C}, penalty={penalty}, solver={solver}")
        logger.info(f"  Class weight: {class_weight}, Normalizar: {normalizar}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegressaoLogistica":
        logger.info(f"Treinando {self.nome}...")
        logger.info(f"  Features: {X.shape[1]}, Amostras: {X.shape[0]}")

        self.features_treino = X.columns.tolist()

        # Regressao Logistica nao aceita NaN no sklearn:
        # a imputacao fica restrita a este modelo.
        self.imputer = SimpleImputer(strategy="median")
        X_imputed = self.imputer.fit_transform(X)
        X_imputed = np.nan_to_num(X_imputed, nan=0.0)

        if self.normalizar:
            logger.info("  Aplicando normalizacao (StandardScaler)...")
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_scaled = X_imputed

        self.modelo = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight,
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
        )

        self.modelo.fit(X_scaled, y)
        self.is_fitted = True
        logger.success(f"{self.nome} treinado com sucesso!")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Modelo nao foi treinado. Execute fit() primeiro.")

        X_pred = X[self.features_treino]

        if self.imputer is None:
            raise ValueError("Imputer nao inicializado. Execute fit() primeiro.")
        X_pred_imputed = self.imputer.transform(X_pred)
        X_pred_imputed = np.nan_to_num(X_pred_imputed, nan=0.0)

        if self.normalizar and self.scaler is not None:
            X_scaled = self.scaler.transform(X_pred_imputed)
        else:
            X_scaled = X_pred_imputed

        return self.modelo.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Modelo nao foi treinado. Execute fit() primeiro.")

        coeficientes = self.modelo.coef_[0]
        df_importance = pd.DataFrame(
            {
                "feature": self.features_treino,
                "coeficiente": coeficientes,
                "abs_coeficiente": np.abs(coeficientes),
            }
        )
        return df_importance.sort_values("abs_coeficiente", ascending=False)
