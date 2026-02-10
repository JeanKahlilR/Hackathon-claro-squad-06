from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier

from .base import ModeloBase


class HistGradientBoosting(ModeloBase):
    """
    Wrapper para HistGradientBoostingClassifier.
    """

    def __init__(
        self,
        nome: str = "Hist Gradient Boosting",
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_depth: Optional[int] = None,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        max_bins: int = 255,
        l2_regularization: float = 0.0,
        early_stopping: str | bool = "auto",
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        random_state: int = 42,
        verbose: int = 0,
    ):
        super().__init__(nome)

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.l2_regularization = l2_regularization
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.verbose = verbose

        logger.info(f"Inicializando {self.nome}")
        logger.info(f"  Parametros: max_iter={max_iter}, learning_rate={learning_rate}")
        logger.info(
            f"  max_depth={max_depth}, max_leaf_nodes={max_leaf_nodes}, "
            f"early_stopping={early_stopping}"
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistGradientBoosting":
        logger.info(f"Treinando {self.nome}...")
        logger.info(f"  Features: {X.shape[1]}, Amostras: {X.shape[0]}")

        self.features_treino = X.columns.tolist()
        self.modelo = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            max_bins=self.max_bins,
            l2_regularization=self.l2_regularization,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # HistGradientBoostingClassifier aceita NaN nativamente no sklearn.
        self.modelo.fit(X, y)
        self.is_fitted = True
        logger.success(f"{self.nome} treinado com sucesso!")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Modelo nao foi treinado. Execute fit() primeiro.")

        X_pred = X[self.features_treino]
        return self.modelo.predict_proba(X_pred)[:, 1]

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if not self.is_fitted:
            raise ValueError("Modelo nao foi treinado. Execute fit() primeiro.")

        if hasattr(self.modelo, "feature_importances_"):
            importances = self.modelo.feature_importances_
            df_importance = pd.DataFrame(
                {
                    "feature": self.features_treino,
                    "importance": importances,
                }
            )
            return df_importance.sort_values("importance", ascending=False)

        logger.warning(
            "HistGradientBoostingClassifier nao disponibiliza "
            "feature_importances_ nativamente; retornando None."
        )
        return None
