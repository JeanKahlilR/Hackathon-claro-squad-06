from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from .base import ModeloBase

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - depende do ambiente.
    XGBClassifier = None  # type: ignore[assignment]


class XGBoost(ModeloBase):
    """
    Wrapper para XGBClassifier.
    """

    def __init__(
        self,
        nome: str = "XGBoost",
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        objective: str = "binary:logistic",
        eval_metric: str = "auc",
        importance_type: str = "gain",
        tree_method: str = "hist",
        n_jobs: int = -1,
        random_state: int = 42,
        verbosity: int = 0,
    ):
        super().__init__(nome)

        if XGBClassifier is None:
            raise ImportError(
                "xgboost nao esta instalado. "
                "Instale a dependencia 'xgboost' para usar este modelo."
            )

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.eval_metric = eval_metric
        self.importance_type = importance_type
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbosity = verbosity

        logger.info(f"Inicializando {self.nome}")
        logger.info(
            f"  Parametros: n_estimators={n_estimators}, learning_rate={learning_rate}"
        )
        logger.info(
            f"  max_depth={max_depth}, subsample={subsample}, "
            f"colsample_bytree={colsample_bytree}"
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoost":
        logger.info(f"Treinando {self.nome}...")
        logger.info(f"  Features: {X.shape[1]}, Amostras: {X.shape[0]}")

        self.features_treino = X.columns.tolist()
        self.modelo = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective=self.objective,
            eval_metric=self.eval_metric,
            importance_type=self.importance_type,
            tree_method=self.tree_method,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbosity=self.verbosity,
        )

        # XGBoost aceita NaN nativamente.
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

        logger.warning("XGBClassifier nao disponibiliza feature_importances_; retornando None.")
        return None
