import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier

from .base import ModeloBase


class GradientBoosting(ModeloBase):
    """
    Wrapper para Gradient Boosting Classifier.
    """

    def __init__(
        self,
        nome: str = "Gradient Boosting",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        min_samples_split: int = 100,
        min_samples_leaf: int = 50,
        subsample: float = 0.8,
        random_state: int = 42,
        verbose: int = 0,
    ):
        super().__init__(nome)

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.verbose = verbose

        logger.info(f"Inicializando {self.nome}")
        logger.info(
            f"  Parametros: n_estimators={n_estimators}, learning_rate={learning_rate}"
        )
        logger.info(f"  max_depth={max_depth}, subsample={subsample}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoosting":
        logger.info(f"Treinando {self.nome}...")
        logger.info(f"  Features: {X.shape[1]}, Amostras: {X.shape[0]}")

        self.features_treino = X.columns.tolist()
        self.modelo = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # Mantido sem imputacao explicita: comportamento pedido para GB.
        self.modelo.fit(X, y)
        self.is_fitted = True
        logger.success(f"{self.nome} treinado com sucesso!")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Modelo nao foi treinado. Execute fit() primeiro.")

        X_pred = X[self.features_treino]
        return self.modelo.predict_proba(X_pred)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Modelo nao foi treinado. Execute fit() primeiro.")

        importances = self.modelo.feature_importances_
        df_importance = pd.DataFrame({"feature": self.features_treino, "importance": importances})
        return df_importance.sort_values("importance", ascending=False)

