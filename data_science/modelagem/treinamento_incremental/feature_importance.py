"""
Funcoes utilitarias para calcular e salvar importance de variaveis por etapa.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _obter_proba_classe_positiva(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    """
    Extrai probabilidade da classe positiva em formato 1D.
    """
    y_pred_proba = np.asarray(estimator.predict_proba(x), dtype=float)
    if y_pred_proba.ndim == 1:
        return y_pred_proba
    if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] >= 2:
        return y_pred_proba[:, 1]
    raise ValueError(
        "predict_proba retornou formato inesperado para classificacao binaria: "
        f"shape={y_pred_proba.shape}"
    )


def _resolver_scoring(
    scoring: Union[str, Callable[[Any, pd.DataFrame, Union[pd.Series, np.ndarray]], float]],
) -> Callable[[Any, pd.DataFrame, Union[pd.Series, np.ndarray]], float]:
    """
    Resolve scoring para callable compativel com wrappers de modelo.
    """
    if callable(scoring):
        logger.info("Scoring de permutation importance recebido como callable customizado.")
        return scoring

    scoring_normalizado = scoring.strip().lower()
    logger.info(f"Scoring de permutation importance resolvido para: {scoring_normalizado}")
    if scoring_normalizado == "roc_auc":
        return lambda estimator, x, y: roc_auc_score(y, _obter_proba_classe_positiva(estimator, x))
    if scoring_normalizado == "accuracy":
        return lambda estimator, x, y: accuracy_score(y, estimator.predict(x))
    if scoring_normalizado == "precision":
        return lambda estimator, x, y: precision_score(y, estimator.predict(x), zero_division=0)
    if scoring_normalizado == "recall":
        return lambda estimator, x, y: recall_score(y, estimator.predict(x), zero_division=0)
    if scoring_normalizado == "f1":
        return lambda estimator, x, y: f1_score(y, estimator.predict(x), zero_division=0)

    raise ValueError(
        f"Scoring '{scoring}' nao suportado. "
        "Use: roc_auc, accuracy, precision, recall, f1 ou passe um callable."
    )


def calcular_permutation_importance(
    estimator: Any,
    x: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    scoring: Union[str, Callable[[Any, pd.DataFrame, Union[pd.Series, np.ndarray]], float]] = "roc_auc",
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Calcula permutation importance para um estimador em um conjunto de avaliacao.

    Parameters
    ----------
    estimator : Any
        Estimador treinado com API compativel com scikit-learn.
    x : pd.DataFrame
        Features de avaliacao.
    y : Union[pd.Series, np.ndarray]
        Target real de avaliacao.
    scoring : str, optional
        Metrica usada no calculo da importance.
    n_repeats : int, optional
        Numero de embaralhamentos por variavel.
    random_state : int, optional
        Semente para reproducibilidade.
    n_jobs : int, optional
        Numero de jobs paralelos no scikit-learn.

    Returns
    -------
    pd.DataFrame
        DataFrame com importance media e desvio padrao por variavel.
    """
    if x.empty:
        raise ValueError("x nao pode ser vazio para calcular permutation importance.")

    logger.info(
        "Iniciando permutation importance | amostras={} | features={} | n_repeats={}",
        x.shape[0],
        x.shape[1],
        n_repeats,
    )
    scoring_resolvido = _resolver_scoring(scoring)
    resultado = permutation_importance(
        estimator=estimator,
        X=x,
        y=y,
        scoring=scoring_resolvido,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=None,
        max_samples=50000,
    )

    df_importancia = pd.DataFrame(
        {
            "feature": x.columns,
            "importance_mean": resultado.importances_mean,
            "importance_std": resultado.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    df_importancia = df_importancia.reset_index(drop=True)
    df_importancia["ranking"] = np.arange(1, len(df_importancia) + 1)
    if not df_importancia.empty:
        top_feature = df_importancia.iloc[0]["feature"]
        top_importance = float(df_importancia.iloc[0]["importance_mean"])
        logger.info(
            "Permutation importance finalizada | top_1_feature={} | top_1_importance={:.6f}",
            top_feature,
            top_importance,
        )
    return df_importancia[["ranking", "feature", "importance_mean", "importance_std"]]


def _extrair_estimador_xgboost(modelo: Any) -> tuple[Any, Optional[List[str]], str]:
    """
    Extrai XGBClassifier e metadados a partir de wrapper ou estimador direto.
    """
    if hasattr(modelo, "modelo") and getattr(modelo, "modelo") is not None:
        estimador = getattr(modelo, "modelo")
        features_treino = getattr(modelo, "features_treino", None)
        importance_type = getattr(modelo, "importance_type", "gain")
    else:
        estimador = modelo
        features_treino = getattr(modelo, "feature_names_in_", None)
        importance_type = getattr(modelo, "importance_type", "gain")

    if not hasattr(estimador, "get_booster"):
        raise ValueError(
            "Modelo informado nao parece ser XGBoost. "
            "Esperado objeto com metodo get_booster()."
        )

    return estimador, features_treino, str(importance_type)


def _mapear_score_para_features(
    score_dict: dict[str, float],
    features_ordenadas: List[str],
) -> dict[str, float]:
    """
    Mapeia chaves do booster para nomes de features.
    """
    score_por_feature = {feature: 0.0 for feature in features_ordenadas}

    for chave_feature, valor in score_dict.items():
        if chave_feature in score_por_feature:
            score_por_feature[chave_feature] = float(valor)
            continue

        if chave_feature.startswith("f") and chave_feature[1:].isdigit():
            indice = int(chave_feature[1:])
            if 0 <= indice < len(features_ordenadas):
                score_por_feature[features_ordenadas[indice]] = float(valor)

    return score_por_feature


def calcular_feature_importance_xgboost(
    modelo: Any,
    importance_type: Optional[str] = None,
    incluir_zeros: bool = True,
) -> pd.DataFrame:
    """
    Calcula importance nativa do XGBoost com schema proprio.

    Colunas de saida:
    - ranking
    - feature
    - importance
    - importance_type
    - metodo_importancia
    """
    estimador, features_treino, importance_type_modelo = _extrair_estimador_xgboost(modelo)
    importance_type_final = str((importance_type or importance_type_modelo or "gain")).strip()

    booster = estimador.get_booster()
    score_dict = booster.get_score(importance_type=importance_type_final)

    if features_treino is None:
        booster_features = booster.feature_names
        if booster_features:
            features_treino = list(booster_features)
        else:
            features_treino = sorted(score_dict.keys())

    score_por_feature = _mapear_score_para_features(score_dict, list(features_treino))
    if incluir_zeros:
        features_saida = list(features_treino)
    else:
        features_saida = [f for f in features_treino if score_por_feature.get(f, 0.0) > 0.0]

    if not features_saida:
        return pd.DataFrame(
            columns=["ranking", "feature", "importance", "importance_type", "metodo_importancia"]
        )

    df_importancia = pd.DataFrame(
        {
            "feature": features_saida,
            "importance": [float(score_por_feature.get(f, 0.0)) for f in features_saida],
            "importance_type": importance_type_final,
            "metodo_importancia": "xgboost_native",
        }
    ).sort_values("importance", ascending=False)

    df_importancia = df_importancia.reset_index(drop=True)
    df_importancia["ranking"] = np.arange(1, len(df_importancia) + 1)
    return df_importancia[
        ["ranking", "feature", "importance", "importance_type", "metodo_importancia"]
    ]


def salvar_importance_csv(df_importancia: pd.DataFrame, caminho_saida: str) -> str:
    """
    Salva importance em CSV.

    Parameters
    ----------
    df_importancia : pd.DataFrame
        DataFrame de importance.
    caminho_saida : str
        Caminho do arquivo de saida.

    Returns
    -------
    str
        Caminho final do arquivo salvo.
    """
    path = Path(caminho_saida)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_importancia.to_csv(path, index=False)
    logger.info(f"CSV de importance salvo em: {path}")
    return str(path)


def salvar_importance_plot(
    df_importancia: pd.DataFrame,
    caminho_saida: str,
    titulo: str,
    top_n: int = 20,
) -> str:
    """
    Salva grafico horizontal com Top N variaveis mais importantes.

    Parameters
    ----------
    df_importancia : pd.DataFrame
        DataFrame de importance.
    caminho_saida : str
        Caminho da imagem de saida.
    titulo : str
        Titulo do grafico.
    top_n : int, optional
        Quantidade de variaveis no grafico.

    Returns
    -------
    str
        Caminho final do arquivo salvo.
    """
    if df_importancia.empty:
        raise ValueError("df_importancia nao pode ser vazio para gerar grafico.")

    path = Path(caminho_saida)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_plot = (
        df_importancia.head(top_n)
        .sort_values("importance_mean", ascending=True)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        df_plot["feature"],
        df_plot["importance_mean"],
        xerr=df_plot["importance_std"],
    )
    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.set_xlabel("Permutation Importance", fontweight="bold")
    ax.set_ylabel("Variavel", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Grafico de importance salvo em: {path}")

    return str(path)


def salvar_importance_xgboost_csv(df_importancia: pd.DataFrame, caminho_saida: str) -> str:
    """
    Salva CSV de importance nativa do XGBoost.
    """
    colunas_esperadas = {
        "ranking",
        "feature",
        "importance",
        "importance_type",
        "metodo_importancia",
    }
    colunas_df = set(df_importancia.columns)
    if not colunas_esperadas.issubset(colunas_df):
        raise ValueError(
            "Schema invalido para importance nativa do XGBoost. "
            f"Esperado {sorted(colunas_esperadas)}, recebido {sorted(colunas_df)}"
        )

    path = Path(caminho_saida)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_importancia.to_csv(path, index=False)
    logger.info(f"CSV de importance nativa do XGBoost salvo em: {path}")
    return str(path)


def salvar_importance_xgboost_plot(
    df_importancia: pd.DataFrame,
    caminho_saida: str,
    titulo: str,
    top_n: int = 20,
) -> str:
    """
    Salva grafico horizontal com Top N da importance nativa do XGBoost.
    """
    if df_importancia.empty:
        raise ValueError("df_importancia nao pode ser vazio para gerar grafico.")

    path = Path(caminho_saida)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_plot = (
        df_importancia.head(top_n)
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )
    importance_type = str(df_plot.iloc[0]["importance_type"])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df_plot["feature"], df_plot["importance"])
    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"XGBoost Importance ({importance_type})", fontweight="bold")
    ax.set_ylabel("Variavel", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Grafico de importance nativa do XGBoost salvo em: {path}")

    return str(path)
