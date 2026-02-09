"""
Modulo para calculo de metricas de avaliacao de modelos:
KS, Gini, AUC-ROC, Matriz de Confusao, Swap-in/Swap-out.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import confusion_matrix, roc_auc_score

from .config import BENCHMARK_KS


def _validar_binario_sem_nan(
    nome: str,
    valores: np.ndarray,
    tamanho_esperado: Optional[int] = None,
) -> np.ndarray:
    """
    Converte vetor para numerico e valida dominio binario {0, 1} sem NaN.
    """
    serie = pd.to_numeric(pd.Series(valores), errors="coerce")

    if tamanho_esperado is not None and len(serie) != tamanho_esperado:
        raise ValueError(
            f"'{nome}' possui tamanho {len(serie)}, mas o esperado e {tamanho_esperado}."
        )

    nan_mask = serie.isna()
    if nan_mask.any():
        raise ValueError(f"'{nome}' contem {int(nan_mask.sum())} valor(es) invalido(s)/NaN.")

    unicos = np.unique(serie.to_numpy())
    if not np.isin(unicos, [0, 1]).all():
        raise ValueError(
            f"'{nome}' deve conter apenas valores binarios {{0, 1}}. "
            f"Valores encontrados: {sorted(unicos.tolist())[:10]}"
        )

    return serie.to_numpy(dtype=int)


def _validar_probabilidades(
    nome: str,
    valores: np.ndarray,
    tamanho_esperado: Optional[int] = None,
) -> np.ndarray:
    """
    Converte vetor para float e valida ausencia de NaN.
    """
    serie = pd.to_numeric(pd.Series(valores), errors="coerce")

    if tamanho_esperado is not None and len(serie) != tamanho_esperado:
        raise ValueError(
            f"'{nome}' possui tamanho {len(serie)}, mas o esperado e {tamanho_esperado}."
        )

    nan_mask = serie.isna()
    if nan_mask.any():
        raise ValueError(f"'{nome}' contem {int(nan_mask.sum())} valor(es) invalido(s)/NaN.")

    return serie.to_numpy(dtype=float)


def _encontrar_threshold_max_ks_com_arrays_validados(
    y_true_int: np.ndarray,
    y_pred_proba_float: np.ndarray,
) -> tuple[float, float]:
    """
    Encontra o threshold que maximiza KS assumindo entradas validadas.

    Retorna
    -------
    tuple[float, float]
        (threshold_otimo, ks_no_threshold_otimo_em_percentual)
    """
    df = pd.DataFrame(
        {
            "y_true": y_true_int,
            "y_pred_proba": y_pred_proba_float,
        }
    )

    df["bad"] = df["y_true"]
    df["good"] = 1 - df["y_true"]

    total_bad = int(df["bad"].sum())
    total_good = int(df["good"].sum())

    if total_bad == 0 or total_good == 0:
        logger.warning("Threshold otimo por KS nao pode ser calculado: apenas uma classe presente")
        return 0.5, 0.0

    # Agrupar por score evita distorcao por empates no ponto de corte.
    curva_ks = (
        df.groupby("y_pred_proba", as_index=False)[["bad", "good"]]
        .sum()
        .sort_values("y_pred_proba", ascending=False)
        .reset_index(drop=True)
    )
    curva_ks["cum_bad_rate"] = curva_ks["bad"].cumsum() / total_bad
    curva_ks["cum_good_rate"] = curva_ks["good"].cumsum() / total_good
    curva_ks["ks"] = abs(curva_ks["cum_bad_rate"] - curva_ks["cum_good_rate"])

    idx_melhor = int(curva_ks["ks"].idxmax())
    melhor_threshold = float(curva_ks.loc[idx_melhor, "y_pred_proba"])
    melhor_ks = float(curva_ks.loc[idx_melhor, "ks"] * 100)
    return melhor_threshold, melhor_ks


def _calcular_ks_com_arrays_validados(
    y_true_int: np.ndarray,
    y_pred_proba_float: np.ndarray,
) -> float:
    """
    Implementacao do KS assumindo entradas ja validadas.
    """
    df = pd.DataFrame(
        {
            "y_true": y_true_int,
            "y_pred_proba": y_pred_proba_float,
        }
    )

    # Ordenar por probabilidade decrescente.
    df = df.sort_values("y_pred_proba", ascending=False).reset_index(drop=True)

    # Calcular proporcoes acumuladas de maus e bons.
    df["bad"] = df["y_true"]
    df["good"] = 1 - df["y_true"]

    total_bad = df["bad"].sum()
    total_good = df["good"].sum()

    if total_bad == 0 or total_good == 0:
        logger.warning("KS nao pode ser calculado: apenas uma classe presente")
        return 0.0

    df["cum_bad_rate"] = df["bad"].cumsum() / total_bad
    df["cum_good_rate"] = df["good"].cumsum() / total_good

    # KS e a maxima diferenca entre as curvas acumuladas.
    df["ks"] = abs(df["cum_bad_rate"] - df["cum_good_rate"])
    return float(df["ks"].max() * 100)


def calcular_ks(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calcula a estatistica KS (Kolmogorov-Smirnov).

    Parameters
    ----------
    y_true : np.ndarray
        Valores reais do target (0 ou 1)
    y_pred_proba : np.ndarray
        Probabilidades preditas pelo modelo

    Returns
    -------
    float
        Valor do KS (0 a 100)

    Examples
    --------
    >>> ks = calcular_ks(y_true, y_pred_proba)
    >>> print(f"KS: {ks:.2f}")
    KS: 35.42
    """
    y_true_int = _validar_binario_sem_nan("y_true", y_true)
    y_pred_proba_float = _validar_probabilidades(
        "y_pred_proba",
        y_pred_proba,
        tamanho_esperado=len(y_true_int),
    )
    return _calcular_ks_com_arrays_validados(y_true_int, y_pred_proba_float)


def calcular_gini(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calcula o coeficiente de Gini.

    Parameters
    ----------
    y_true : np.ndarray
        Valores reais do target
    y_pred_proba : np.ndarray
        Probabilidades preditas

    Returns
    -------
    float
        Coeficiente de Gini (0 a 100)

    Examples
    --------
    >>> gini = calcular_gini(y_true, y_pred_proba)
    >>> print(f"Gini: {gini:.2f}")
    Gini: 45.30
    """
    auc = roc_auc_score(y_true, y_pred_proba)
    gini = (2 * auc - 1) * 100
    return float(gini)


def calcular_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calcula a area sob a curva ROC (AUC-ROC).

    Parameters
    ----------
    y_true : np.ndarray
        Valores reais do target
    y_pred_proba : np.ndarray
        Probabilidades preditas

    Returns
    -------
    float
        AUC-ROC (0 a 1)

    Examples
    --------
    >>> auc = calcular_auc(y_true, y_pred_proba)
    >>> print(f"AUC: {auc:.4f}")
    AUC: 0.7265
    """
    return float(roc_auc_score(y_true, y_pred_proba))


def calcular_matriz_confusao(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None,
) -> np.ndarray:
    """
    Calcula a matriz de confusao.

    Parameters
    ----------
    y_true : np.ndarray
        Valores reais do target
    y_pred : np.ndarray
        Predicoes binarias (0 ou 1)
    labels : Optional[list], optional
        Labels das classes (default: [0, 1])

    Returns
    -------
    np.ndarray
        Matriz de confusao 2x2

    Examples
    --------
    >>> cm = calcular_matriz_confusao(y_true, y_pred)
    >>> print(cm)
    [[TN FP]
     [FN TP]]
    """
    if labels is None:
        labels = [0, 1]

    y_true_int = _validar_binario_sem_nan("y_true", y_true)
    y_pred_int = _validar_binario_sem_nan(
        "y_pred",
        y_pred,
        tamanho_esperado=len(y_true_int),
    )

    return confusion_matrix(y_true_int, y_pred_int, labels=labels)


def calcular_metricas_confusao(cm: np.ndarray) -> Dict[str, float]:
    """
    Calcula metricas derivadas da matriz de confusao.

    Parameters
    ----------
    cm : np.ndarray
        Matriz de confusao 2x2

    Returns
    -------
    Dict[str, float]
        Dicionario com metricas: accuracy, precision, recall, f1, specificity

    Examples
    --------
    >>> metricas = calcular_metricas_confusao(cm)
    >>> print(f"Accuracy: {metricas['accuracy']:.4f}")
    Accuracy: 0.8542
    """
    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp

    metricas = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "accuracy": (tp + tn) / total if total > 0 else 0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
    }

    # F1-Score
    if metricas["precision"] + metricas["recall"] > 0:
        metricas["f1_score"] = (
            2 * (metricas["precision"] * metricas["recall"])
            / (metricas["precision"] + metricas["recall"])
        )
    else:
        metricas["f1_score"] = 0

    return metricas


def calcular_swap_in_swap_out(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    flag_instalacao: np.ndarray,
) -> Dict[str, int]:
    """
    Calcula analise de Swap-in e Swap-out comparando modelo vs politica vigente.

    Swap-in: Clientes que seriam reprovados pela politica (FLAG_INSTALACAO=0)
             mas sao aprovados pelo modelo (y_pred=0, bom pagador)

    Swap-out: Clientes que seriam aprovados pela politica (FLAG_INSTALACAO=1)
              mas sao reprovados pelo modelo (y_pred=1, mau pagador)

    Parameters
    ----------
    y_true : np.ndarray
        Valores reais do target (FPD)
    y_pred : np.ndarray
        Predicoes do modelo (0=aprovar, 1=reprovar)
    flag_instalacao : np.ndarray
        Flag da politica vigente (0=reprovado, 1=aprovado)

    Returns
    -------
    Dict[str, int]
        Dicionario com contagens de swap-in, swap-out e analises

    Examples
    --------
    >>> swap = calcular_swap_in_swap_out(y_true, y_pred, flag_instalacao)
    >>> print(f"Swap-in: {swap['swap_in_total']}")
    Swap-in: 15420
    """
    y_true_int = _validar_binario_sem_nan("y_true", y_true)
    y_pred_int = _validar_binario_sem_nan("y_pred", y_pred, tamanho_esperado=len(y_true_int))
    flag_instalacao_int = _validar_binario_sem_nan(
        "flag_instalacao",
        flag_instalacao,
        tamanho_esperado=len(y_true_int),
    )

    df = pd.DataFrame(
        {
            "y_true": y_true_int,
            "y_pred": y_pred_int,
            "flag_instalacao": flag_instalacao_int,
        }
    )

    # Swap-in: Reprovados pela politica (0) mas aprovados pelo modelo (0)
    # Modelo prediz 0 (nao inadimplente) = aprovar.
    swap_in = df[(df["flag_instalacao"] == 0) & (df["y_pred"] == 0)]
    swap_in_bons = swap_in[swap_in["y_true"] == 0]
    swap_in_maus = swap_in[swap_in["y_true"] == 1]

    # Swap-out: Aprovados pela politica (1) mas reprovados pelo modelo (1)
    # Modelo prediz 1 (inadimplente) = reprovar.
    swap_out = df[(df["flag_instalacao"] == 1) & (df["y_pred"] == 1)]
    swap_out_bons = swap_out[swap_out["y_true"] == 0]
    swap_out_maus = swap_out[swap_out["y_true"] == 1]

    # Mantidos
    mantidos_aprovados = df[(df["flag_instalacao"] == 1) & (df["y_pred"] == 0)]
    mantidos_reprovados = df[(df["flag_instalacao"] == 0) & (df["y_pred"] == 1)]

    resultado = {
        # Swap-in
        "swap_in_total": len(swap_in),
        "swap_in_bons": len(swap_in_bons),
        "swap_in_maus": len(swap_in_maus),
        "swap_in_taxa_acerto": len(swap_in_bons) / len(swap_in) * 100 if len(swap_in) > 0 else 0,
        # Swap-out
        "swap_out_total": len(swap_out),
        "swap_out_bons": len(swap_out_bons),
        "swap_out_maus": len(swap_out_maus),
        "swap_out_taxa_acerto": len(swap_out_maus) / len(swap_out) * 100 if len(swap_out) > 0 else 0,
        # Mantidos
        "mantidos_aprovados": len(mantidos_aprovados),
        "mantidos_reprovados": len(mantidos_reprovados),
        # Totais
        "total_registros": len(df),
        "total_aprovados_politica": len(df[df["flag_instalacao"] == 1]),
        "total_reprovados_politica": len(df[df["flag_instalacao"] == 0]),
        "total_aprovados_modelo": len(df[df["y_pred"] == 0]),
        "total_reprovados_modelo": len(df[df["y_pred"] == 1]),
    }

    return resultado


def avaliar_modelo_completo(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    usar_threshold_otimo_ks: bool = True,
    flag_instalacao: Optional[np.ndarray] = None,
    nome_modelo: str = "Modelo",
) -> Dict:
    """
    Avaliacao completa do modelo com todas as metricas.

    Parameters
    ----------
    y_true : np.ndarray
        Valores reais do target (classe positiva = 1 = mau pagador)
    y_pred_proba : np.ndarray
        Probabilidades preditas para a classe positiva (mau pagador)
    threshold : float, optional
        Threshold para classificacao binaria (default: 0.5)
    usar_threshold_otimo_ks : bool, optional
        Se True, usa o threshold que maximiza KS para gerar predicoes binarias.
        Se False, usa o threshold informado no parametro `threshold`.
    flag_instalacao : Optional[np.ndarray], optional
        Flag da politica vigente para analise swap-in/out
    nome_modelo : str, optional
        Nome do modelo para logging

    Returns
    -------
    Dict
        Dicionario com todas as metricas calculadas

    Examples
    --------
    >>> metricas = avaliar_modelo_completo(y_true, y_pred_proba, flag_instalacao=flag_inst)
    """
    logger.info("=" * 70)
    logger.info(f"AVALIACAO DO MODELO: {nome_modelo}")
    logger.info("=" * 70)

    y_true_int = _validar_binario_sem_nan("y_true", y_true)
    y_pred_proba_float = _validar_probabilidades(
        "y_pred_proba",
        y_pred_proba,
        tamanho_esperado=len(y_true_int),
    )

    flag_instalacao_int = None
    if flag_instalacao is not None:
        flag_instalacao_int = _validar_binario_sem_nan(
            "flag_instalacao",
            flag_instalacao,
            tamanho_esperado=len(y_true_int),
        )

    if threshold < 0 or threshold > 1:
        raise ValueError(f"'threshold' deve estar no intervalo [0, 1]. Valor recebido: {threshold}")

    resultados = {
        "nome_modelo": nome_modelo,
        "threshold_informado": float(threshold),
        "usar_threshold_otimo_ks": usar_threshold_otimo_ks,
    }

    # 1. KS e threshold otimo pelo proprio KS
    threshold_otimo_ks, ks_threshold_otimo = _encontrar_threshold_max_ks_com_arrays_validados(
        y_true_int,
        y_pred_proba_float,
    )
    ks = ks_threshold_otimo
    resultados["ks"] = ks
    logger.info(f"KS: {ks:.2f}")
    logger.info(f"Benchmark KS: {BENCHMARK_KS:.2f}")
    logger.info(f"Delta vs Benchmark: {ks - BENCHMARK_KS:+.2f}")

    threshold_utilizado = threshold_otimo_ks if usar_threshold_otimo_ks else float(threshold)

    resultados["threshold_otimo_ks"] = threshold_otimo_ks
    resultados["ks_threshold_otimo"] = ks_threshold_otimo
    resultados["threshold"] = threshold_utilizado

    logger.info(f"Threshold informado: {float(threshold):.6f}")
    logger.info(
        f"Threshold otimo (max KS): {threshold_otimo_ks:.6f} | KS no cutoff: {ks_threshold_otimo:.2f}"
    )
    logger.info(
        f"Threshold utilizado na classificacao: {threshold_utilizado:.6f} "
        f"({'KS' if usar_threshold_otimo_ks else 'informado'})"
    )

    # 2. Gini
    gini = calcular_gini(y_true_int, y_pred_proba_float)
    resultados["gini"] = gini
    logger.info(f"Gini: {gini:.2f}")

    # 3. AUC
    auc = calcular_auc(y_true_int, y_pred_proba_float)
    resultados["auc"] = auc
    logger.info(f"AUC-ROC: {auc:.4f}")

    # 4. Matriz de Confusao
    y_pred = (y_pred_proba_float >= threshold_utilizado).astype(int)
    cm = confusion_matrix(y_true_int, y_pred, labels=[0, 1])
    resultados["matriz_confusao"] = cm

    logger.info("\nMatriz de Confusao:")
    logger.info("                Predito: 0    Predito: 1")
    logger.info(f"Real: 0 (Bom)   {cm[0,0]:>8}    {cm[0,1]:>8}")
    logger.info(f"Real: 1 (Mau)   {cm[1,0]:>8}    {cm[1,1]:>8}")

    # 5. Metricas derivadas
    metricas_cm = calcular_metricas_confusao(cm)
    resultados.update(metricas_cm)

    logger.info(f"\nAccuracy: {metricas_cm['accuracy']:.4f}")
    logger.info(f"Precision: {metricas_cm['precision']:.4f}")
    logger.info(f"Recall: {metricas_cm['recall']:.4f}")
    logger.info(f"F1-Score: {metricas_cm['f1_score']:.4f}")
    logger.info(f"Specificity: {metricas_cm['specificity']:.4f}")

    # 6. Swap-in/Swap-out (se disponivel)
    if flag_instalacao_int is not None:
        logger.info("\n" + "=" * 70)
        logger.info("ANALISE SWAP-IN / SWAP-OUT")
        logger.info("=" * 70)

        swap = calcular_swap_in_swap_out(y_true_int, y_pred, flag_instalacao_int)
        resultados["swap_analysis"] = swap

        logger.info("\nSwap-in (Reprovados pela politica -> Aprovados pelo modelo):")
        logger.info(f"  Total: {swap['swap_in_total']:,}")
        logger.info(f"  Bons pagadores: {swap['swap_in_bons']:,} ({swap['swap_in_taxa_acerto']:.2f}%)")
        logger.info(f"  Maus pagadores: {swap['swap_in_maus']:,}")

        logger.info("\nSwap-out (Aprovados pela politica -> Reprovados pelo modelo):")
        logger.info(f"  Total: {swap['swap_out_total']:,}")
        logger.info(f"  Maus pagadores evitados: {swap['swap_out_maus']:,} ({swap['swap_out_taxa_acerto']:.2f}%)")
        logger.info(f"  Bons pagadores perdidos: {swap['swap_out_bons']:,}")

        logger.info("\nResumo:")
        logger.info(f"  Aprovados pela politica: {swap['total_aprovados_politica']:,}")
        logger.info(f"  Aprovados pelo modelo: {swap['total_aprovados_modelo']:,}")
        logger.info(f"  Diferenca: {swap['total_aprovados_modelo'] - swap['total_aprovados_politica']:+,}")

    logger.info("=" * 70)

    return resultados


def comparar_modelos(resultados_modelos: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compara metricas de multiplos modelos.

    Parameters
    ----------
    resultados_modelos : Dict[str, Dict]
        Dicionario com resultados de cada modelo

    Returns
    -------
    pd.DataFrame
        DataFrame comparativo com metricas

    Examples
    --------
    >>> df_comp = comparar_modelos({'LR': res_lr, 'GB': res_gb})
    """
    comparacao = []

    for nome, resultado in resultados_modelos.items():
        comparacao.append(
            {
                "Modelo": nome,
                "KS": resultado.get("ks", 0),
                "Gini": resultado.get("gini", 0),
                "AUC": resultado.get("auc", 0),
                "Accuracy": resultado.get("accuracy", 0),
                "Precision": resultado.get("precision", 0),
                "Recall": resultado.get("recall", 0),
                "F1-Score": resultado.get("f1_score", 0),
            }
        )

    return pd.DataFrame(comparacao)
