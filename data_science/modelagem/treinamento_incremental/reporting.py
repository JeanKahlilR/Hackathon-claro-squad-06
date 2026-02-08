"""
Funcoes utilitarias para montagem de relatorios do treinamento incremental.
"""

from typing import Dict, List

import pandas as pd


def criar_resumo_incremental(resultados_lista: List[Dict]) -> pd.DataFrame:
    """
    Converte lista de resultados por etapa em DataFrame resumo.

    Parameters
    ----------
    resultados_lista : List[Dict]
        Lista com dicionarios de resultados gerados em cada etapa.

    Returns
    -------
    pd.DataFrame
        Tabela padronizada com metricas principais por etapa.
    """
    resumo = []
    for res in resultados_lista:
        # Seleciona somente campos de interesse para leitura executiva.
        resumo.append(
            {
                "Etapa": res["etapa"],
                "Nome": res["nome_etapa"],
                "Features": res["num_features"],
                "KS": res["ks"],
                "Delta_KS": res["delta_ks"],
                "Gini": res["gini"],
                "AUC": res["auc"],
                "Accuracy": res["accuracy"],
                "Precision": res["precision"],
                "Recall": res["recall"],
            }
        )
    return pd.DataFrame(resumo)


def criar_comparacao_modelos(df_resumo_lr: pd.DataFrame, df_resumo_gb: pd.DataFrame) -> pd.DataFrame:
    """
    Gera comparativo entre resumos de regressao logistica e gradient boosting.

    Parameters
    ----------
    df_resumo_lr : pd.DataFrame
        Resumo incremental da regressao logistica.
    df_resumo_gb : pd.DataFrame
        Resumo incremental do gradient boosting.

    Returns
    -------
    pd.DataFrame
        DataFrame com metricas lado a lado e diferenca de KS por etapa.
    """
    # Assume que os resumos ja estao alinhados por etapa/ordem.
    return pd.DataFrame(
        {
            "Etapa": df_resumo_lr["Etapa"],
            "Nome": df_resumo_lr["Nome"],
            "Features": df_resumo_lr["Features"],
            "KS_LR": df_resumo_lr["KS"],
            "Delta_KS_LR": df_resumo_lr["Delta_KS"],
            "KS_GB": df_resumo_gb["KS"],
            "Delta_KS_GB": df_resumo_gb["Delta_KS"],
            "Diferenca_KS": df_resumo_gb["KS"] - df_resumo_lr["KS"],
        }
    )


def resultados_para_dataframe(resultados: Dict[str, Dict]) -> pd.DataFrame:
    """
    Serializa dicionario de resultados para estrutura tabular unica.

    Parameters
    ----------
    resultados : Dict[str, Dict]
        Estrutura consolidada com resultados por modelo/etapa.

    Returns
    -------
    pd.DataFrame
        DataFrame pronto para exportacao (ex.: CSV).
    """
    dados = []
    for chave, resultado in resultados.items():
        # Usa `.get` para tolerar campos opcionais em diferentes modelos/etapas.
        dados.append(
            {
                "modelo": chave,
                "etapa": resultado.get("etapa"),
                "nome_etapa": resultado.get("nome_etapa"),
                "num_features": resultado.get("num_features"),
                "ks": resultado.get("ks"),
                "delta_ks": resultado.get("delta_ks"),
                "gini": resultado.get("gini"),
                "auc": resultado.get("auc"),
                "accuracy": resultado.get("accuracy"),
                "precision": resultado.get("precision"),
                "recall": resultado.get("recall"),
                "f1_score": resultado.get("f1_score"),
            }
        )
    return pd.DataFrame(dados)
