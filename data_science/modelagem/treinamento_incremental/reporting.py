"""
Funcoes utilitarias para montagem de relatorios do treinamento incremental.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _import_scikitplot_compat():
    """
    Importa scikit-plot com compatibilidade para versoes novas do SciPy.

    Algumas versoes do `scikit-plot` ainda tentam importar `scipy.interp`,
    removido em versoes recentes do SciPy. Neste caso, cria um alias para
    `numpy.interp` antes de importar o pacote.
    """
    import scipy

    if not hasattr(scipy, "interp"):
        scipy.interp = np.interp  # type: ignore[attr-defined]

    import scikitplot as skplt

    return skplt


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


def salvar_grafico_ks_etapa(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    nome_modelo: str,
    etapa: int,
    output_dir: str = "output",
) -> str:
    """
    Gera e salva o grafico de KS para uma etapa usando scikit-plot.

    Parameters
    ----------
    y_true : np.ndarray
        Vetor real binario da etapa.
    y_pred_proba : np.ndarray
        Probabilidade prevista para classe positiva (shape: [n_amostras]).
    nome_modelo : str
        Nome do modelo.
    etapa : int
        Numero da etapa.
    output_dir : str, optional
        Diretorio onde a imagem sera salva.

    Returns
    -------
    str
        Caminho do arquivo gerado.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    y_true_array = np.asarray(y_true)
    y_pred_proba_array = np.asarray(y_pred_proba, dtype=float)

    if y_true_array.shape[0] != y_pred_proba_array.shape[0]:
        raise ValueError(
            "y_true e y_pred_proba precisam ter o mesmo tamanho para plotar KS."
        )

    y_pred_proba_2d = np.column_stack([1.0 - y_pred_proba_array, y_pred_proba_array])

    nome_limpo = nome_modelo.lower().replace(" ", "_")
    nome_arquivo = f"ks_{nome_limpo}_etapa_{etapa}.png"
    caminho_saida = output_path / nome_arquivo

    fig, ax = plt.subplots(figsize=(10, 6))
    skplt = _import_scikitplot_compat()
    skplt.metrics.plot_ks_statistic(y_true_array, y_pred_proba_2d, ax=ax)

    # Reaplica estilos apos o plot para sobrescrever configuracoes padrao.
    ax.set_title(
        f"KS - {nome_modelo} - Etapa {etapa}",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
    ax.set_ylabel(ax.get_ylabel(), fontweight="bold")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(caminho_saida)
