"""
Script para treino incremental usando apenas Hist Gradient Boosting (HGB).
"""

import os
from pathlib import Path

import pandas as pd
from loguru import logger
from modelagem import TreinamentoIncremental, preparar_dados_modelagem
from processamento_dados import load_all_sources, make_spark
from utils import configure_logger


def salvar_matrizes_confusao_por_etapa(resultados: dict, caminho: str) -> None:
    """
    Salva as matrizes de confusao (2x2) de cada etapa em um unico CSV.
    """
    registros = []

    for modelo, resultado in resultados.items():
        matriz_confusao = resultado.get("matriz_confusao")
        if matriz_confusao is None:
            continue

        try:
            tn = int(matriz_confusao[0][0])
            fp = int(matriz_confusao[0][1])
            fn = int(matriz_confusao[1][0])
            tp = int(matriz_confusao[1][1])
        except (TypeError, IndexError, ValueError):
            logger.warning(f"Matriz de confusao invalida ignorada para {modelo}")
            continue

        registros.append(
            {
                "modelo": modelo,
                "etapa": resultado.get("etapa"),
                "nome_etapa": resultado.get("nome_etapa"),
                "tn_real0_pred0": tn,
                "fp_real0_pred1": fp,
                "fn_real1_pred0": fn,
                "tp_real1_pred1": tp,
            }
        )

    if not registros:
        logger.warning("Nenhuma matriz de confusao encontrada para salvar")
        return

    output_path = Path(caminho)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_matriz_confusao = pd.DataFrame(registros).sort_values(
        by=["etapa", "modelo"],
        na_position="last",
    )
    df_matriz_confusao.to_csv(output_path, index=False)
    logger.success(f"Matrizes de confusao salvas em: {output_path}")


def main():
    """
    Funcao principal para execucao do pipeline de modelagem com HGB.
    """
    configure_logger()

    logger.info("=" * 70)
    logger.info("PIPELINE DE MODELAGEM INCREMENTAL - HIST GRADIENT BOOSTING")
    logger.info("=" * 70)

    # 1. Carregar dados
    logger.info("\n>>> ETAPA 1: CARREGAMENTO DE DADOS")
    spark = make_spark(app_name="hackathon-modelagem-hgb")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, "data")

    logger.info(f"Diretorio de dados: {base_dir}")

    tabelas = load_all_sources(
        spark=spark,
        base_dir=base_dir,
        csv_sep=",",
        csv_header=True,
        csv_infer_schema=True,
    )

    # 2. Preparar dados para modelagem (split treino/OOT)
    logger.info("\n>>> ETAPA 2: SPLIT TREINO/OOT")
    tabelas_treino_oot, metadata, tabelas_gc = preparar_dados_modelagem(tabelas)

    logger.info(f"\nDados preparados para {len(tabelas_treino_oot)} tabelas:")
    for table_name, splits in tabelas_treino_oot.items():
        treino_count = splits["treino"].count()
        oot_count = splits["oot"].count()

        logger.info(f"  - Tabela {table_name}:")
        logger.info(f"    - Treino: {treino_count:,} registros")
        logger.info(f"    - OOT: {oot_count:,} registros")

    logger.info("\nTamanho dos dados para treinamento:")
    for table_name, splits in tabelas_treino_oot.items():
        treino_count = splits["treino"].count()
        oot_count = splits["oot"].count()
        treino_columns = len(splits["treino"].columns)
        logger.info(
            f"  - Tabela {table_name}: "
            f"Treino={treino_count:,} registros, "
            f"{treino_columns} colunas, "
            f"OOT={oot_count:,} registros"
        )

    # 3. Treinamento incremental
    logger.info("\n>>> ETAPA 3: TREINAMENTO INCREMENTAL")
    treinamento = TreinamentoIncremental(
        tables_splited=tabelas_treino_oot,
        etapas=[1, 2, 3, 4, 5, 6],
    )

    logger.info("\n" + "=" * 70)
    logger.info("PARTE UNICA: HIST GRADIENT BOOSTING")
    logger.info("=" * 70)

    df_resumo_hgb = treinamento.executar_treinamento_incremental(
        tipo_modelo="hgb",
        params_modelo={
            "max_iter": 1000,
            "learning_rate": 0.1,
            "max_depth": None,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 50,
            "max_bins":128,
            "l2_regularization": 1.0,
            "early_stopping": "auto",
            "n_iter_no_change":50,
        },
        salvar_graficos_ks=True,
        output_dir_graficos="output/ks_por_etapa_hgb",
    )

    # 4. Salvar resultados
    logger.info("\n>>> ETAPA 4: SALVANDO RESULTADOS")
    caminho_resultados = "output/resultados_incrementais_hgb.csv"
    caminho_matriz_confusao = "output/matriz_confusao_por_etapa_hgb.csv"

    treinamento.salvar_resultados(caminho=caminho_resultados)
    salvar_matrizes_confusao_por_etapa(
        resultados=treinamento.resultados,
        caminho=caminho_matriz_confusao,
    )
    df_resumo_hgb.to_csv("output/resumo_hist_gradient_boosting_hgb.csv", index=False)
    graficos_ks = treinamento.graficos_ks

    logger.success("\n" + "=" * 70)
    logger.success("PIPELINE CONCLUIDO COM SUCESSO!")
    logger.success("=" * 70)
    logger.info("\nArquivos gerados:")
    logger.info(f"  - {caminho_resultados}")
    logger.info(f"  - {caminho_matriz_confusao}")
    logger.info("  - output/resumo_hist_gradient_boosting_hgb.csv")
    logger.info("  - output/ks_por_etapa_hgb/*.png")
    for nome_modelo, caminhos_por_etapa in graficos_ks.items():
        for etapa, caminho_grafico in caminhos_por_etapa.items():
            logger.info(f"  - {caminho_grafico} ({nome_modelo}, etapa {etapa})")

    # Retornar objetos para uso interativo
    return {
        "spark": spark,
        "tables": tabelas,
        "tabelas_treino_oot": tabelas_treino_oot,
        "tables_gc": tabelas_gc,
        "metadata": metadata,
        "treinamento": treinamento,
        "resultados_hgb": df_resumo_hgb,
        "graficos_ks": graficos_ks,
    }


if __name__ == "__main__":
    resultados = main()
