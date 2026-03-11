"""
Script para treino incremental usando apenas XGBoost (XGB).
"""

import os
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from modelagem import TreinamentoIncremental, preparar_dados_modelagem
from processamento_dados import load_all_sources, make_spark
from utils import configure_logger


def amostrar_apenas_treino_por_tabela(
    tabelas_treino_oot: dict,
    fracao_treino: float,
    seed: int = 42,
) -> dict:
    """
    Aplica amostragem somente no split de treino, mantendo OOT completo.
    """
    if not 0 < fracao_treino <= 1:
        raise ValueError("fracao_treino deve estar no intervalo (0, 1].")

    if fracao_treino == 1:
        logger.info("Amostragem de treino desabilitada (fracao_treino=1.0).")
        return tabelas_treino_oot

    logger.info(
        "Aplicando amostragem apenas no treino | fracao_treino={} | seed={}",
        fracao_treino,
        seed,
    )

    tabelas_amostradas = {}
    for table_name, splits in tabelas_treino_oot.items():
        df_treino = splits["treino"]
        df_oot = splits["oot"]

        if isinstance(df_treino, pd.DataFrame):
            df_treino_amostrado = df_treino.sample(frac=fracao_treino, random_state=seed)
            treino_count_amostrado = len(df_treino_amostrado)
        else:
            df_treino_amostrado = df_treino.sample(
                withReplacement=False,
                fraction=fracao_treino,
                seed=seed,
            )
            treino_count_amostrado = df_treino_amostrado.count()

        oot_count = len(df_oot) if isinstance(df_oot, pd.DataFrame) else df_oot.count()
        logger.info(
            "  - {}: treino_amostrado={} | oot_original={}",
            table_name,
            f"{treino_count_amostrado:,}",
            f"{oot_count:,}",
        )

        tabelas_amostradas[table_name] = {
            "treino": df_treino_amostrado,
            "oot": df_oot,
        }

    return tabelas_amostradas


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


def salvar_modelo_final_joblib(
    modelos_treinados: dict,
    caminho: str,
    etapa_final: int,
) -> None:
    """
    Salva em joblib o modelo treinado da etapa final.
    """
    if not modelos_treinados:
        logger.warning("Nenhum modelo treinado encontrado para salvar")
        return

    prefixo_etapa_final = f"etapa_{etapa_final}_"
    candidatos = [
        (chave, modelo)
        for chave, modelo in modelos_treinados.items()
        if chave.startswith(prefixo_etapa_final)
    ]

    if not candidatos:
        logger.warning(f"Nenhum modelo encontrado para a etapa final {etapa_final}")
        return

    chave_modelo, modelo_final = candidatos[-1]
    output_path = Path(caminho)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo_final, output_path)
    logger.success(f"Modelo final salvo em: {output_path} ({chave_modelo})")


def main():
    """
    Funcao principal para execucao do pipeline de modelagem com XGBoost.
    """
    configure_logger()

    logger.info("=" * 70)
    logger.info("PIPELINE DE MODELAGEM INCREMENTAL - XGBOOST")
    logger.info("=" * 70)

    # 1. Carregar dados
    logger.info("\n>>> ETAPA 1: CARREGAMENTO DE DADOS")
    spark = make_spark(app_name="hackathon-modelagem-xgb")

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
    """tabelas_treino_oot = amostrar_apenas_treino_por_tabela(
        tabelas_treino_oot=tabelas_treino_oot,
        fracao_treino=0.20,
        seed=42,
    )"""

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
    logger.info("PARTE UNICA: XGBOOST")
    logger.info("=" * 70)

    df_resumo_xgb = treinamento.executar_treinamento_incremental(
        tipo_modelo="xgboost",
        params_modelo={
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "importance_type": "gain",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": 0,
        },
        salvar_graficos_ks=True,
        output_dir_graficos="output/ks_por_etapa_xgb",
        calcular_importancia_variaveis=True,
        metodo_importancia="xgboost_native",
        output_dir_importancia="output/importancia_variaveis_xgb",
        scoring_importancia="roc_auc",
        n_repeats_importancia=10,
        salvar_grafico_importancia=True,
        top_n_grafico_importancia=20,
    )

    # 4. Salvar resultados
    logger.info("\n>>> ETAPA 4: SALVANDO RESULTADOS")
    caminho_resultados = "output/resultados_incrementais_xgb.csv"
    caminho_matriz_confusao = "output/matriz_confusao_por_etapa_xgb.csv"
    caminho_modelo_final = "output/modelo_final_xgb.pkl"

    treinamento.salvar_resultados(caminho=caminho_resultados)
    salvar_matrizes_confusao_por_etapa(
        resultados=treinamento.resultados,
        caminho=caminho_matriz_confusao,
    )
    salvar_modelo_final_joblib(
        modelos_treinados=treinamento.modelos_treinados,
        caminho=caminho_modelo_final,
        etapa_final=max(treinamento.etapas),
    )
    df_resumo_xgb.to_csv("output/resumo_xgboost_xgb.csv", index=False)
    graficos_ks = treinamento.graficos_ks
    arquivos_importancia = treinamento.estado.arquivos_importancia

    logger.success("\n" + "=" * 70)
    logger.success("PIPELINE CONCLUIDO COM SUCESSO!")
    logger.success("=" * 70)
    logger.info("\nArquivos gerados:")
    logger.info(f"  - {caminho_resultados}")
    logger.info(f"  - {caminho_matriz_confusao}")
    logger.info(f"  - {caminho_modelo_final}")
    logger.info("  - output/resumo_xgboost_xgb.csv")
    logger.info("  - output/ks_por_etapa_xgb/*.png")
    logger.info("  - output/importancia_variaveis_xgb/*.csv")
    logger.info("  - output/importancia_variaveis_xgb/*.png")
    for nome_modelo, caminhos_por_etapa in graficos_ks.items():
        for etapa, caminho_grafico in caminhos_por_etapa.items():
            logger.info(f"  - {caminho_grafico} ({nome_modelo}, etapa {etapa})")
    for chave, artefatos in arquivos_importancia.items():
        caminho_csv = artefatos.get("csv")
        caminho_png = artefatos.get("png")
        if caminho_csv:
            logger.info(f"  - {caminho_csv} ({chave})")
        if caminho_png:
            logger.info(f"  - {caminho_png} ({chave})")

    # Retornar objetos para uso interativo
    return {
        "spark": spark,
        "tables": tabelas,
        "tabelas_treino_oot": tabelas_treino_oot,
        "tables_gc": tabelas_gc,
        "metadata": metadata,
        "treinamento": treinamento,
        "resultados_xgb": df_resumo_xgb,
        "graficos_ks": graficos_ks,
        "arquivos_importancia": arquivos_importancia,
    }


if __name__ == "__main__":
    resultados = main()
