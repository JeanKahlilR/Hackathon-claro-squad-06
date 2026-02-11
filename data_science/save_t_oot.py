"""
Script para gerar o split treino/OOT e salvar tabelas_treino_oot em parquet.
"""

import os
from pathlib import Path
from typing import Dict

from loguru import logger
from pyspark.sql import DataFrame, SparkSession

from modelagem import preparar_dados_modelagem
from processamento_dados import load_all_sources, make_spark
from utils import configure_logger


def salvar_tabelas_treino_oot_parquet(
    tabelas_treino_oot: Dict[str, Dict[str, DataFrame]],
    diretorio_saida: str,
) -> None:
    """
    Salva tabelas_treino_oot em parquet separado por tabela e split.

    Estrutura de saida:
      - <diretorio_saida>/<nome_tabela>/treino/
      - <diretorio_saida>/<nome_tabela>/oot/
    """
    output_root = Path(diretorio_saida)
    output_root.mkdir(parents=True, exist_ok=True)

    for nome_tabela, splits in tabelas_treino_oot.items():
        for nome_split in ("treino", "oot"):
            df_split = splits.get(nome_split)
            if df_split is None:
                logger.warning(
                    f"Split '{nome_split}' ausente para a tabela '{nome_tabela}', ignorando."
                )
                continue

            caminho_split = output_root / nome_tabela / nome_split
            df_split.write.mode("overwrite").parquet(str(caminho_split))
            logger.info(f"Parquet salvo: {caminho_split}")

    logger.success(f"tabelas_treino_oot salva em: {output_root}")


def carregar_tabelas_treino_oot_parquet(
    spark: SparkSession,
    diretorio_entrada: str,
) -> Dict[str, Dict[str, DataFrame]]:
    """
    Carrega tabelas_treino_oot salva em parquet separado por tabela/split.
    """
    input_root = Path(diretorio_entrada)
    if not input_root.exists():
        raise FileNotFoundError(f"Diretorio nao encontrado: {input_root}")

    tabelas_treino_oot: Dict[str, Dict[str, DataFrame]] = {}

    for diretorio_tabela in sorted(
        [path for path in input_root.iterdir() if path.is_dir()]
    ):
        nome_tabela = diretorio_tabela.name
        splits: Dict[str, DataFrame] = {}

        for nome_split in ("treino", "oot"):
            caminho_split = diretorio_tabela / nome_split
            if not caminho_split.exists():
                logger.warning(
                    f"Split '{nome_split}' nao encontrado para a tabela '{nome_tabela}'."
                )
                continue

            splits[nome_split] = spark.read.parquet(str(caminho_split))

        if splits:
            tabelas_treino_oot[nome_tabela] = splits

    logger.success(
        f"tabelas_treino_oot carregada com {len(tabelas_treino_oot)} tabela(s) de {input_root}"
    )
    return tabelas_treino_oot


def main():
    """
    Funcao principal para execucao ate a etapa de split treino/OOT.
    """
    configure_logger()

    logger.info("=" * 70)
    logger.info("PIPELINE PARA SALVAR TABELAS TREINO/OOT")
    logger.info("=" * 70)

    # 1. Carregar dados
    logger.info("\n>>> ETAPA 1: CARREGAMENTO DE DADOS")
    spark = make_spark(app_name="hackathon-save-tabelas-treino-oot")

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
    for nome_tabela, splits in tabelas_treino_oot.items():
        treino_count = splits["treino"].count()
        oot_count = splits["oot"].count()

        logger.info(f"  - Tabela {nome_tabela}:")
        logger.info(f"    - Treino: {treino_count:,} registros")
        logger.info(f"    - OOT: {oot_count:,} registros")

    caminho_saida_parquet = "output/tabelas_treino_oot"
    salvar_tabelas_treino_oot_parquet(
        tabelas_treino_oot=tabelas_treino_oot,
        diretorio_saida=caminho_saida_parquet,
    )

    logger.success("\n" + "=" * 70)
    logger.success("PROCESSO CONCLUIDO COM SUCESSO!")
    logger.success("=" * 70)
    logger.info("\nDiretorio gerado:")
    logger.info(f"  - {caminho_saida_parquet}")

    return {
        "spark": spark,
        "tables": tabelas,
        "tabelas_treino_oot": tabelas_treino_oot,
        "tables_gc": tabelas_gc,
        "metadata": metadata,
        "caminho_saida_parquet": caminho_saida_parquet,
    }


if __name__ == "__main__":
    resultados = main()
