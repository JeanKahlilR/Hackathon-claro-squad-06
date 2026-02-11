"""
Utilitarios para carregamento de tabelas_treino_oot salvas em parquet.
"""

from pathlib import Path
from typing import Dict

from loguru import logger
from pyspark.sql import DataFrame, SparkSession


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
