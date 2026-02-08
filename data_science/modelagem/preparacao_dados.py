# modelagem/preparacao_dados.py
"""
Módulo para preparação de dados: conversão Spark->Pandas, split treino/teste/OOT,
tratamento de missing values e encoding de variáveis categóricas.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col
from loguru import logger

from .config import SAFRA_OOT, TARGET_COLUMN


def tirar_grupo_controle(
    tables: Dict[str, SparkDataFrame],
    coluna_grupo_controle: str = "GRUPO_CONTROLE",
    valor_grupo_controle: int = 1,
) -> Tuple[Dict[str, SparkDataFrame], Dict[str, SparkDataFrame]]:
    """
    Separa registros do grupo controle em um dicionario dedicado.

    Parameters
    ----------
    tables : Dict[str, SparkDataFrame]
        Dicionario com tabelas Spark.
    coluna_grupo_controle : str, optional
        Nome da coluna indicadora de grupo controle.
    valor_grupo_controle : int, optional
        Valor que identifica grupo controle.

    Returns
    -------
    Tuple[Dict[str, SparkDataFrame], Dict[str, SparkDataFrame]]
        (tables_sem_gc, tables_gc) onde:
        - tables_sem_gc mantem apenas registros fora do grupo controle
        - tables_gc usa chave "<nome_tabela>_gc" com registros do grupo controle
    """
    logger.info("=" * 70)
    logger.info("SEPARACAO DO GRUPO CONTROLE")
    logger.info("=" * 70)

    tables_sem_gc = {}
    tables_gc = {}

    for table_name, spark_df in tables.items():
        if coluna_grupo_controle not in spark_df.columns:
            raise ValueError(
                f"Coluna '{coluna_grupo_controle}' nao encontrada na tabela '{table_name}'"
            )

        df_gc = spark_df.filter(col(coluna_grupo_controle) == valor_grupo_controle)
        df_sem_gc = spark_df.filter(
            col(coluna_grupo_controle).isNull() | (col(coluna_grupo_controle) != valor_grupo_controle)
        )

        gc_table_name = f"{table_name}_gc"
        tables_sem_gc[table_name] = df_sem_gc
        tables_gc[gc_table_name] = df_gc

        registros_total = spark_df.count()
        registros_gc = df_gc.count()
        registros_sem_gc = df_sem_gc.count()

        logger.info(
            f"Tabela {table_name}: total={registros_total:,} | sem_gc={registros_sem_gc:,} | gc={registros_gc:,}"
        )

    logger.info("=" * 70)
    return tables_sem_gc, tables_gc

def split_treino_oot_spark(
    df: SparkDataFrame,
    safra_col: str = "SAFRA",
    safra_oot: List[int] = SAFRA_OOT
) -> Tuple[SparkDataFrame, SparkDataFrame]:
    """
    Separa dados em treino e OOT (Out-of-Time) baseado na coluna SAFRA para Spark DataFrames.

    Parameters
    ----------
    df : SparkDataFrame
        DataFrame Spark completo
    safra_col : str, optional
        Nome da coluna de safra (default: "SAFRA")
    safra_oot : List[int], optional
        Lista de safras para OOT (default: [202502, 202503])

    Returns
    -------
    Tuple[SparkDataFrame, SparkDataFrame]
        (df_treino, df_oot)

    Examples
    --------
    >>> df_treino, df_oot = split_treino_oot_spark(spark_df)
    >>> print(f"Treino: {df_treino.count()}, OOT: {df_oot.count()}")
    """
    logger.info("="*70)
    logger.info("SPLIT TREINO/OOT (SPARK)")
    logger.info("="*70)

    if safra_col not in df.columns:
        raise ValueError(f"Coluna '{safra_col}' não encontrada no DataFrame")

    logger.info(f"Safras OOT: {safra_oot}")

    # Separar OOT
    df_oot = df.filter(col(safra_col).isin(safra_oot))

    # Separar Treino (todas as safras que não são OOT)
    df_treino = df.filter(~col(safra_col).isin(safra_oot))

    total_registros = df.count()
    registros_treino = df_treino.count()
    registros_oot = df_oot.count()

    logger.info(f"Total de registros: {total_registros:,}")
    logger.info(f"  - Treino: {registros_treino:,} ({registros_treino/total_registros*100:.2f}%)")
    logger.info(f"  - OOT: {registros_oot:,} ({registros_oot/total_registros*100:.2f}%)")

    logger.info("="*70)

    return df_treino, df_oot

def split_treino_oot(
    df: pd.DataFrame,
    safra_col: str = "SAFRA",
    safra_oot: List[int] = SAFRA_OOT
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa dados em treino e OOT (Out-of-Time) baseado na coluna SAFRA.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame completo
    safra_col : str, optional
        Nome da coluna de safra (default: "SAFRA")
    safra_oot : List[int], optional
        Lista de safras para OOT (default: [202502, 202503])

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (df_treino, df_oot)

    Examples
    --------
    >>> df_treino, df_oot = split_treino_oot(df)
    >>> print(f"Treino: {len(df_treino)}, OOT: {len(df_oot)}")
    Treino: 1200000, OOT: 379853
    """
    logger.info("="*70)
    logger.info("SPLIT TREINO/OOT")
    logger.info("="*70)

    if safra_col not in df.columns:
        raise ValueError(f"Coluna '{safra_col}' não encontrada no DataFrame")

    logger.info(f"Safras OOT: {safra_oot}")

    # Separar OOT
    df_oot = df[df[safra_col].isin(safra_oot)].copy()

    # Separar Treino (todas as safras que não são OOT)
    df_treino = df[~df[safra_col].isin(safra_oot)].copy()

    logger.info(f"Total de registros: {len(df):,}")
    logger.info(f"  - Treino: {len(df_treino):,} ({len(df_treino)/len(df)*100:.2f}%)")
    logger.info(f"  - OOT: {len(df_oot):,} ({len(df_oot)/len(df)*100:.2f}%)")

    # Distribuição do target
    if TARGET_COLUMN in df.columns:
        try:
            # Converter para numérico se necessário
            df_treino_fpd = pd.to_numeric(df_treino[TARGET_COLUMN], errors='coerce')
            df_oot_fpd = pd.to_numeric(df_oot[TARGET_COLUMN], errors='coerce')

            taxa_fpd_treino = df_treino_fpd.mean() * 100
            taxa_fpd_oot = df_oot_fpd.mean() * 100

            logger.info(f"Taxa de FPD:")
            logger.info(f"  - Treino: {taxa_fpd_treino:.2f}%")
            logger.info(f"  - OOT: {taxa_fpd_oot:.2f}%")
        except Exception as e:
            logger.warning(f"Não foi possível calcular taxa de FPD: {e}")

    logger.info("="*70)

    return df_treino, df_oot

def preparar_dados_modelagem(
    tables: Dict[str, SparkDataFrame],
    safra_oot: List[int] = SAFRA_OOT
) -> Tuple[Dict[str, Dict[str, SparkDataFrame]], Dict, Dict[str, SparkDataFrame]]:
    """
    Pipeline completo de preparação de dados para modelagem.

    Parameters
    ----------
    tables : Dict[str, SparkDataFrame]
        Dicionário contendo as tabelas Spark a serem processadas
    safra_oot : List[int], optional
        Lista de safras para OOT

    Returns
    -------
    Tuple[Dict[str, Dict[str, SparkDataFrame]], Dict, Dict[str, SparkDataFrame]]
        (tables_splited, metadata, tables_gc) onde:
        - tables_splited contém para cada tabela um dicionário com 'treino' e 'oot'
        - metadata contém contagens de apoio
        - tables_gc contém as tabelas do grupo controle com sufixo "_gc"

    Examples
    --------
    >>> tables_splited, metadata, tables_gc = preparar_dados_modelagem(tables)
    """
    logger.info("="*70)
    logger.info("PIPELINE DE PREPARAÇÃO DE DADOS")
    logger.info("="*70)

    metadata = {}
    tables_splited = {}
    tables_sem_gc, tables_gc = tirar_grupo_controle(tables)

    # Processar cada tabela individualmente
    for table_name, spark_df in tables_sem_gc.items():
        logger.info(f"\nProcessando tabela: {table_name}")

        # Split Treino/OOT para cada tabela
        df_treino, df_oot = split_treino_oot_spark(spark_df, safra_oot=safra_oot)

        # Armazenar os resultados
        tables_splited[table_name] = {
            'treino': df_treino,
            'oot': df_oot
        }

        # Armazenar metadata
        registros_treino = df_treino.count()
        registros_oot = df_oot.count()
        registros_sem_gc = spark_df.count()
        registros_gc = tables_gc[f"{table_name}_gc"].count()
        total_registros = registros_sem_gc + registros_gc

        metadata[table_name] = {
            'registros_treino': registros_treino,
            'registros_oot': registros_oot,
            'registros_sem_gc': registros_sem_gc,
            'registros_grupo_controle': registros_gc,
            'total_registros': total_registros
        }

        logger.info(f"Tabela {table_name} processada:")
        logger.info(f"  - Treino: {registros_treino:,} registros")
        logger.info(f"  - OOT: {registros_oot:,} registros")
        logger.info(f"  - Grupo controle ({table_name}_gc): {registros_gc:,} registros")

    logger.success("\nPREPARAÇÃO DE DADOS CONCLUÍDA")
    logger.info("="*70)

    return tables_splited, metadata, tables_gc
