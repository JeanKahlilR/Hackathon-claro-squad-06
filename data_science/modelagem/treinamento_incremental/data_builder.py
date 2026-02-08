"""
Funcoes de acesso aos dados por etapa (Spark -> pandas).
"""

from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType

TableDataFrame = Union[SparkDataFrame, pd.DataFrame]


# Contrato atual: cada etapa usa uma tabela ja agregada.
ETAPA_PARA_TABELA: Dict[int, str] = {
    1: "etapa_1_score_01",
    2: "etapa_2_score_02",
    3: "etapa_3_plus_telco",
    4: "etapa_4_plus_cadastral",
    5: "etapa_5_plus_recarga",
    6: "etapa_6_plus_pagamento",
}

def _resolver_nome_tabela(
    nome: str,
    tables_splited: Dict[str, Dict[str, TableDataFrame]],
) -> Optional[str]:
    """
    Resolve nome de tabela com tolerancia a diferencas de maiusculas/minusculas.
    """
    if nome in tables_splited:
        return nome
    nome_lower = nome.lower()
    for existente in tables_splited.keys():
        if existente.lower() == nome_lower:
            return existente
    return None


def obter_splits_etapa(
    etapa: int,
    tables_splited: Dict[str, Dict[str, TableDataFrame]],
) -> Tuple[str, TableDataFrame, TableDataFrame]:
    """
    Retorna os DataFrames de treino/OOT da etapa.

    Parameters
    ----------
    etapa : int
        Etapa incremental.
    tables_splited : Dict[str, Dict[str, SparkDataFrame]]
        Tabelas separadas em treino e OOT.

    Returns
    -------
    Tuple[str, TableDataFrame, TableDataFrame]
        (nome_tabela, df_treino, df_oot)
    """
    if etapa not in ETAPA_PARA_TABELA:
        raise ValueError(f"Etapa {etapa} nao mapeada para tabela")

    nome_esperado = ETAPA_PARA_TABELA[etapa]
    nome_resolvido = _resolver_nome_tabela(nome_esperado, tables_splited)
    if nome_resolvido is not None:
        splits = tables_splited[nome_resolvido]
        return nome_resolvido, splits["treino"], splits["oot"]

    raise ValueError(
        "Tabela da etapa {} nao encontrada. Esperado: {}. Disponiveis: {}".format(
            etapa,
            nome_esperado,
            list(tables_splited.keys()),
        )
    )


def selecionar_para_pandas(df: TableDataFrame, colunas: List[str]) -> pd.DataFrame:
    """
    Seleciona colunas e retorna pandas DataFrame.

    Para Spark, converte Decimal para double antes do `.toPandas()`.
    Para pandas, apenas seleciona as colunas.
    """
    if isinstance(df, pd.DataFrame):
        faltantes = [c for c in colunas if c not in df.columns]
        if faltantes:
            raise ValueError(f"Colunas ausentes no DataFrame pandas: {faltantes}")
        return df[colunas].copy()

    exprs = []
    for col_name in colunas:
        data_type = df.schema[col_name].dataType
        if isinstance(data_type, DecimalType):
            exprs.append(F.col(col_name).cast("double").alias(col_name))
        else:
            exprs.append(F.col(col_name))
    return df.select(*exprs).toPandas()
