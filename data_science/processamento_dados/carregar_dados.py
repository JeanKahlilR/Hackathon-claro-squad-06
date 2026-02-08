"""
Funcoes para carregamento de arquivos de dados em DataFrames Spark.
"""

import re
from itertools import count
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
from pyspark.sql import DataFrame, SparkSession

SUPPORTED_EXTS: Tuple[str, ...] = (".parquet", ".snappy.parquet", ".csv")

# Mapeia nome de pasta em `data/` para chave canonica por etapa de modelagem.
PASTA_PARA_TABELA_ETAPA: Dict[str, str] = {
    "score_01": "etapa_1_score_01",
    "score_02": "etapa_2_score_02",
    "plus_telco": "etapa_3_plus_telco",
    "plus_cadastral": "etapa_4_plus_cadastral",
    "plus_recarga": "etapa_5_plus_recarga",
    "plus_pagamento": "etapa_6_plus_pagamento",
}


def list_data_files(root_dir: Optional[str] = None, exts: Tuple[str, ...] = SUPPORTED_EXTS) -> List[str]:
    """
    Lista recursivamente arquivos suportados dentro de `root_dir`.

    Se `root_dir` nao for informado, usa a pasta `data` relativa ao projeto.

    Parameters
    ----------
    root_dir : Optional[str], optional
        Diretorio base para busca recursiva.
    exts : Tuple[str, ...], optional
        Extensoes aceitas na busca.

    Returns
    -------
    List[str]
        Lista ordenada com os caminhos dos arquivos encontrados.
    """
    if root_dir is None:
        root_dir = Path(__file__).parent.parent.parent / "data"
        logger.debug(f"Nenhum diretorio especificado, usando padrao: {root_dir}")
    else:
        root_dir = Path(root_dir)
        logger.debug(f"Buscando arquivos em: {root_dir}")

    paths = [
        str(file_path)
        for file_path in root_dir.rglob("*")
        if file_path.is_file() and any(file_path.name.lower().endswith(ext) for ext in exts)
    ]

    logger.info(f"Encontrados {len(paths)} arquivo(s) com extensoes {exts} em {root_dir}")
    return sorted(paths)


def safe_key_from_path(path: str) -> str:
    """
    Gera um nome seguro para uso como chave de dicionario.

    Parameters
    ----------
    path : str
        Caminho de arquivo ou pasta.

    Returns
    -------
    str
        Nome normalizado com caracteres invalidos substituidos por `_`.
    """
    base = Path(path).stem
    base = re.sub(r"[^0-9a-zA-Z_]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "table"


def table_key_from_folder_name(folder_name: str) -> str:
    """
    Resolve chave de tabela priorizando mapeamento por etapa.

    Parameters
    ----------
    folder_name : str
        Nome da subpasta em `data/`.

    Returns
    -------
    str
        Nome canonico da tabela para uso no pipeline.
    """
    safe_name = safe_key_from_path(folder_name).lower()
    return PASTA_PARA_TABELA_ETAPA.get(safe_name, safe_name)


def _generate_unique_key(base_key: str, existing_keys: set) -> str:
    """
    Gera uma chave unica adicionando sufixos numericos quando necessario.

    Parameters
    ----------
    base_key : str
        Chave base desejada.
    existing_keys : set
        Conjunto de chaves ja existentes.

    Returns
    -------
    str
        Chave unica que nao existe em `existing_keys`.
    """
    if base_key not in existing_keys:
        return base_key

    for i in count(2):
        unique_key = f"{base_key}_{i}"
        if unique_key not in existing_keys:
            logger.debug(f"Colisao de nome detectada: '{base_key}' renomeado para '{unique_key}'")
            return unique_key


def load_folder_as_table(
    spark: SparkSession,
    folder_path: str,
    csv_sep: str = ",",
    csv_header: bool = True,
    csv_infer_schema: bool = True,
) -> DataFrame:
    """
    Carrega todos os arquivos suportados de `folder_path` em um unico DataFrame.

    Parameters
    ----------
    spark : SparkSession
        Sessao Spark ativa.
    folder_path : str
        Caminho da pasta com arquivos de entrada.
    csv_sep : str, optional
        Separador usado para leitura de CSV.
    csv_header : bool, optional
        Indica se CSV possui cabecalho.
    csv_infer_schema : bool, optional
        Indica se o schema do CSV deve ser inferido.

    Returns
    -------
    DataFrame
        DataFrame Spark consolidando os dados da pasta.

    Raises
    ------
    ValueError
        Quando nao ha arquivos na pasta ou o formato nao e suportado.
    """
    logger.info(f"Iniciando carregamento de arquivos da pasta: {folder_path}")

    files = list_data_files(folder_path)

    if not files:
        logger.warning(f"Nenhum arquivo encontrado em: {folder_path}")
        raise ValueError(f"Nenhum arquivo encontrado em: {folder_path}")

    folder_name = Path(folder_path).name
    table_name = safe_key_from_path(folder_name)

    try:
        first_file = Path(files[0])
        file_extension = first_file.suffix.lower()

        logger.debug(f"Detectada extensao: {file_extension} para arquivo: {first_file.name}")

        if file_extension == ".csv":
            df = (
                spark.read.option("header", str(csv_header).lower())
                .option("inferSchema", str(csv_infer_schema).lower())
                .option("sep", csv_sep)
                .csv(files)
            )
        elif file_extension in [".parquet", ".snappy.parquet"]:
            df = spark.read.parquet(*files)
        else:
            supported = ", ".join(SUPPORTED_EXTS)
            error_msg = f"Formato nao suportado '{file_extension}'. Formatos aceitos: {supported}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Pasta carregada com sucesso: '{table_name}' ({df.count()} linhas, {len(df.columns)} colunas)")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar arquivos da pasta {folder_path}: {e}")
        raise


def load_all_sources(
    spark: SparkSession,
    base_dir: str,
    csv_sep: str = ",",
    csv_header: bool = True,
    csv_infer_schema: bool = True,
) -> Dict[str, DataFrame]:
    """
    Carrega todas as subpastas de `base_dir` como tabelas Spark.

    Cada subpasta eh convertida em uma tabela, combinando arquivos `.parquet`
    e `.csv` suportados.

    Parameters
    ----------
    spark : SparkSession
        Sessao Spark ativa.
    base_dir : str
        Diretorio raiz onde buscar subpastas.
    csv_sep : str, optional
        Separador do CSV.
    csv_header : bool, optional
        Se CSV possui cabecalho.
    csv_infer_schema : bool, optional
        Se deve inferir schema no CSV.

    Returns
    -------
    Dict[str, DataFrame]
        Dicionario no formato `{nome_tabela_etapa: DataFrame}`.

    Examples
    --------
    >>> spark = make_spark()
    >>> tables = load_all_sources(spark, "data")
    >>> print(tables.keys())
    dict_keys(['etapa_1_score_01', 'etapa_2_score_02', 'etapa_3_plus_telco'])
    """
    logger.info(f"Iniciando carregamento de subpastas em: {base_dir}")

    base_path = Path(base_dir)

    if not base_path.exists():
        logger.error(f"Diretorio nao encontrado: {base_dir}")
        return {}

    subfolders = [
        str(subfolder)
        for subfolder in base_path.iterdir()
        if subfolder.is_dir()
    ]

    if not subfolders:
        logger.warning(f"Nenhuma subpasta encontrada em: {base_dir}")
        return {}

    logger.info(f"Encontradas {len(subfolders)} subpasta(s) para processar")

    tables: Dict[str, DataFrame] = {}

    for folder_path in subfolders:
        folder_name = Path(folder_path).name
        logger.info(f"Processando subpasta: {folder_name}")

        try:
            df = load_folder_as_table(
                spark=spark,
                folder_path=folder_path,
                csv_sep=csv_sep,
                csv_header=csv_header,
                csv_infer_schema=csv_infer_schema,
            )

            table_name = table_key_from_folder_name(folder_name)
            table_name = _generate_unique_key(table_name, set(tables.keys()))
            tables[table_name] = df
            logger.success(f"'{table_name}' carregado: {df.count()} linhas, {len(df.columns)} colunas")
        except ValueError as e:
            logger.warning(f"Subpasta '{folder_name}' ignorada: {e}")
            continue
        except Exception as e:
            logger.error(f"Erro ao carregar subpasta {folder_path}: {e}")
            raise

    logger.info("=" * 70)
    logger.success(f"Carregamento concluido: {len(tables)} tabela(s) carregada(s)")
    logger.info("=" * 70)

    for table_name in sorted(tables.keys()):
        df = tables[table_name]
        logger.info(f"  - {table_name}: {len(df.columns)} colunas")

    logger.info("=" * 70)

    return tables
