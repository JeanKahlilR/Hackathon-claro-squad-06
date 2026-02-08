"""
Funcoes utilitarias para criacao e configuracao de SparkSession.
"""

import json
from pathlib import Path
from typing import Dict, Optional

from loguru import logger
from pyspark.sql import SparkSession

SPARK_DEFAULT_CONFIG: Dict[str, str] = {
    "spark.driver.memory": "4g",
    "spark.sql.autoBroadcastJoinThreshold": "-1",
    "spark.sql.shuffle.partitions": "200",
}


def _load_spark_config(config_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Carrega configuracoes do Spark de arquivo JSON.

    O formato esperado e um objeto com pares chave/valor no padrao:
    {
      "spark.driver.memory": "4g",
      "spark.sql.autoBroadcastJoinThreshold": "-1"
    }
    """
    config_file = config_path or Path(__file__).with_name("spark_config.json")

    if not config_file.exists():
        logger.warning(f"Arquivo de configuracao nao encontrado: {config_file}. Usando defaults.")
        return dict(SPARK_DEFAULT_CONFIG)

    try:
        with config_file.open(mode="r", encoding="utf-8") as file:
            loaded_config = json.load(file)
    except json.JSONDecodeError as e:
        logger.warning(f"Erro ao ler JSON de configuracao ({config_file}): {e}. Usando defaults.")
        return dict(SPARK_DEFAULT_CONFIG)

    if not isinstance(loaded_config, dict):
        logger.warning(f"Formato invalido em {config_file}. Usando defaults.")
        return dict(SPARK_DEFAULT_CONFIG)

    spark_config = dict(SPARK_DEFAULT_CONFIG)
    for key, value in loaded_config.items():
        if key.startswith("spark."):
            spark_config[key] = str(value)
        else:
            logger.warning(f"Chave ignorada no config Spark: {key}")

    return spark_config


def make_spark(app_name: str = "hackathon-load-data") -> SparkSession:
    """
    Cria ou recupera uma SparkSession com configuracoes base do projeto.

    Parameters
    ----------
    app_name : str, optional
        Nome da aplicacao Spark.

    Returns
    -------
    SparkSession
        Sessao Spark pronta para leitura e processamento.
    """
    spark_config = _load_spark_config()
    logger.info(f"Criando/recuperando SparkSession com app_name='{app_name}'")
    builder = SparkSession.builder.appName(app_name)
    for key, value in spark_config.items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()

    logger.success(f"SparkSession criada com sucesso: {spark.sparkContext.appName}")
    logger.info("Configuracoes aplicadas:")
    for key, value in spark_config.items():
        logger.info(f"  - {key}: {value}")
    return spark
