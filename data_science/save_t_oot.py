"""
Script para gerar o split treino/OOT e salvar tabelas_treino_oot em parquet.
"""

import os
from pathlib import Path
from typing import Dict

from loguru import logger
from pyspark.sql import DataFrame

from modelagem import preparar_dados_modelagem
from processamento_dados import load_all_sources, make_spark
from utils import configure_logger


def validar_hadoop_home_windows() -> None:
    """
    Valida pre-requisitos de escrita Spark em ambiente Windows.
    """
    if os.name != "nt":
        return

    hadoop_home = os.environ.get("HADOOP_HOME") or os.environ.get("hadoop.home.dir")
    if not hadoop_home:
        raise RuntimeError(
            "Ambiente Windows sem HADOOP_HOME/hadoop.home.dir. "
            "Defina HADOOP_HOME apontando para um diretorio com bin/winutils.exe."
        )

    winutils_path = Path(hadoop_home) / "bin" / "winutils.exe"
    if not winutils_path.exists():
        raise RuntimeError(
            f"winutils.exe nao encontrado em: {winutils_path}. "
            "Ajuste HADOOP_HOME para a pasta correta do Hadoop no Windows."
        )

    os.environ["HADOOP_HOME"] = str(Path(hadoop_home))
    os.environ["hadoop.home.dir"] = str(Path(hadoop_home))


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


def main():
    """
    Funcao principal para execucao ate a etapa de split treino/OOT.
    """
    configure_logger()
    validar_hadoop_home_windows()

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
