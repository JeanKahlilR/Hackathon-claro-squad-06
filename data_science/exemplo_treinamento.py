"""
Script de exemplo para uso do módulo de modelagem incremental.
Demonstra como treinar modelos com visão incremental de KS.
"""

import os
from loguru import logger
from data_science.processamento_dados import make_spark, load_all_sources
from data_science.utils import configure_logger
from data_science.modelagem import preparar_dados_modelagem


def main():
    """
    Função principal para execução do pipeline de modelagem.
    """
    configure_logger()

    logger.info("="*70)
    logger.info("PIPELINE DE MODELAGEM INCREMENTAL")
    logger.info("="*70)

    # 1. Carregar dados
    logger.info("\n>>> ETAPA 1: CARREGAMENTO DE DADOS")
    spark = make_spark(app_name="hackathon-modelagem")

    # Obter caminho absoluto para o diretório de dados
    # Baseado na localização deste script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    BASE_DIR = os.path.join(project_root, "data")

    logger.info(f"Diretório de dados: {BASE_DIR}")

    tables = load_all_sources(
        spark=spark,
        base_dir=BASE_DIR,
        csv_sep=",",
        csv_header=True,
        csv_infer_schema=True,
    )

    # 2. Preparar dados para modelagem (Split Treino/OOT)
    logger.info("\n>>> ETAPA 2: SPLIT TREINO/OOT")

    # Chamar a função de preparação de dados
    tabelas_treino_oot, metadata, tabelas_gc = preparar_dados_modelagem(tables)

    # Exibir informações sobre os dados preparados
    logger.info(f"\nDados preparados para {len(tabelas_treino_oot)} tabelas:")

    for table_name, splits in tabelas_treino_oot.items():
        treino_count = splits['treino'].count()
        oot_count = splits['oot'].count()

        logger.info(f"  - Tabela {table_name}:")
        logger.info(f"    - Treino: {treino_count:,} registros")
        logger.info(f"    - OOT: {oot_count:,} registros")

    # Verificar tamanho dos dados
    logger.info("\nTamanho dos dados para treinamento:")
    for table_name, splits in tabelas_treino_oot.items():
        treino_count = splits['treino'].count()
        oot_count = splits['oot'].count()
        treino_columns = len(splits['treino'].columns)
        logger.info(f"  - Tabela {table_name}: Treino={treino_count:,} registros, {treino_columns} colunas, OOT={oot_count:,} registros")


    # 3. Treinamento Incremental


    # 4. Salvar resultados
    return

if __name__ == "__main__":
    resultados = main()