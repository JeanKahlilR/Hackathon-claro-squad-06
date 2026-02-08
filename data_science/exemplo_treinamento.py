"""
Script de exemplo para uso do módulo de modelagem incremental.
Demonstra como treinar modelos com visão incremental de KS.
"""

import os
from loguru import logger
from data_science.processamento_dados import make_spark, load_all_sources
from data_science.utils import configure_logger



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


    # 3. Treinamento Incremental


    # 4. Salvar resultados
    return

if __name__ == "__main__":
    resultados = main()