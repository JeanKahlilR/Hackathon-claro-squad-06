"""
Script de exemplo para uso do módulo de modelagem incremental.
Demonstra como treinar modelos com visão incremental de KS.
"""

import os
from loguru import logger
from data_science.processamento_dados import make_spark, load_all_sources
from data_science.utils import configure_logger
from data_science.modelagem import preparar_dados_modelagem, TreinamentoIncremental


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

    tabelas = load_all_sources(
        spark=spark,
        base_dir=BASE_DIR,
        csv_sep=",",
        csv_header=True,
        csv_infer_schema=True,
    )

    # 2. Preparar dados para modelagem (Split Treino/OOT)
    logger.info("\n>>> ETAPA 2: SPLIT TREINO/OOT")

    # Chamar a função de preparação de dados
    tabelas_treino_oot, metadata, tabelas_gc = preparar_dados_modelagem(tabelas)

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
    logger.info("\n>>> ETAPA 3: TREINAMENTO INCREMENTAL")

    # Inicializar treinamento incremental
    treinamento = TreinamentoIncremental(
        tables_splited=tabelas_treino_oot,
        etapas=[1, 2, 3, 4, 5, 6]  # Etapas disponiveis
    )

    # PARTE 1: Treinar Regressão Logística
    logger.info("\n" + "="*70)
    logger.info("PARTE 1: REGRESSÃO LOGÍSTICA")
    logger.info("="*70)

    df_resumo_lr = treinamento.executar_treinamento_incremental(
        tipo_modelo='logistica',
        params_modelo={
            'C': 1.0,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'normalizar': True
        },
        threshold=0.5,
        salvar_graficos_ks=True,
        output_dir_graficos="output/ks_por_etapa",
    )

    # PARTE 2: Treinar Gradient Boosting
    logger.info("\n" + "="*70)
    logger.info("PARTE 2: GRADIENT BOOSTING")
    logger.info("="*70)

    df_resumo_gb = treinamento.executar_treinamento_incremental(
        tipo_modelo='gradient_boosting',
        params_modelo={
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 0.8
        },
        threshold=0.5,
        salvar_graficos_ks=True,
        output_dir_graficos="output/ks_por_etapa",
    )

    # Opção 3: Comparar ambos os modelos
    logger.info("\n" + "="*70)
    logger.info("OPÇÃO 3: COMPARAÇÃO DE MODELOS")
    logger.info("="*70)

    # Comparação de modelos
    resultados_comparacao = treinamento.comparar_modelos(
        df_resumo_lr=df_resumo_lr,
        df_resumo_gb=df_resumo_gb
    )


    # 4. Salvar resultados
    logger.info("\n>>> ETAPA 4: SALVANDO RESULTADOS")
    treinamento.salvar_resultados(
        caminho="output/resultados_incrementais.csv"
    )

    # Salvar resumos individuais
    df_resumo_lr.to_csv("output/resumo_regressao_logistica.csv", index=False)
    df_resumo_gb.to_csv("output/resumo_gradient_boosting.csv", index=False)
    resultados_comparacao['comparacao'].to_csv(
        "output/comparacao_modelos.csv",
        index=False
    )
    graficos_ks = treinamento.graficos_ks

    logger.success("\n" + "="*70)
    logger.success("PIPELINE CONCLUÍDO COM SUCESSO!")
    logger.success("="*70)
    logger.info("\nArquivos gerados:")
    logger.info("  - output/resultados_incrementais.csv")
    logger.info("  - output/resumo_regressao_logistica.csv")
    logger.info("  - output/resumo_gradient_boosting.csv")
    logger.info("  - output/comparacao_modelos.csv")
    logger.info("  - output/ks_por_etapa/*.png")
    for nome_modelo, caminhos_por_etapa in graficos_ks.items():
        for etapa, caminho_grafico in caminhos_por_etapa.items():
            logger.info(f"  - {caminho_grafico} ({nome_modelo}, etapa {etapa})")

    # Retornar objetos para uso interativo
    return {
        'spark': spark,
        'tables': tabelas,
        'tabelas_treino_oot': tabelas_treino_oot,
        'tables_gc': tabelas_gc,
        'metadata': metadata,
        'treinamento': treinamento,
        'resultados': resultados_comparacao,
        'graficos_ks': graficos_ks,
    }

if __name__ == "__main__":
    resultados = main()
