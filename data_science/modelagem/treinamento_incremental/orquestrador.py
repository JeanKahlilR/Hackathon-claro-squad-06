"""
Arquivo orquestrador da pipeline de treinamento incremental
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..config import BENCHMARK_KS, COLUNAS_EXCLUIR, TARGET_COLUMN, listar_etapas_disponiveis, obter_nome_etapa
from ..modelos import ModeloBase
from .data_builder import obter_splits_etapa, selecionar_para_pandas
from .evaluator import avaliar_resultado_etapa
from .reporting import (
    criar_comparacao_modelos,
    criar_resumo_incremental,
    resultados_para_dataframe,
    salvar_grafico_ks_etapa,
)
from .trainer import criar_modelo, treinar_e_prever_oot
from .types import EstadoIncremental, ResultadoEtapa, TablesSplited





class TreinamentoIncremental:
    """
    Classe para gerenciar treinamento incremental de modelos.
    """

    def __init__(
        self,
        tables_splited: TablesSplited,
        etapas: Optional[List[int]] = None,
    ):
        """
        Inicializa o orquestrador do treinamento incremental.

        Parameters
        ----------
        tables_splited : TablesSplited
            Dicionario com tabelas Spark separadas em treino e OOT por etapa.
        etapas : Optional[List[int]], optional
            Lista de etapas a executar. Se None, usa todas as etapas disponiveis.
        """
        # Entrada principal: dicionario de tabelas Spark ja separadas em treino/OOT.
        self.tables_splited = tables_splited
        self.etapas = etapas if etapas is not None else listar_etapas_disponiveis()
        # Estado mutavel da execucao (resultados e modelos).
        self.estado = EstadoIncremental()

        logger.info("=" * 70)
        logger.info("TREINAMENTO INCREMENTAL INICIALIZADO")
        logger.info("=" * 70)
        logger.info(f"Numero de tabelas: {len(tables_splited)}")
        logger.info(f"Tabelas disponiveis: {list(tables_splited.keys())}")
        logger.info(f"Etapas a executar: {self.etapas}")
        logger.info(f"Benchmark KS: {BENCHMARK_KS:.2f}")
        logger.info("=" * 70)

    @property
    def resultados(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna os resultados acumulados por modelo/etapa.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dicionario com metricas e artefatos de avaliacao por execucao.
        """
        return self.estado.resultados

    @property
    def modelos_treinados(self) -> Dict[str, Any]:
        """
        Retorna os modelos treinados no fluxo incremental.

        Returns
        -------
        Dict[str, Any]
            Dicionario com modelos indexados por chave de etapa e nome.
        """
        return self.estado.modelos_treinados

    @property
    def graficos_ks(self) -> Dict[str, Dict[int, str]]:
        """
        Retorna caminhos de graficos KS por tipo de modelo e etapa.
        """
        return self.estado.graficos_ks

    def _preparar_dados_etapa(
        self,
        etapa: int,
    ) -> Tuple[List[str], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Optional[Any]]:
        """
        Seleciona dados prontos de treino e OOT para uma etapa.

        Parameters
        ----------
        etapa : int
            Numero da etapa.

        Returns
        -------
        Tuple[List[str], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Optional[Any]]
            Tupla com:
            - features validas
            - X treino
            - y treino
            - X OOT
            - y OOT
            - flag de instalacao no OOT (quando disponivel)

        Raises
        ------
        ValueError
            Quando a tabela/colunas obrigatorias nao sao encontradas.
        """
        table_name, df_treino_raw, df_oot_raw = obter_splits_etapa(etapa, self.tables_splited)
        logger.info(f"Tabela da etapa: {table_name}")

        treino_cols = list(df_treino_raw.columns)
        oot_cols = list(df_oot_raw.columns)

        if TARGET_COLUMN not in treino_cols:
            raise ValueError(f"Etapa {etapa}: coluna target ausente no treino ({table_name}): {TARGET_COLUMN}")
        if TARGET_COLUMN not in oot_cols:
            raise ValueError(f"Etapa {etapa}: coluna target ausente no OOT ({table_name}): {TARGET_COLUMN}")

        colunas_excluir = set(COLUNAS_EXCLUIR + ["Flag_OOT", "flag_oot"])
        features = [c for c in treino_cols if c not in colunas_excluir]
        
        if not features:
            raise ValueError(f"Etapa {etapa}: nenhuma feature valida encontrada para treino/OOT em {table_name}")

        logger.info(f"Features selecionadas automaticamente: {len(features)}")

        colunas_treino = [TARGET_COLUMN] + features
        colunas_oot = [TARGET_COLUMN] + features
        if "FLAG_INSTALACAO" in oot_cols:
            colunas_oot.append("FLAG_INSTALACAO")

        df_treino = selecionar_para_pandas(df_treino_raw, colunas_treino)
        df_oot = selecionar_para_pandas(df_oot_raw, colunas_oot)

        x_treino = df_treino[features]
        y_treino = pd.to_numeric(df_treino[TARGET_COLUMN], errors="coerce")
        x_oot = df_oot[features]
        y_oot = pd.to_numeric(df_oot[TARGET_COLUMN], errors="coerce")

        # FLAG_INSTALACAO e opcional e so usada na analise swap-in/swap-out.
        flag_instalacao = None
        if "FLAG_INSTALACAO" in df_oot.columns:
            flag_instalacao = df_oot["FLAG_INSTALACAO"].values

        return features, x_treino, y_treino, x_oot, y_oot, flag_instalacao

    def _treinar_etapa(
        self,
        etapa: int,
        modelo: ModeloBase,
        threshold: float = 0.5,
    ) -> Tuple[ResultadoEtapa, np.ndarray, np.ndarray]:
        """
        Treina e avalia uma etapa especifica do fluxo incremental.

        Parameters
        ----------
        etapa : int
            Numero da etapa incremental.
        modelo : ModeloBase
            Instancia de modelo a ser treinada.
        threshold : float, optional
            Threshold para classificacao binaria na avaliacao.

        Returns
        -------
        Tuple[ResultadoEtapa, np.ndarray, np.ndarray]
            Resultado tipado e vetores usados no grafico KS da etapa:
            - resultado
            - y_true_oot
            - y_pred_proba_oot
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"ETAPA {etapa}: {obter_nome_etapa(etapa)}")
        logger.info("=" * 70)

        features_validas, x_treino, y_treino, x_oot, y_oot, flag_instalacao = self._preparar_dados_etapa(etapa)

        # O treino acontece em pandas/scikit, apos consolidacao incremental em Spark.
        y_pred_proba_oot = treinar_e_prever_oot(
            modelo=modelo,
            x_treino=x_treino,
            y_treino=y_treino,
            x_oot=x_oot,
        )

        resultado = avaliar_resultado_etapa(
            etapa=etapa,
            nome_modelo=modelo.nome,
            y_true=y_oot.values,
            y_pred_proba=y_pred_proba_oot,
            threshold=threshold,
            features=features_validas,
            flag_instalacao=flag_instalacao,
        )

        chave_modelo = f"etapa_{etapa}_{modelo.nome.replace(' ', '_').lower()}"
        self.estado.modelos_treinados[chave_modelo] = modelo
        return resultado, y_oot.values, y_pred_proba_oot

    def executar_treinamento_incremental(
        self,
        tipo_modelo: str = "logistica",
        params_modelo: Optional[Dict] = None,
        threshold: float = 0.5,
        salvar_graficos_ks: bool = True,
        output_dir_graficos: str = "output/ks_por_etapa",
    ) -> pd.DataFrame:
        """
        Executa o treinamento incremental de um unico tipo de modelo.

        Parameters
        ----------
        tipo_modelo : str, optional
            Tipo de modelo a treinar (ex.: "logistica", "gradient_boosting").
        params_modelo : Optional[Dict], optional
            Hiperparametros do modelo.
        threshold : float, optional
            Threshold para classificacao binaria na avaliacao.
        salvar_graficos_ks : bool, optional
            Se deve salvar grafico KS de cada etapa usando `skplt.metrics.plot_ks_statistic`.
        output_dir_graficos : str, optional
            Diretorio de saida dos graficos de KS.

        Returns
        -------
        pd.DataFrame
            DataFrame resumo com metricas por etapa.
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"EXECUTANDO TREINAMENTO INCREMENTAL - {tipo_modelo.upper()}")
        logger.info("=" * 70)

        params_modelo = params_modelo or {}
        resultados_lista = []
        ks_anterior = 0.0
        graficos_tipo_modelo: Dict[int, str] = {}

        for etapa in self.etapas:
            # Cada etapa e independente: treino e avaliacao usam apenas a propria tabela da etapa.
            modelo = criar_modelo(tipo_modelo=tipo_modelo, etapa=etapa, params_modelo=params_modelo)
            resultado, y_true_oot, y_pred_proba_oot = self._treinar_etapa(etapa, modelo, threshold)

            ks_atual = resultado["ks"]
            delta_ks = ks_atual - ks_anterior
            resultado["delta_ks"] = delta_ks
            resultado["ks_anterior"] = ks_anterior
            resultado["y_true_oot"] = y_true_oot
            resultado["y_pred_proba_oot"] = y_pred_proba_oot

            if salvar_graficos_ks:
                caminho_grafico = salvar_grafico_ks_etapa(
                    y_true=y_true_oot,
                    y_pred_proba=y_pred_proba_oot,
                    nome_modelo=tipo_modelo,
                    etapa=etapa,
                    output_dir=output_dir_graficos,
                )
                graficos_tipo_modelo[etapa] = caminho_grafico

            logger.info("\n" + "-" * 70)
            logger.info(f"GANHO INCREMENTAL - ETAPA {etapa}")
            logger.info("-" * 70)
            logger.info(f"KS Anterior: {ks_anterior:.2f}")
            logger.info(f"KS Atual: {ks_atual:.2f}")
            logger.info(f"Delta KS: {delta_ks:+.2f}")
            logger.info(f"Delta vs Benchmark: {ks_atual - BENCHMARK_KS:+.2f}")
            logger.info("-" * 70)

            resultados_lista.append(resultado)
            ks_anterior = ks_atual
            self.estado.resultados[f"{tipo_modelo}_etapa_{etapa}"] = resultado

        if graficos_tipo_modelo:
            self.estado.graficos_ks[tipo_modelo] = graficos_tipo_modelo

        df_resumo = criar_resumo_incremental(resultados_lista)

        logger.info("\n" + "=" * 70)
        logger.info("RESUMO DO TREINAMENTO INCREMENTAL")
        logger.info("=" * 70)
        logger.info(f"\n{df_resumo.to_string(index=False)}")
        logger.info("=" * 70)
        return df_resumo

    def salvar_graficos_ks(
        self,
        output_dir_graficos: str = "output/ks_por_etapa",
    ) -> Dict[str, Dict[int, str]]:
        """
        Salva graficos de KS por etapa a partir dos resultados em memoria.

        Essa funcao usa os vetores `y_true_oot` e `y_pred_proba_oot` guardados
        durante `executar_treinamento_incremental`.
        """
        output_dir = Path(output_dir_graficos)
        output_dir.mkdir(parents=True, exist_ok=True)

        graficos_gerados: Dict[str, Dict[int, str]] = {}
        for chave, resultado in self.estado.resultados.items():
            y_true = resultado.get("y_true_oot")
            y_pred_proba = resultado.get("y_pred_proba_oot")
            etapa = resultado.get("etapa")

            if y_true is None or y_pred_proba is None or etapa is None:
                continue

            tipo_modelo = chave.split("_etapa_")[0]
            caminho = salvar_grafico_ks_etapa(
                y_true=y_true,
                y_pred_proba=y_pred_proba,
                nome_modelo=tipo_modelo,
                etapa=int(etapa),
                output_dir=str(output_dir),
            )

            if tipo_modelo not in graficos_gerados:
                graficos_gerados[tipo_modelo] = {}
            graficos_gerados[tipo_modelo][int(etapa)] = caminho

        if graficos_gerados:
            self.estado.graficos_ks.update(graficos_gerados)
        else:
            logger.warning("Nenhum grafico KS foi gerado. Rode o treinamento incremental antes.")

        return self.estado.graficos_ks

    def salvar_resultados(self, caminho: str = "data_science/output/resultados_incrementais.csv") -> None:
        """
        Salva resultados acumulados em arquivo CSV.

        Parameters
        ----------
        caminho : str, optional
            Caminho do arquivo CSV de saida.
        """
        if not self.estado.resultados:
            logger.warning("Nenhum resultado para salvar")
            return

        df_resultados = resultados_para_dataframe(self.estado.resultados)
        df_resultados.to_csv(caminho, index=False)
        logger.success(f"Resultados salvos em: {caminho}")

    def comparar_modelos(
        self,
        params_lr: Optional[Dict] = None,
        params_gb: Optional[Dict] = None,
        df_resumo_lr: Optional[pd.DataFrame] = None,
        df_resumo_gb: Optional[pd.DataFrame] = None,
        threshold: float = 0.5,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compara regressao logistica e gradient boosting por etapa.

        Parameters
        ----------
        params_lr : Optional[Dict], optional
            Hiperparametros para regressao logistica.
        params_gb : Optional[Dict], optional
            Hiperparametros para gradient boosting.
        df_resumo_lr : Optional[pd.DataFrame], optional
            Resumo pre-calculado de regressao logistica para reutilizacao.
        df_resumo_gb : Optional[pd.DataFrame], optional
            Resumo pre-calculado de gradient boosting para reutilizacao.
        threshold : float, optional
            Threshold para classificacao binaria na avaliacao.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dicionario com:
            - resumo de regressao logistica
            - resumo de gradient boosting
            - comparacao consolidada
        """
        logger.info("\n" + "=" * 70)
        logger.info("COMPARACAO: REGRESSAO LOGISTICA vs GRADIENT BOOSTING")
        logger.info("=" * 70)

        if df_resumo_lr is None:
            logger.info("\n>>> TREINANDO REGRESSAO LOGISTICA")
            df_resumo_lr = self.executar_treinamento_incremental(
                tipo_modelo="logistica",
                params_modelo=params_lr,
                threshold=threshold,
            )
        else:
            logger.info("\n>>> REUTILIZANDO RESUMO DE REGRESSAO LOGISTICA")

        if df_resumo_gb is None:
            logger.info("\n>>> TREINANDO GRADIENT BOOSTING")
            df_resumo_gb = self.executar_treinamento_incremental(
                tipo_modelo="gradient_boosting",
                params_modelo=params_gb,
                threshold=threshold,
            )
        else:
            logger.info("\n>>> REUTILIZANDO RESUMO DE GRADIENT BOOSTING")

        df_comparacao = criar_comparacao_modelos(df_resumo_lr, df_resumo_gb)
        logger.info("\n" + "=" * 70)
        logger.info("COMPARACAO DE KS POR ETAPA")
        logger.info("=" * 70)
        logger.info(f"\n{df_comparacao.to_string(index=False)}")
        logger.info("=" * 70)

        return {
            "regressao_logistica": df_resumo_lr,
            "gradient_boosting": df_resumo_gb,
            "comparacao": df_comparacao,
        }
