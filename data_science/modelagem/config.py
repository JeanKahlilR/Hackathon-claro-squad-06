"""
Configurações e mapeamento de features por fonte de informação.
"""

# Configurações de SAFRA para split temporal
SAFRA_OOT = [202502, 202503]  # Fevereiro e Março de 2025 (benchmark KS = 33.1

# Benchmark de referência
BENCHMARK_KS = 33.1

# Identificação do grupo controle (6º e 7º dígitos do CPF)
GRUPO_CONTROLE_DIGITOS = ["ZZ", "ZX"]

# Target
TARGET_COLUMN = "FPD"

# Colunas que não devem ser usadas como features
COLUNAS_EXCLUIR = [
    "NUM_CPF",           # Identificador
    "ID_UNICO",          # Identificador
    "SAFRA",             # Usado para split temporal
    "FPD",               # Target
    "FLAG_INSTALACAO",   # Usado para análise swap-in/out
    "GRUPO_CONTROLE",    # Usado para análise de grupo controle
    "PROD",              # Produto (pode ser usado depois se necessário)
    "flag_mig2"          # Flag de migração (pode ser usado depois se necessário)
]

FEATURES_POR_ETAPA: Dict[int, Dict[str, List[str]]] = {
    1: {
        "nome": "Score 1",
        "features": ["SCORE_01"]
    },
    2: {
        "nome": "Score 1 + Score 2",
        "features": ["SCORE_01", "SCORE_02"]
    },
    3: {
        "nome": "Scores + Telco",
        "features": [
            "SCORE_01", "SCORE_02",
            # Features Telco (var_26 a var_93)
            "var_26", "var_27", "var_28", "var_29", "var_30",
            "var_31", "var_32", "var_33", "var_34", "var_35",
            "var_36", "var_37", "var_38", "var_39", "var_40",
            "var_41", "var_42", "var_43", "var_44", "var_45",
            "var_46", "var_47", "var_48", "var_49", "var_50",
            "var_51", "var_52", "var_53", "var_54", "var_55",
            "var_56", "var_57", "var_58", "var_59", "var_60",
            "var_61", "var_62", "var_63", "var_64", "var_65",
            "var_66", "var_67", "var_68", "var_69", "var_70",
            "var_71", "var_72", "var_73", "var_74", "var_75",
            "var_76", "var_77", "var_78", "var_79", "var_80",
            "var_81", "var_82", "var_83", "var_84", "var_85",
            "var_86", "var_87", "var_88", "var_89", "var_90",
            "var_91", "var_92", "var_93",
            "var26_a_var77_MISSING",
            "var26_a_var93_NAO_SE_APLICA"
        ]
    },
    4: {
        "nome": "Scores + Telco + Cadastrais",
        "features": [
            "SCORE_01", "SCORE_02",
            # Features Telco
            "var_26", "var_27", "var_28", "var_29", "var_30",
            "var_31", "var_32", "var_33", "var_34", "var_35",
            "var_36", "var_37", "var_38", "var_39", "var_40",
            "var_41", "var_42", "var_43", "var_44", "var_45",
            "var_46", "var_47", "var_48", "var_49", "var_50",
            "var_51", "var_52", "var_53", "var_54", "var_55",
            "var_56", "var_57", "var_58", "var_59", "var_60",
            "var_61", "var_62", "var_63", "var_64", "var_65",
            "var_66", "var_67", "var_68", "var_69", "var_70",
            "var_71", "var_72", "var_73", "var_74", "var_75",
            "var_76", "var_77", "var_78", "var_79", "var_80",
            "var_81", "var_82", "var_83", "var_84", "var_85",
            "var_86", "var_87", "var_88", "var_89", "var_90",
            "var_91", "var_92", "var_93",
            "var26_a_var77_MISSING",
            "var26_a_var93_NAO_SE_APLICA",
            # Features Cadastrais
            "IDADE", "IDADE_MISSING",
            "CMV", "NET", "DTH",
            "NUM_STATUSRF",
            "STATUSRF_REGULAR", "STATUSRF_PENDENTE DE REGULARIZACAO", "STATUSRF_SEM_STATUS",
            "STATUSRF_TITULAR FALECIDO", "STATUSRF_TITULAR CANCELADO", "STATUSRF_CANCELADA",
            "var_07_MONETARIO", "var_07_MONETARIO_MISSING",
            "FUNC_PUBL", "SALARIO_FUNC_PUBL", "SALARIO_FUNC_PUBL_MISSING",
            "EMPR_DIRETOR", "VALOR_EMPR_DIRETOR", "VALOR_EMPR_DIRETOR_MISSING",
            "BOLSA_FAMILIA", "BENEFICIO_BOLSA_FAMILIA", "BENEFICIO_BOLSA_FAMILIA_MISSING",
            "SAFRA_BOLSA_FAMILIA_MISSING",
            "APOSENTADO", "NUMERO_APOSENTADO", "NUMERO_APOSENTADO_MISSING",
            "AUXILLIO_EMERGENCIAL", "MESES_AUXILIO_EMERGENCIAL", "MESES_AUXILIO_EMERGENCIAL_MISSING",
            "FUNC_PRIVADO", "DATA_FUNC_PRIVADO_MISSING",
            "STATUS_FUNC_PRIVADO_ADMITIDO", "STATUS_FUNC_PRIVADO_DISPENSADO", "STATUS_FUNC_PRIVADO_NAO_APLICAVEL",
            "var_02", "var_02_missing",
            "var_03", "var_03_MISSING",
            "var_05", "var_05_MISSING"
        ]
    },
    5: {
        "nome": "All Features + Book Recarga",
        "features": [
            "SCORE_01", "SCORE_02",
            # Features Telco
            "var_26", "var_27", "var_28", "var_29", "var_30",
            "var_31", "var_32", "var_33", "var_34", "var_35",
            "var_36", "var_37", "var_38", "var_39", "var_40",
            "var_41", "var_42", "var_43", "var_44", "var_45",
            "var_46", "var_47", "var_48", "var_49", "var_50",
            "var_51", "var_52", "var_53", "var_54", "var_55",
            "var_56", "var_57", "var_58", "var_59", "var_60",
            "var_61", "var_62", "var_63", "var_64", "var_65",
            "var_66", "var_67", "var_68", "var_69", "var_70",
            "var_71", "var_72", "var_73", "var_74", "var_75",
            "var_76", "var_77", "var_78", "var_79", "var_80",
            "var_81", "var_82", "var_83", "var_84", "var_85",
            "var_86", "var_87", "var_88", "var_89", "var_90",
            "var_91", "var_92", "var_93",
            "var26_a_var77_MISSING",
            "var26_a_var93_NAO_SE_APLICA",
            # Features Cadastrais
            "IDADE", "IDADE_MISSING",
            "CMV", "NET", "DTH",
            "NUM_STATUSRF",
            "STATUSRF_REGULAR", "STATUSRF_PENDENTE DE REGULARIZACAO", "STATUSRF_SEM_STATUS",
            "STATUSRF_TITULAR FALECIDO", "STATUSRF_TITULAR CANCELADO", "STATUSRF_CANCELADA",
            "var_07_MONETARIO", "var_07_MONETARIO_MISSING",
            "FUNC_PUBL", "SALARIO_FUNC_PUBL", "SALARIO_FUNC_PUBL_MISSING",
            "EMPR_DIRETOR", "VALOR_EMPR_DIRETOR", "VALOR_EMPR_DIRETOR_MISSING",
            "BOLSA_FAMILIA", "BENEFICIO_BOLSA_FAMILIA", "BENEFICIO_BOLSA_FAMILIA_MISSING",
            "SAFRA_BOLSA_FAMILIA_MISSING",
            "APOSENTADO", "NUMERO_APOSENTADO", "NUMERO_APOSENTADO_MISSING",
            "AUXILLIO_EMERGENCIAL", "MESES_AUXILIO_EMERGENCIAL", "MESES_AUXILIO_EMERGENCIAL_MISSING",
            "FUNC_PRIVADO", "DATA_FUNC_PRIVADO_MISSING",
            "STATUS_FUNC_PRIVADO_ADMITIDO", "STATUS_FUNC_PRIVADO_DISPENSADO", "STATUS_FUNC_PRIVADO_NAO_APLICAVEL",
            "var_02", "var_02_missing",
            "var_03", "var_03_MISSING",
            "var_05", "var_05_MISSING",
            # Features Book Recarga
            "MIG_Aquisicao", "MIG_PRE", "MIG_FLEX", "MIG_SEM_MIGRACAO",
            "REC_TOTAL_VALOR_HIST", "REC_QTD_RECARGAS_HIST", "REC_MEDIA_VALOR_HIST",
            "REC_MAX_VALOR_HIST", "QTD_PRE_PAGO", "QTD_CONTROLE", "QTD_POS_PAGO",
            "QTD_OUTROS", "QTD_SOS", "TOTAL_VALOR_SOS", "MEDIA_SOS", "REC_DIAS_DESDE_ULTIMA"
        ]
    },
    6: {
        "nome": "All Features + Book Recarga + Book Pagamento",
        "features": [
            "SCORE_01", "SCORE_02",
            # Features Telco
            "var_26", "var_27", "var_28", "var_29", "var_30",
            "var_31", "var_32", "var_33", "var_34", "var_35",
            "var_36", "var_37", "var_38", "var_39", "var_40",
            "var_41", "var_42", "var_43", "var_44", "var_45",
            "var_46", "var_47", "var_48", "var_49", "var_50",
            "var_51", "var_52", "var_53", "var_54", "var_55",
            "var_56", "var_57", "var_58", "var_59", "var_60",
            "var_61", "var_62", "var_63", "var_64", "var_65",
            "var_66", "var_67", "var_68", "var_69", "var_70",
            "var_71", "var_72", "var_73", "var_74", "var_75",
            "var_76", "var_77", "var_78", "var_79", "var_80",
            "var_81", "var_82", "var_83", "var_84", "var_85",
            "var_86", "var_87", "var_88", "var_89", "var_90",
            "var_91", "var_92", "var_93",
            "var26_a_var77_MISSING",
            "var26_a_var93_NAO_SE_APLICA",
            # Features Cadastrais
            "IDADE", "IDADE_MISSING",
            "CMV", "NET", "DTH",
            "NUM_STATUSRF",
            "STATUSRF_REGULAR", "STATUSRF_PENDENTE DE REGULARIZACAO", "STATUSRF_SEM_STATUS",
            "STATUSRF_TITULAR FALECIDO", "STATUSRF_TITULAR CANCELADO", "STATUSRF_CANCELADA",
            "var_07_MONETARIO", "var_07_MONETARIO_MISSING",
            "FUNC_PUBL", "SALARIO_FUNC_PUBL", "SALARIO_FUNC_PUBL_MISSING",
            "EMPR_DIRETOR", "VALOR_EMPR_DIRETOR", "VALOR_EMPR_DIRETOR_MISSING",
            "BOLSA_FAMILIA", "BENEFICIO_BOLSA_FAMILIA", "BENEFICIO_BOLSA_FAMILIA_MISSING",
            "SAFRA_BOLSA_FAMILIA_MISSING",
            "APOSENTADO", "NUMERO_APOSENTADO", "NUMERO_APOSENTADO_MISSING",
            "AUXILLIO_EMERGENCIAL", "MESES_AUXILIO_EMERGENCIAL", "MESES_AUXILIO_EMERGENCIAL_MISSING",
            "FUNC_PRIVADO", "DATA_FUNC_PRIVADO_MISSING",
            "STATUS_FUNC_PRIVADO_ADMITIDO", "STATUS_FUNC_PRIVADO_DISPENSADO", "STATUS_FUNC_PRIVADO_NAO_APLICAVEL",
            "var_02", "var_02_missing",
            "var_03", "var_03_MISSING",
            "var_05", "var_05_MISSING",
            # Features Book Recarga
            "MIG_Aquisicao", "MIG_PRE", "MIG_FLEX", "MIG_SEM_MIGRACAO",
            "REC_TOTAL_VALOR_HIST", "REC_QTD_RECARGAS_HIST", "REC_MEDIA_VALOR_HIST",
            "REC_MAX_VALOR_HIST", "QTD_PRE_PAGO", "QTD_CONTROLE", "QTD_POS_PAGO",
            "QTD_OUTROS", "QTD_SOS", "TOTAL_VALOR_SOS", "MEDIA_SOS", "REC_DIAS_DESDE_ULTIMA",
            # Features Book Pagamento
            "PAG_QTD_FATURAS_HIST", "PAG_TOTAL_PAGO_HIST", "PAG_MEDIA_PAGO_HIST",
            "PAG_MAX_DIAS_ATRASO_HIST", "PAG_MEDIA_DIAS_ATRASO_HIST",
            "PAG_QTD_PAGAMENTOS_ATRASADOS", "PAG_DESVIO_PADRAO_ATRASO",
            "PAG_DIAS_DESDE_ULTIMO_PAG", "PAG_HISTORICO_MISSING"
        ]
    }
}

def obter_features_etapa(etapa: int) -> List[str]:
    """
    Retorna a lista de features para uma etapa específica.
    
    Parameters
    ----------
    etapa : int
        Número da etapa (1, 2, 3 ou 4)
    
    Returns
    -------
    List[str]
        Lista de nomes de features
    
    Examples
    --------
    >>> features = obter_features_etapa(1)
    >>> print(features)
    ['SCORE_01']
    """
    if etapa not in FEATURES_POR_ETAPA:
        raise ValueError(f"Etapa {etapa} inválida. Etapas disponíveis: {list(FEATURES_POR_ETAPA.keys())}")
    
    return FEATURES_POR_ETAPA[etapa]["features"]


def obter_nome_etapa(etapa: int) -> str:
    """
    Retorna o nome descritivo de uma etapa.
    
    Parameters
    ----------
    etapa : int
        Número da etapa (1, 2, 3 ou 4)
    
    Returns
    -------
    str
        Nome descritivo da etapa
    
    Examples
    --------
    >>> nome = obter_nome_etapa(1)
    >>> print(nome)
    'Score 1'
    """
    if etapa not in FEATURES_POR_ETAPA:
        raise ValueError(f"Etapa {etapa} inválida. Etapas disponíveis: {list(FEATURES_POR_ETAPA.keys())}")
    
    return FEATURES_POR_ETAPA[etapa]["nome"]


def validar_features_disponiveis(df_columns: List[str], etapa: int) -> bool:
    """
    Valida se todas as features necessárias para uma etapa estão disponíveis no DataFrame.
    
    Parameters
    ----------
    df_columns : List[str]
        Lista de colunas disponíveis no DataFrame
    etapa : int
        Número da etapa a validar
    
    Returns
    -------
    bool
        True se todas as features estão disponíveis, False caso contrário
    
    Examples
    --------
    >>> colunas = ['SCORE_01', 'SCORE_02', 'FPD']
    >>> validar_features_disponiveis(colunas, 1)
    True
    """
    features_necessarias = obter_features_etapa(etapa)
    features_faltantes = [f for f in features_necessarias if f not in df_columns]
    
    if features_faltantes:
        logger.warning(f"Features faltantes para etapa {etapa}: {features_faltantes}")
        return False
    
    logger.success(f"Todas as features da etapa {etapa} estão disponíveis")
    return True


def listar_etapas_disponiveis() -> List[int]:
    """
    Retorna lista de etapas disponíveis.
    
    Returns
    -------
    List[int]
        Lista de números de etapas disponíveis
    """
    return sorted(FEATURES_POR_ETAPA.keys())


