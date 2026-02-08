"""
Configuração centralizada do sistema de logging usando Loguru.

Este módulo fornece a configuração padrão de logging, incluindo 
handlers para console e arquivo com rotação automática.
"""

import sys
from pathlib import Path
from loguru import logger


def configure_logger(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_dir: str = "logs",
    rotation: str = "00:00",
    retention: str = "7 days",
    compression: str = "zip",
) -> None:
    """
    Configura o loguru com formatação personalizada e múltiplos destinos.
    
    Esta função configura dois handlers de log:
    1. Console (stdout) - Para visualização em tempo real
    2. Arquivo - Para histórico e auditoria
    
    Parameters
    ----------
    console_level : str
        Nível mínimo de log para o console (default: "INFO")
        Opções: "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    file_level : str
        Nível mínimo de log para arquivo (default: "DEBUG")
    log_dir : str
        Diretório onde os logs serão salvos (default: "logs")
    rotation : str
        Quando criar novo arquivo de log (default: "00:00" = meia-noite)
        Exemplos: "500 MB", "1 week", "10:00"
    retention : str
        Por quanto tempo manter logs antigos (default: "7 days")
        Exemplos: "10 days", "1 month", "1 year"
    compression : str
        Formato de compressão para logs antigos (default: "zip")
        Opções: "zip", "gz", "bz2", "xz", None
    
    Examples
    --------
    Configuração padrão:
    >>> configure_logger()
    
    Configuração customizada:
    >>> configure_logger(
    ...     console_level="DEBUG",
    ...     file_level="INFO",
    ...     retention="30 days"
    ... )
    """
    # Remove todos os handlers padrão do loguru
    # Por padrão, loguru já vem com um handler para stderr
    # Removemos para ter controle total sobre a configuração
    logger.remove()
    
    # Garante que o diretório de logs existe
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # ========== CONFIGURAÇÃO 1: LOG NO CONSOLE (stdout) ==========
    logger.add(
        sys.stdout,  # Destino: saída padrão (console/terminal)
        
        # Formato com cores usando tags HTML-like do loguru:
        # <green>...</green> = texto verde
        # <level>...</level> = cor automática baseada no nível (INFO=azul, ERROR=vermelho, etc.)
        # <cyan>...</cyan> = texto ciano
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        
        # Campos disponíveis no formato:
        # {time} = timestamp do log
        # {level} = nível do log (INFO, DEBUG, ERROR, etc.)
        # {level: <8} = nível alinhado à esquerda com 8 caracteres
        # {name} = nome do módulo que gerou o log
        # {function} = nome da função que gerou o log
        # {message} = mensagem do log
        
        level=console_level,  # Nível mínimo configurável
        colorize=True,  # Ativa colorização no terminal
    )

    # ========== CONFIGURAÇÃO 2: LOG EM ARQUIVO ==========
    logger.add(
        log_path / "data_loading_{time:YYYY-MM-DD}.log",  # Nome do arquivo com data dinâmica
        # Exemplo: logs/data_loading_2024-01-15.log
        
        # Formato SEM cores (arquivos de texto não suportam cores)
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        # {line} = número da linha que gerou o log (útil para debug)
        
        level=file_level,  # Nível mínimo configurável
        # Arquivo geralmente tem mais detalhes que o console
        
        rotation=rotation,  # Rotação configurável
        # Exemplos: "00:00" (meia-noite), "500 MB", "1 week"
        
        retention=retention,  # Retenção configurável
        # Exemplos: "7 days", "1 month", "1 year"
        
        compression=compression,  # Compressão configurável
        # Opções: "zip", "gz", "bz2", "xz", None
    )
    
    logger.info(f"Logger configurado: console={console_level}, arquivo={file_level}, diretório={log_dir}")


def get_logger():
    """
    Retorna a instância do logger configurado.
    
    Returns
    -------
    logger
        Instância do loguru logger
        
    Examples
    --------
    >>> from utils.config_logger import get_logger
    >>> logger = get_logger()
    >>> logger.info("Mensagem de log")
    """
    return logger
