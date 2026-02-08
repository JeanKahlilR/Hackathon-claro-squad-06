"""
API publica do pacote de processamento de dados.
"""

from .carregar_dados import load_all_sources
from .spark_session import make_spark

__all__ = ["make_spark", "load_all_sources"]
