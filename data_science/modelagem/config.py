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