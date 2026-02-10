# Hackathon: Risco de Crédito
![Descrição da Imagem](docs/Capa.png)

# Problema de Negócio
Uma grande empresa do setor de telecomunicações busca expandir sua base de clientes em planos de maior valor (Controle) de forma sustentável. O desafio central é equilibrar o crescimento da carteira com uma gestão de risco eficiente, focando na migração de usuários do segmento Pré-pago, que representam a principal porta de entrada de novos clientes.

O objetivo deste projeto é desenvolver um Modelo de Behavior baseado na Visão Unificada do Cliente (CPF) para estimar a probabilidade de inadimplência durante esse processo de migração.

Objetivos Principais:
- Identificar perfis de baixo risco: Localizar clientes pré-pagos com maior potencial de adimplência em planos Controle.

- Otimizar Ofertas: Direcionar propostas de forma mais eficiente e personalizada.

- Rentabilidade: Reduzir o risco de crédito e aumentar a saúde financeira da carteira de clientes.

# Arquitetura
```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '18px', 'fontFamily': 'arial'}}}%%
flowchart LR
    %% --- ESTILOS ---
    classDef raw fill:#fdfdfd,stroke:#333,stroke-width:2px,color:#333;
    classDef bronze fill:#cd7f32,stroke:#333,stroke-width:2px,color:white;
    classDef silver fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b;
    classDef gold fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:#9a7d0a,font-weight:bold;
    classDef model fill:#1168bd,stroke:#333,stroke-width:2px,color:white,font-weight:bold;

    %% --- NODOS ---
    F[("FONTE (CSV)<br/>Cadastral, Bureau,<br/>Telco, Pagos,<br/>Recarga + 11 Dims")]:::raw
    B[("BRONZE<br/>(Raw Data)")]:::bronze
    S[("SILVER (Parquet)<br/>Books e<br/>Dimensionais")]:::silver
    G[("GOLD (ABT)<br/>6 Steps<br/>Incrementais")]:::gold
    M(("MODELO .pkl")):::model
    
    %% --- ESPAÇO PARA O CONTROLE DO GITHUB ---
    End(( ))
    style End fill:none,stroke:none

    %% --- CONEXÕES ---
    F ==> |"Ingestão"| B
    B ==> |"Tipagem"| S
    S ==> |"Features"| G
    G ==> |"Treino"| M
    M --- End
```
