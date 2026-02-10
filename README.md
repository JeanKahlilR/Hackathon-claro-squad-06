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
## Detalhamento das Camadas

**1 - Bronze Layer (Ingestão)**
<br>
`Objetivo`: Armazenamento dos dados brutos em formato .parquet sem alterações de negócio.

- Fontes: Dados internos de telecom (uso, recarga, fatura) e dados externos de Bureau (Score, Negativação).
- Ações: Leitura otimizada (Lazy Evaluation) e validação inicial de schema.

**2 - Silver Layer (Refinamento e Engenharia)**
<br>
`Objetivo:` Limpeza, padronização e criação de Books de Variáveis (Feature Engineering). 

- Book de Pagamento:
    - Cálculo de dias de atraso (Vencimento vs Pagamento Real).
    - Criação de flags de atraso histórico.
    - Regra de Negócio: Filtro para remover pagamentos futuros à data da safra (prevenção de Data Leakage).

- Book de Recarga:
    - Agregações (Soma, Média, Máximo) de valores recarregados.
    - Clusterização de comportamento (Pré-pago vs Controle).
    - Cálculo de Recência (dias desde a última recarga).
    - Regra de Negócio: Filtro para remover pagamentos futuros à data da safra (prevenção de Data Leakage).

- Módulo Cadastral & Telco:
    - Tratamento de variáveis anonimizadas (var_X).
    - Normalização de categorias de emprego e renda.

**3 - Gold Layer (Agregação Final)**
<br>
`Objetivo`: Consolidação da Visão 360º do Cliente (ABT) pronta para modelagem.

- Estratégia de Join: Utilização de Left Joins partindo da população alvo (Safra/Cliente).
    - Justificativa: Preservar clientes sem histórico (Recarga/Pagamento) na base, pois a "ausência de informação" é um preditor de risco relevante.

- Tratamento de Nulos:
    - Sentinela -1: Aplicado para missings gerados pelo Join (ex: cliente nunca recarregou).
    - Sentinela -4: Mantido para missings originais da fonte de dados.

- Flags de Metadados: Criação de colunas binárias (ex: PAG_HISTORICO_MISSING) para sinalizar explicitamente ao modelo a origem da ausência do dado.

**Notebooks**
<br>
[Tratamento Inicial](notebooks/Broze-Silver_EMR.ipynb)
<br>
[Baseline](notebooks/Baseline_Colab.ipynb )

**Tecnologias Utilizadas**
<br>
![Python](https://img.shields.io/badge/python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
<br>
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-PySpark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
<br>
![Parquet](https://img.shields.io/badge/Data-Parquet-4E9A06?style=for-the-badge&logo=databricks&logoColor=white)
<br>
![Google Colab](https://img.shields.io/badge/Colab-Research-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)


## Arquitetura de Dados Proposta (AWS Cloud)
![Descrição da Imagem](docs/Arquitetura_AWS.webp)

A arquitetura proposta foi desenhada seguindo os pilares do `AWS Well-Architected Framework`, priorizando uma abordagem `Serverless (sem servidor)`. Isso garante que o foco do time esteja na lógica de negócio e na ciência de dados, delegando o gerenciamento de infraestrutura para a AWS, o que resulta em maior eficiência operacional, escalabilidade automática e otimização de custos.

O fluxo de dados segue a arquitetura de referência `"Medalhão" (Bronze/Silver/Gold)`, garantindo a qualidade, rastreabilidade e governança dos dados desde a ingestão até a modelagem preditiva.

**1 - Ingestão e Orquestração (Ingestion Layer)**

- `Ferramentas`: AWS Glue Python Shell, Amazon EventBridge, AWS Secrets Manager.
- `Como funciona`: O processo inicia-se com uma rotina agendada no Amazon EventBridge, que dispara um script de ingestão no AWS Glue Python Shell. Este script conecta-se ao Google Drive, baixa os arquivos de origem e os deposita na camada crua. **O encadeamento das etapas subsequentes é gerenciado nativamente, garantindo que a transformação só inicie após o sucesso da ingestão.**
- `Motivo da Escolha`:
    - `AWS Glue Python Shell`: Escolhido em detrimento do AWS Lambda para eliminar o risco de timeout (limite de 15 min) e estouro de memória ao processar arquivos volumosos, garantindo robustez para grandes cargas de dados.
    - `Amazon EventBridge`: Desacopla o agendamento da execução, permitindo uma arquitetura orientada a eventos.
    - `AWS Secrets Manager`: Elimina credenciais hardcoded, garantindo que chaves de API trafeguem criptografadas e em conformidade com normas de segurança.

**2 - Armazenamento e Data Lake (Storage Layer)**

- `Ferramentas`: Amazon S3 (Buckets Raw, Silver, Gold).
- `Como funciona`: O Amazon S3 atua como o repositório central com três estágios lógicos (Bronze/Silver/Gold).
    - `Raw`: Dados brutos e imutáveis (Auditabilidade).
    - `Silver`: Dados limpos e convertidos para Parquet com compressão Snappy, otimizando performance de I/O.
    - `Gold`: Dados agregados (Feature Store) prontos para o modelo.
- `Motivo da Escolha`: Além da durabilidade de "11 noves", o S3 permite aplicar **Lifecycle Policies (Políticas de Ciclo de Vida)** para mover dados antigos para classes de armazenamento mais baratas (Glacier) automaticamente, atendendo ao pilar de Otimização de Custos.

**3 - Processamento e Transformação (Processing Layer)**

- `Ferramentas`: AWS Glue Jobs (Spark).
- `Como funciona`: Jobs Spark processam os dados de forma distribuída.
- `Motivo da Escolha`: O AWS Glue é totalmente gerenciado. A escolha pelo Spark permite processamento paralelo massivo, essencial para recalcular o score de milhões de clientes em minutos, garantindo `escalabilidade horizontal` transparente.

**4 - Governança e Catálogo (Data Governance)**

- `Ferramentas`: AWS Glue Crawler, AWS Glue Data Catalog, AWS Lake Formation.
- `Como funciona`: O Crawler infere esquemas automaticamente (Schema Evolution) e o Lake Formation gerencia o acesso.
- `Motivo da Escolha`:
    - `AWS Lake Formation`: É o diferencial de segurança do projeto. Ele permite implementar **segurança a nível de linha e coluna (ex: mascarar o CPF e Renda para analistas júnior)**, garantindo conformidade estrita com a **LGPD** sem necessidade de duplicar dados anonimizados.

**5 - Consumo, Analytics e Machine Learning**

- `Ferramentas`: Amazon Athena, Amazon SageMaker.
- `Como funciona`: O Athena permite SQL ad-hoc e o SageMaker gerencia o ciclo de vida do modelo.
- `Motivo da Escolha`:
    - `Amazon Athena`: Custo zero de infraestrutura fixa, pagando apenas por terabyte scaneado.
    - `Amazon SageMaker`: Além do treinamento, utilizamos o **SageMaker Model Registry** para versionar os artefatos do modelo (.pkl), garantindo que saibamos exatamente qual versão do modelo gerou qual score, assegurando a reprodutibilidade científica.

**6 - Observabilidade e Operações (Ops & Monitoring)**

- `Ferramentas`: Amazon CloudWatch, Amazon SNS.
- `Como funciona`: Monitoramento centralizado de logs, métricas e alarmes.
`Motivo da Escolha`: Atende ao pilar de Excelência Operacional. A configuração de alarmes no CloudWatch integrados ao SNS permite uma abordagem **proativa**: o time é notificado sobre falhas de pipeline ou degradação de performance antes que o negócio seja impactado.


