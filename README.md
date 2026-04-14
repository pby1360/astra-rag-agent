# 🚀 Astra-RAG-Agent
> **Aerospace/Defense Agentic RAG System for MIL-SPEC Technical Verification**
> 
**Astra-RAG-Agent**는 우주항공 및 방위산업의 복잡한 기술 규격(MIL-SPEC, NASA-STD)과 상용 부품(COTS) 데이터시트를 교차 검증하는 지능형 에이전트 시스템입니다. 단순한 텍스트 검색을 넘어 지식 그래프(Knowledge Graph)를 통한 부품 의존성 추론과 에이전트 간의 자율적 논리 검증(Self-Correction)을 통해 엔지니어링 환각(Hallucination) 문제를 해결합니다.
## 🎯 핵심 문제 정의 (Pain Points)
 * **의존성 추론의 한계:** 수많은 규격 문서와 상하위 부품 간의 얽힌 의존성을 일반 RAG(Vector Search)로는 파악 불가.
  * **치명적인 수치 환각:** 규격 수치 오인으로 인한 임무 실패 리스크 존재 (에이전트 기반 교차 검증 필요).
   * **언어 및 도메인 장벽:** 고도의 전문 영문 규격과 국문 질의 사이의 의미론적 단절 발생.
   ## ✨ 핵심 기능 (Key Features)
    * **GraphRAG 기반 의존성 탐색:** Neo4j를 활용하여 부품-규격-상위표준 간의 복잡한 계층 구조를 시각화하고 추적.
     * **Agentic AI 워크플로우:** LangGraph를 통해 분석-검색-검증-보고 단계를 자율적으로 순환하며 수치 무결성 확보.
      * **다국어 하이브리드 검색:** BGE-M3 임베딩을 활용하여 한-영 교차 검색 및 고유 부품 번호(Sparse) 검색 성능 최적화.
       * **MSA 아키텍처:** Spring Boot(비즈니스 코어)와 FastAPI(AI 엔진)의 분리로 확장성 및 신뢰성 확보.
       ## 🏗️ 시스템 아키텍처 (Architecture)
       시스템은 **Spring Boot** 기반의 API Gateway와 **Python/FastAPI** 기반의 AI Inference Server가 연동되는 MSA 구조로 설계되었습니다.
       ```mermaid
       graph TD
           User((Engineer)) -->|Query| SB[Spring Boot Backend]
               SB -->|Async Request| FA[FastAPI AI Engine]
                   
                       subgraph "Agentic Reasoning (LangGraph)"
                               FA --> RA[Requirement Analyzer]
                                       RA --> HR[Hybrid Retriever]
                                               HR --> CV[Cross Verifier]
                                                       CV -->|Insufficient Data| HR
                                                               CV --> RG[Report Generator]
                                                                   end
                                                                       
                                                                           HR -->|Search| RS[(Redis Stack: Vector DB)]
                                                                               HR -->|Trace| NJ[(Neo4j: Graph DB)]

                                                                               ```
                                                                               ## 🛠️ 기술 스택 (Tech Stack)
                                                                               ### Backend & AI Core
                                                                                * **Core:** Java 17, Spring Boot 3.x, Spring WebFlux
                                                                                 * **AI Engine:** Python 3.11, FastAPI, LangGraph, LangChain
                                                                                  * **LLM:** GPT-4o
                                                                                   * **Embedding:** BGE-M3 (Multi-lingual Hybrid Search)
                                                                                   ### Database & Storage
                                                                                    * **Vector DB:** Redis Stack (HNSW Indexing)
                                                                                     * **Graph DB:** Neo4j (Technical Dependency Mapping)
                                                                                      * **Data Parsing:** LlamaParse (Table-aware PDF to Markdown)
                                                                                      ## 📊 지식 그래프 스키마 (Graph Schema)
                                                                                      시스템은 다음과 같은 지식 노드와 관계를 정의하여 복합적인 규격 추론을 수행합니다.
                                                                                       * (Component) -[:TESTED_PER]-> (Test_Method)
                                                                                        * (Standard) -[:INCLUDES]-> (Requirement)
                                                                                         * (Component) -[:PART_OF]-> (System)
                                                                                          * (Standard) -[:REFERENCED_BY]-> (Standard)
                                                                                          ## 🚀 시작하기 (Quick Start)
                                                                                          ### Prerequisites
                                                                                           * Docker & Docker Compose
                                                                                            * OpenAI API Key
                                                                                             * LlamaCloud API Key (for PDF parsing)
                                                                                             ### Setup & Run
                                                                                              1. **Repository Clone**
                                                                                                 ```bash
                                                                                                    git clone https://github.com/your-id/astra-rag-agent.git
                                                                                                       cd astra-rag-agent
                                                                                                          
                                                                                                             ```
                                                                                                              2. **Infrastructure Up (Docker)**
                                                                                                                 ```bash
                                                                                                                    docker-compose up -d  # Redis Stack, Neo4j, PostgreSQL
                                                                                                                       
                                                                                                                          ```
                                                                                                                           3. **AI Core (Python) Run**
                                                                                                                              ```bash
                                                                                                                                 cd ai-core
                                                                                                                                    pip install -r requirements.txt
                                                                                                                                       uvicorn app.main:app --host 0.0.0.0 --port 8000
                                                                                                                                          
                                                                                                                                             ```
                                                                                                                                              4. **Backend (Java) Run**
                                                                                                                                                 ```bash
                                                                                                                                                    cd backend-api
                                                                                                                                                       ./gradlew bootRun
                                                                                                                                                          
                                                                                                                                                             ```
                                                                                                                                                             ## 📂 프로젝트 구조 (Project Structure)
                                                                                                                                                             ```text
                                                                                                                                                             astra-rag-agent/
                                                                                                                                                             ├── ai-core/            # Python FastAPI & LangGraph 에이전트
                                                                                                                                                             │   ├── app/agents/     # 에이전트 노드 및 상태 정의
                                                                                                                                                             │   ├── app/services/   # Vector/Graph DB 연동 로직
                                                                                                                                                             │   └── main.py
                                                                                                                                                             ├── backend-api/        # Spring Boot 비즈니스 로직 및 API Gateway
                                                                                                                                                             ├── data-pipeline/      # LlamaParse 기반 PDF 전처리 스크립트
                                                                                                                                                             ├── docker-compose.yml  # 인프라 구성 (Redis, Neo4j, DB)
                                                                                                                                                             └── docs/               # 설계 문서 및 MIL-STD 샘플 데이터

                                                                                                                                                             ```
                                                                                                                                                             ## 📝 License
                                                                                                                                                             This project is licensed under the MIT License - see the LICENSE file for details.
                                                                                                                                                             **Author:** Byung-yoon Park (Konkuk University Graduate School of AI)
                                                                                                                                                             **Contact:** [Your Email or LinkedIn]
                                                                                                                                                             GitHub 레포지토리의 첫인상을 결정하는 전문적이고 세련된 스타일의 README.md 템플릿입니다. 8년 차 백엔드 개발자의 전문성과 AI 대학원생의 연구 깊이가 동시에 느껴지도록 기술적인 디테일과 아키텍처 설명을 강조하여 작성했습니다.
                                                                                                                                                             # 🚀 Astra-RAG-Agent
                                                                                                                                                             > **Aerospace/Defense Agentic RAG System for MIL-SPEC Technical Verification**
                                                                                                                                                             > 
                                                                                                                                                             **Astra-RAG-Agent**는 우주항공 및 방위산업의 복잡한 기술 규격(MIL-SPEC, NASA-STD)과 상용 부품(COTS) 데이터시트를 교차 검증하는 지능형 에이전트 시스템입니다. 단순한 텍스트 검색을 넘어 지식 그래프(Knowledge Graph)를 통한 부품 의존성 추론과 에이전트 간의 자율적 논리 검증(Self-Correction)을 통해 엔지니어링 환각(Hallucination) 문제를 해결합니다.
                                                                                                                                                             ## 🎯 핵심 문제 정의 (Pain Points)
                                                                                                                                                              * **의존성 추론의 한계:** 수많은 규격 문서와 상하위 부품 간의 얽힌 의존성을 일반 RAG(Vector Search)로는 파악 불가.
                                                                                                                                                               * **치명적인 수치 환각:** 규격 수치 오인으로 인한 임무 실패 리스크 존재 (에이전트 기반 교차 검증 필요).
                                                                                                                                                                * **언어 및 도메인 장벽:** 고도의 전문 영문 규격과 국문 질의 사이의 의미론적 단절 발생.
                                                                                                                                                                ## ✨ 핵심 기능 (Key Features)
                                                                                                                                                                 * **GraphRAG 기반 의존성 탐색:** Neo4j를 활용하여 부품-규격-상위표준 간의 복잡한 계층 구조를 시각화하고 추적.
                                                                                                                                                                  * **Agentic AI 워크플로우:** LangGraph를 통해 분석-검색-검증-보고 단계를 자율적으로 순환하며 수치 무결성 확보.
                                                                                                                                                                   * **다국어 하이브리드 검색:** BGE-M3 임베딩을 활용하여 한-영 교차 검색 및 고유 부품 번호(Sparse) 검색 성능 최적화.
                                                                                                                                                                    * **MSA 아키텍처:** Spring Boot(비즈니스 코어)와 FastAPI(AI 엔진)의 분리로 확장성 및 신뢰성 확보.
                                                                                                                                                                    ## 🏗️ 시스템 아키텍처 (Architecture)
                                                                                                                                                                    시스템은 **Spring Boot** 기반의 API Gateway와 **Python/FastAPI** 기반의 AI Inference Server가 연동되는 MSA 구조로 설계되었습니다.
                                                                                                                                                                    ```mermaid
                                                                                                                                                                    graph TD
                                                                                                                                                                        User((Engineer)) -->|Query| SB[Spring Boot Backend]
                                                                                                                                                                            SB -->|Async Request| FA[FastAPI AI Engine]
                                                                                                                                                                                
                                                                                                                                                                                    subgraph "Agentic Reasoning (LangGraph)"
                                                                                                                                                                                            FA --> RA[Requirement Analyzer]
                                                                                                                                                                                                    RA --> HR[Hybrid Retriever]
                                                                                                                                                                                                            HR --> CV[Cross Verifier]
                                                                                                                                                                                                                    CV -->|Insufficient Data| HR
                                                                                                                                                                                                                            CV --> RG[Report Generator]
                                                                                                                                                                                                                                end
                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                        HR -->|Search| RS[(Redis Stack: Vector DB)]
                                                                                                                                                                                                                                            HR -->|Trace| NJ[(Neo4j: Graph DB)]

                                                                                                                                                                                                                                            ```
                                                                                                                                                                                                                                            ## 🛠️ 기술 스택 (Tech Stack)
                                                                                                                                                                                                                                            ### Backend & AI Core
                                                                                                                                                                                                                                             * **Core:** Java 17, Spring Boot 3.x, Spring WebFlux
                                                                                                                                                                                                                                              * **AI Engine:** Python 3.11, FastAPI, LangGraph, LangChain
                                                                                                                                                                                                                                               * **LLM:** GPT-4o
                                                                                                                                                                                                                                                * **Embedding:** BGE-M3 (Multi-lingual Hybrid Search)
                                                                                                                                                                                                                                                ### Database & Storage
                                                                                                                                                                                                                                                 * **Vector DB:** Redis Stack (HNSW Indexing)
                                                                                                                                                                                                                                                  * **Graph DB:** Neo4j (Technical Dependency Mapping)
                                                                                                                                                                                                                                                   * **Data Parsing:** LlamaParse (Table-aware PDF to Markdown)
                                                                                                                                                                                                                                                   ## 📊 지식 그래프 스키마 (Graph Schema)
                                                                                                                                                                                                                                                   시스템은 다음과 같은 지식 노드와 관계를 정의하여 복합적인 규격 추론을 수행합니다.
                                                                                                                                                                                                                                                    * (Component) -[:TESTED_PER]-> (Test_Method)
                                                                                                                                                                                                                                                     * (Standard) -[:INCLUDES]-> (Requirement)
                                                                                                                                                                                                                                                      * (Component) -[:PART_OF]-> (System)
                                                                                                                                                                                                                                                       * (Standard) -[:REFERENCED_BY]-> (Standard)
                                                                                                                                                                                                                                                       ## 🚀 시작하기 (Quick Start)
                                                                                                                                                                                                                                                       ### Prerequisites
                                                                                                                                                                                                                                                        * Docker & Docker Compose
                                                                                                                                                                                                                                                         * OpenAI API Key
                                                                                                                                                                                                                                                          * LlamaCloud API Key (for PDF parsing)
                                                                                                                                                                                                                                                          ### Setup & Run
                                                                                                                                                                                                                                                           1. **Repository Clone**
                                                                                                                                                                                                                                                              ```bash
                                                                                                                                                                                                                                                                 git clone https://github.com/your-id/astra-rag-agent.git
                                                                                                                                                                                                                                                                    cd astra-rag-agent
                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                          ```
                                                                                                                                                                                                                                                                           2. **Infrastructure Up (Docker)**
                                                                                                                                                                                                                                                                              ```bash
                                                                                                                                                                                                                                                                                 docker-compose up -d  # Redis Stack, Neo4j, PostgreSQL
                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                       ```
                                                                                                                                                                                                                                                                                        3. **AI Core (Python) Run**
                                                                                                                                                                                                                                                                                           ```bash
                                                                                                                                                                                                                                                                                              cd ai-core
                                                                                                                                                                                                                                                                                                 pip install -r requirements.txt
                                                                                                                                                                                                                                                                                                    uvicorn app.main:app --host 0.0.0.0 --port 8000
                                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                                          ```
                                                                                                                                                                                                                                                                                                           4. **Backend (Java) Run**
                                                                                                                                                                                                                                                                                                              ```bash
                                                                                                                                                                                                                                                                                                                 cd backend-api
                                                                                                                                                                                                                                                                                                                    ./gradlew bootRun
                                                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                                                          ```
                                                                                                                                                                                                                                                                                                                          ## 📂 프로젝트 구조 (Project Structure)
                                                                                                                                                                                                                                                                                                                          ```text
                                                                                                                                                                                                                                                                                                                          astra-rag-agent/
                                                                                                                                                                                                                                                                                                                          ├── ai-core/            # Python FastAPI & LangGraph 에이전트
                                                                                                                                                                                                                                                                                                                          │   ├── app/agents/     # 에이전트 노드 및 상태 정의
                                                                                                                                                                                                                                                                                                                          │   ├── app/services/   # Vector/Graph DB 연동 로직
                                                                                                                                                                                                                                                                                                                          │   └── main.py
                                                                                                                                                                                                                                                                                                                          ├── backend-api/        # Spring Boot 비즈니스 로직 및 API Gateway
                                                                                                                                                                                                                                                                                                                          ├── data-pipeline/      # LlamaParse 기반 PDF 전처리 스크립트
                                                                                                                                                                                                                                                                                                                          ├── docker-compose.yml  # 인프라 구성 (Redis, Neo4j, DB)
                                                                                                                                                                                                                                                                                                                          └── docs/               # 설계 문서 및 MIL-STD 샘플 데이터

                                                                                                                                                                                                                                                                                                                          ```
                                                                                                                                                                                                                                                                                                                          ## 📝 License
                                                                                                                                                                                                                                                                                                                          This project is licensed under the MIT License - see the LICENSE file for details.
                                                                                                                                                                                                                                                                                                                          **Author:** Byung-yoon Park (Konkuk University Graduate School of AI)
                                                                                                                                                                                                                                                                                                                          **Contact:** [Your Email or LinkedIn]
