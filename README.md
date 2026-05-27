# Astra-RAG-Agent

항공우주·방산 기술 규격(MIL-SPEC) 교차 검증을 위한 **지식그래프 기반 Agentic RAG** 시스템.

벡터 검색(Chroma) · 희소 검색(BM25) · 지식그래프(Neo4j)를 결합한 3중 검색 위에
LangGraph 자기수정(self-correction) 워크플로우를 얹어, 부품-규격 간 의존성을 추론하고
환각 없이 PASS/FAIL/INSUFFICIENT 판정을 내린다.

## 기술 스택

| 레이어 | 기술 |
|---|---|
| LLM (검증/리포트) | OpenAI GPT (`gpt-4o-mini` / `gpt-4o`) |
| 임베딩 | OpenAI `text-embedding-3-small` (1536차원) |
| 벡터 DB | Chroma `PersistentClient` (디스크, cosine, HNSW) |
| 희소 검색 | rank-bm25 + RRF(k=60) 결합 |
| 지식그래프 | Neo4j 5 (Component / Standard / Requirement 노드) |
| 워크플로우 | LangGraph `analyze → retrieve → verify → report` |
| 한영 확장 | YAML 용어집 기반 ko↔en 쿼리 확장 |
| PDF 파싱 | LlamaParse(표 많은 핵심 문서) + pdfplumber(나머지) |
| UI | Streamlit (채팅 + LangGraph trace + 출처 expander) |

---

## 빠른 시작

### 1) 환경 설정

```powershell
# Python 3.12 권장 (WindowsApps stub 문제 시 py -3.12 사용)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) `.env` 설정

```env
OPENAI_API_KEY=sk-...
LLAMA_CLOUD_API_KEY=llx-...   # LlamaParse 사용 시 (없으면 pdfplumber 대체)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=astra-rag-password
```

### 3) Neo4j 기동 (Docker Desktop 필요)

```powershell
docker compose up -d
# 브라우저 확인: http://localhost:7474
```

> Docker 없는 경우: [Neo4j AuraDB Free](https://neo4j.com/cloud/aura/) 계정 생성 후 `.env`의 `NEO4J_*` 교체

### 4) Smoke 테스트

```powershell
python tests/smoke_test.py
# OpenAI chat · 임베딩 · Neo4j 연결 3건 확인
```

---

## 데이터 구축 (최초 1회)

```powershell
# (1) PDF → Markdown 변환 (raw_pdfs/ 에 PDF 배치 후)
python -m src.ingestion.pdf_to_md

# (2) Markdown 분할 + frontmatter (standards/ 생성)
python -m src.ingestion.split_standards

# (3) 벡터 DB 적재 (~4000 청크)
python -m src.ingestion.embed_and_store

# (4) Neo4j 지식그래프 구축
python -m src.ingestion.build_kg
```

---

## 실행

```powershell
# Streamlit 웹 UI
streamlit run src/ui/app.py

# CLI 테스트 (기본 3개 시나리오)
python -m tests.run_graph

# 회귀 테스트
python -m tests.regression
python -m tests.regression S2_radiation  # 특정 시나리오만
```

---

## 시연 시나리오

| ID | 이름 | 핵심 기능 |
|---|---|---|
| S1 | 진동 규격 연쇄 위반 탐지 | Neo4j `DEPENDS_ON` 재귀 순회 (GraphRAG) |
| S2 | COTS 방사선 교차검증 | 벡터+BM25 하이브리드 + PASS/FAIL 판정 |
| S3 | 염수분무 코팅 (한영) | ko→en 쿼리 확장 (`염수 분무` → `Salt Fog`) |
| S4 | Self-correction 데모 | LangGraph `verify→retrieve` 재진입 후 INSUFFICIENT 종료 |

---

## 아키텍처

```
[Streamlit UI]
    ↓ graph.stream()
[LangGraph 워크플로우]
  analyze  → 질의 정규화, 부품 식별, 카테고리 감지, 한영 확장
  retrieve → Neo4j DEPENDS_ON 순회 + Dense(Chroma) + BM25 → RRF 결합
  verify   → 수치 충족성 판단 (그래프 판정 있으면 즉시 SUFFICIENT)
           → INSUFFICIENT 시 retrieve 재진입 (최대 2회)
  report   → PASS/FAIL/정보 없음 + 출처 인용 한국어 리포트
    ↓
[Chroma DB]  [BM25 인메모리]  [Neo4j 그래프]
    ↓              ↓                ↓
  벡터 4033개   청크 전체       Component/Standard/Requirement
                              DEPENDS_ON / TESTED_PER / HAS_REQUIREMENT
```

---

## 디렉토리 구조

```
astra-rag-agent/
  data/
    standards/         분할된 규격 MD (frontmatter: standard/method/category)
    components/        부품 스펙 JSON (depends_on / compliance / specs)
    glossary/          한영 용어집 YAML (30+ 쌍)
  src/
    ingestion/         PDF변환 · 청킹 · 임베딩 적재 · 그래프 구축
    retrieval/         벡터 · BM25 · 하이브리드(RRF) · 그래프 검색
    workflow/          LangGraph state / nodes / builder
    generation/        리포트 생성 · 프롬프트 템플릿
    ui/app.py          Streamlit 앱
    config.py          환경변수 · 클라이언트 싱글톤 (lru_cache)
    utils.py           UTF-8 콘솔 설정
  tests/
    smoke_test.py      OpenAI·Neo4j 연결 확인
    run_graph.py       CLI 워크플로우 실행
    regression.py      demo_questions.yaml 기반 자동 회귀
    demo_questions.yaml 시나리오별 expect_keywords · expect_sources
  docker-compose.yml   Neo4j 5 community 컨테이너
  chroma_db/           (gitignore) Chroma 벡터 저장소
  raw_pdfs/            (gitignore) 원본 PDF
```

---

## 설계 대비 구현 변경 사항

| 설계 항목 | 설계 범위 | 구현 결과 |
|---|---|---|
| GraphRAG (Neo4j) | 핵심 포함 | ✅ 유지 — DEPENDS_ON 재귀 순회 구현 |
| 하이브리드 검색 | Dense+BM25+RRF | ✅ 완전 구현 (한영 확장 포함) |
| LangGraph self-correction | verify→retry 루프 | ✅ 구현 (MAX_RETRIES=2) |
| MSA / Spring Boot | 향후 발전 | ❌ 제외 (3주 범위 초과) |
| 멀티에이전트 | 향후 발전 | ❌ 제외 (단일 LangGraph 워크플로우) |
| LlamaParse | 810H 핵심 메서드 | ⚠️ 검증 완료, 전체 실행은 선택적 |
