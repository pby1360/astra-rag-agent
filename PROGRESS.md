# Astra-RAG-Agent 진행 현황

> 최종 업데이트: 2026-05-26  
> 이 문서는 세션 간 작업 연속성을 위한 현황 기록이다. 빌드 전략은 `설계.md` / `구현계획.md` 참조.

---

## 완료된 Phase

| Phase | 내용 | 상태 |
|---|---|---|
| 1 | 환경 셋업 (venv, .env, docker-compose, git) | 완료 |
| 2 | 데이터 준비 (standards MD, components JSON, glossary YAML) | 완료 |
| 3 | 최소 RAG (loader → embed → vector search → LLM 생성) | 완료 |
| 4 | Neo4j 지식그래프 구축 (build_kg, graph_retriever) | 완료 |
| 5 | Streamlit UI (채팅, 출처 expander, 판정 배너, trace) | 완료 |
| 6 | 하이브리드 검색 (BM25 + Dense + RRF + 한영 쿼리 확장) | 완료 |
| 7 | LangGraph 도입 (analyze→retrieve→verify→report, self-correction) | 완료 |
| 8 | 안정화 (로깅, 프롬프트 3유형, 회귀테스트, SpaceX 시나리오) | 진행 중 |

---

## 현재 구현 상태 (세부)

### 데이터 현황

**`data/standards/`** — 분할된 규격 MD (frontmatter 포함)
- MIL-STD-883 Method 1001~1019 (19개 파일)
- MIL-STD-810H Method 500~527 (진동, 온도, 충격 등)
- MIL-STD-461G (전자기적합성)
- SMC-S-016 (우주 방사선)
- `spacex_falcon9_starlink_environment.md` (추가됨, 2026-05-26)

**`data/components/`** — 부품 JSON (12개)

| 파일 | 부품번호 | 종류 | 실부품 |
|---|---|---|---|
| `tps7h1101_sp.json` | TPS7H1101-SP | power (LDO) | O |
| `samv71q21rt.json` | SAMV71Q21RT | processor | O |
| `tps7a4501_sp.json` | TPS7A4501-SP | power (LDO) | O (추가됨) |
| `rad750.json` | RAD750 | processor | O (추가됨) |
| `xqr5vfx130.json` | XQR5VFX130 | fpga | O (추가됨) |
| `lv_engine_x.json` | LV-ENGINE-X | assembly | 가상 |
| `starlink_sat_gen2.json` | STARLINK-SAT-GEN2 | assembly | 가상 (추가됨) |
| `adcs_pcb_v1.json` | ADCS-PCB-V1 | board | 가상 |
| `avionics_mod_e.json` | AVIONICS-MOD-E | module | 가상 |
| `antenna_deploy_k1.json` | ANTENNA-DEPLOY-K1 | mechanism | 가상 |
| `vib_damper_b.json` / `fuel_valve_c.json` / `sensor_brkt_d.json` | 하위 부품들 | 각종 | 가상 |

**`data/requirements.json`** — 규격 임계치 (5개)

| ID | 기준 | 파라미터 | 임계치 |
|---|---|---|---|
| REQ-VIB-CAT24 | MIL-STD-810H M514 | vibration_grms | ≥ 14.1 g rms |
| REQ-TID-GEO15 | MIL-STD-883 M1019 | tid_tolerance_krad_si | ≥ 50 krad |
| REQ-SALTFOG-COAT | MIL-STD-810H M509 | corrosion_coating_thickness_um | ≥ 20 um |
| REQ-TID-STARLINK5 | SpaceX Starlink LEO | tid_tolerance_krad_si | ≥ 15 krad |
| REQ-VIB-FALCON9 | SpaceX Falcon 9 PAF | vibration_grms | ≥ 8.8 g rms |

**`data/glossary/ko_en_terms.yaml`** — 한영 용어 30+ (쿼리 확장 전용, Chroma/BM25에 미포함)

### DB 현황

**Neo4j** (docker-compose로 bolt://localhost:7687 실행)
- 노드: Component 13, Standard 9, Requirement 5
- 엣지: DEPENDS_ON 9, TESTED_PER 19, HAS_REQUIREMENT 5

**Chroma** (`chroma_db/`, `astra_specs` 컬렉션)
- 총 4006개 청크 (text-embedding-3-small 1536차원)

### 핵심 소스 파일

```
src/
  config.py              경로·모델·클라이언트 싱글톤 (lru_cache)
  logging_config.py      파일+stdout 듀얼 로깅 (astra_rag.log)
  utils.py               UTF-8 stdout 강제
  ingestion/
    loader.py            MD/JSON 로딩 → 청크 (glossary 제외)
    embed_and_store.py   OpenAI 임베딩 → Chroma 적재
    build_kg.py          components JSON → Neo4j (MERGE 멱등)
  retrieval/
    vector_retriever.py  Chroma Dense 검색
    bm25_retriever.py    rank_bm25 희소 검색
    hybrid.py            RRF 결합 + 한영 쿼리 확장
    graph_retriever.py   Neo4j Cypher 의존성 순회 / 직접 스펙 조회
  workflow/
    state.py             RAGState TypedDict
    nodes.py             analyze / retrieve / verify / report 노드
    builder.py           StateGraph (MAX_RETRIES=2, recursion_limit=12)
  generation/
    generator.py         OpenAI 챗 완성 (gpt-4o-mini / gpt-4o)
    prompts.py           시스템·유저 프롬프트 (유형 A/B/C 판정 형식)
  ui/app.py              Streamlit 채팅 + 판정 배너 + 출처 expander
```

### LangGraph 흐름

```
START → analyze → retrieve → verify ─(충분 or 재시도 소진)→ report → END
                               └─(부족 & retry_count < 2)→ retrieve
```

- **analyze**: 한영 쿼리 확장, 부품번호 인식, 카테고리 감지 (vibration/radiation/corrosion)
- **retrieve**: Neo4j 의존성 순회(assembly) 또는 직접 스펙 조회(leaf) + 하이브리드 RAG
- **verify**: graph verdict(PASS/FAIL) 존재 시 LLM 호출 없이 SUFFICIENT / 없으면 LLM 판단
- **report**: 유형 A(규격 조회) / B(적합성 판정 테이블) / C(부품 설명) 형식으로 한국어 리포트

### 프롬프트 형식 규칙 (`src/generation/prompts.py`)

| 유형 | 질문 패턴 | 출력 형식 |
|---|---|---|
| A | "얼마야?", "기준이 뭐야?", "얼마로 해야 해?" | 기준값 직접 안내, PASS/FAIL 없음 |
| B | "충족해?", "통과해?", "기준 미달인 것은?" | 마크다운 테이블 (2개↑) 또는 판정:수치 (1개) |
| C | "설명해봐", "뭐야?", "어떤 부품이야?" | 개요·주요스펙·용도·근거, PASS/FAIL 없음 |

---

## 시연 시나리오 현황 (`tests/demo_questions.yaml`)

| ID | 시나리오 | 핵심 검증 포인트 | 회귀 상태 |
|---|---|---|---|
| S1_vibration_cascade | LV-ENGINE-X 진동 규격 상향 → 하위 부품 FAIL 탐지 | GraphRAG DEPENDS_ON 순회 | 통과 |
| S1b_vibration_single | FUEL-VALVE-C 진동 단일 판정 | 직접 스펙 조회 | 미확인 (S1b known issue) |
| S2_radiation | TPS7H1101-SP GEO 15년 TID 판정 | 부품 스펙 vs 규격 비교 | 통과 |
| S3_saltfog_ko | 염수분무 코팅 두께 기준 (한국어) | 한영 쿼리 확장 → 유형 A 출력 | 통과 |
| S4_self_correction | 존재하지 않는 부품 → 정보 없음 | Self-correction 루프 종료 | 통과 |
| S5_spacex_starlink_tid | STARLINK-SAT-GEN2 하위 부품 TID 종합 판정 | GraphRAG + SpaceX 규격 | 통과 |
| S6_spacex_falcon9_vibration | Falcon 9 랜덤 진동 기준 조회 | 유형 A, SpaceX 표준 문서 | 통과 |
| S7_spacex_rad750_description | RAD750 설명 | 유형 C, PASS/FAIL 없음 | 통과 |

회귀 실행:
```
.venv\Scripts\python.exe -m tests.regression          # 전체
.venv\Scripts\python.exe -m tests.regression S5_spacex_starlink_tid  # 단건
```

---

## 알려진 이슈 / 미결 항목

| 항목 | 내용 | 우선순위 |
|---|---|---|
| S1b AVIONICS-MOD-E | 직접 단일 부품 진동 판정 시 간헐적 정보 없음 반환. 그래프 leaf 부품 스펙 조회 로직 확인 필요. | 낮음 |
| Streamlit 터미널 로그 | `setup_logging()`이 Streamlit stdout을 캡처해 터미널에 직접 출력 안 됨. `astra_rag.log` 파일로 확인 가능. | 낮음 |
| Phase 8 남은 항목 | 안전망(LLM 타임아웃 30초), README 완성, 발표용 스크린샷 | 중간 |

---

## 서비스 실행 방법

```powershell
# 1. Neo4j 컨테이너 실행 (최초 1회 또는 재부팅 후)
docker-compose up -d

# 2. 가상환경 활성화
.\.venv\Scripts\Activate.ps1

# 3. Streamlit UI 실행
streamlit run src/ui/app.py

# DB 재구축이 필요한 경우 (부품/규격 데이터 변경 시)
## Neo4j 재적재
.\.venv\Scripts\python.exe -m src.ingestion.build_kg

## Chroma 재적재 (주의: Python 프로세스 모두 종료 후)
## Get-Process python*, streamlit* | Stop-Process -Force
.\.venv\Scripts\python.exe -m src.ingestion.embed_and_store
```

---

## 다음으로 할 작업 (Phase 8 잔여)

1. **LLM 타임아웃 안전망**: `generate_answer()` / `is_sufficient()`에 30초 타임아웃 + 예외 시 청크 원문 직접 반환
2. **README.md 완성**: 설치, 실행, 시나리오 설명, 아키텍처 다이어그램
3. **발표 자료**: 시나리오별 스크린샷 + trace 캡처, 설계.md 대비 구현 결과 섹션
4. **S1b 이슈 분석**: `graph_retriever.get_component_as_document()` 단일 부품 경로 재점검
